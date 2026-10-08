"""Train a model, or refine a trained checkpoint with confusion replay.

  python tools/train.py --config configs/vision/ours_stage1.yaml                 # two-view, size-aware training
  python tools/train.py --config configs/vision/ours.yaml \
      --opts refine.init_checkpoint=outputs/vision_ours_stage1/best.pt refine.replay_file=data/replay/vision/s42.json
  overrides: --opts train.epochs=50 seed=1          (seed != 42 -> output dir <experiment_id>_s<seed>)

Checkpoint selection = best validation pixel AP of the main head, for every method. The test split is never
used here. Every run gets a unique run_id; an existing output directory is never reused (--overwrite moves it
aside). Refinement checks that the replay file was mined from exactly the init checkpoint.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sds.data import NativeCropDataset, NativePointCropDataset, SegDataset, read_split  # noqa: E402
from sds.engine import evaluate_loader, to_device  # noqa: E402
from sds.losses import seg_loss  # noqa: E402
from sds.mining import check_replay_provenance  # noqa: E402
from sds.models import build_model  # noqa: E402
from sds.utils import (Logger, environment_info, file_sha256, load_config, new_run_id, prepare_out_dir,  # noqa: E402
                       rel, resolve, save_json, save_yaml, set_seed, splits_sha256, worker_init_fn)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--exp-id", default=None, help="default: cfg.experiment_id (+ _s<seed> if seed != 42)")
    ap.add_argument("--max-iters", type=int, default=None, help="debug: stop after N iterations")
    ap.add_argument("--overwrite", action="store_true", help="move an existing run directory aside and start fresh")
    return ap.parse_args()


def make_loader(ds, bs, shuffle, cfg, seed, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=cfg["train"].get("num_workers", 4),
                      pin_memory=True, drop_last=drop_last, worker_init_fn=worker_init_fn, generator=g,
                      persistent_workers=cfg["train"].get("num_workers", 4) > 0)


def infinite(loader):
    assert len(loader) > 0, "empty loader"
    while True:
        yield from loader


def build_replay_loaders(cfg, seed):
    """(small-defect crop iterator, replayed-background crop iterator, sizes) or None without replay.

    The replay file (tools/mine_native_fp.py) holds points in ORIGINAL pixel coordinates: centres of small
    defects ("pos") and of background regions per replay type ("native_fp" = the model's own native-resolution
    false positives, "random" = random background, the control)."""
    rc = cfg["refine"]
    if rc.get("replay_type", "none") == "none":
        return None
    with open(resolve(rc["replay_file"])) as f:
        rep = json.load(f)
    check_replay_provenance(rep, cfg, resolve(rc["init_checkpoint"]))
    rows = {}
    for split in rep["splits"]:
        for r in read_split(split):
            rows[r["id"]] = r
    mk = lambda es: [{"row": rows[e["id"]], "cy": e["cy"], "cx": e["cx"]} for e in es]  # noqa: E731
    bg_key = rc["replay_type"]
    assert bg_key in rep["bg"], f"{bg_key} not in replay file ({list(rep['bg'])})"
    pos_ds = NativePointCropDataset(mk(rep["pos"]), crop=rep["crop"], aug=cfg["train"].get("aug"))
    bg_ds = NativePointCropDataset(mk(rep["bg"][bg_key]), crop=rep["crop"], aug=cfg["train"].get("aug"))
    half = rc.get("replay_batch_size", cfg["train"]["batch_size"]) // 2
    # drop_last with fewer samples than a half batch would yield no batch -> infinite() would spin forever
    for name, ds in (("positive", pos_ds), ("background", bg_ds)):
        if len(ds) < half:
            raise SystemExit(f"replay {name} set has {len(ds)} crops < half batch {half}")
    return (infinite(make_loader(pos_ds, half, True, cfg, seed + 1, drop_last=True)),
            infinite(make_loader(bg_ds, half, True, cfg, seed + 2, drop_last=True)),
            len(pos_ds), len(bg_ds))


def random_crop(batch, size):
    """Per-sample random crop of all full-size tensors (crop recipe of the small-target baselines)."""
    h, w = batch["image"].shape[-2:]
    n = batch["image"].shape[0]
    ys, xs = torch.randint(0, h - size + 1, (n,)), torch.randint(0, w - size + 1, (n,))
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() == 4 and v.shape[-2:] == (h, w):
            out[k] = torch.stack([v[i, :, y:y + size, x:x + size] for i, (y, x) in enumerate(zip(ys, xs))])
        else:
            out[k] = v
    return out


def main():
    args = parse()
    cfg = load_config(args.config, args.opts)
    seed = cfg.get("seed", 42)
    exp_id = args.exp_id or cfg["experiment_id"] + (f"_s{seed}" if seed != 42 else "")
    out_dir = prepare_out_dir(resolve(cfg.get("output_root", "outputs")) / exp_id, args.overwrite)
    log = Logger(out_dir / "train.log")
    cfg["experiment_id_resolved"] = exp_id
    cfg["run_id"] = new_run_id()
    prov = {"run_id": cfg["run_id"], "train_split_sha": splits_sha256(cfg["data"]["train_split"]),
            "val_split_sha": splits_sha256(cfg["data"]["val_split"])}
    refine = cfg.get("refine", {}).get("enabled", False)
    if refine:
        rc = cfg["refine"]
        prov["init_checkpoint"] = rel(resolve(rc["init_checkpoint"]))
        prov["init_checkpoint_sha"] = file_sha256(rc["init_checkpoint"])
        if rc.get("replay_type", "none") != "none":
            prov["replay_file"] = rel(resolve(rc["replay_file"]))
            prov["replay_file_sha"] = file_sha256(rc["replay_file"])
    cfg["provenance"] = prov
    save_yaml(cfg, out_dir / "config.yaml")
    save_json(environment_info(), out_dir / "environment.json")
    set_seed(seed, cfg["train"].get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[{exp_id}] device={device} cfg={args.config}")

    data, tr = cfg["data"], cfg["train"]
    # "auto" small-defect area = 33rd percentile of the training (Core) defect-component area ratios -- the same
    # edge that defines the small-defect metrics -- times the input area
    if any(isinstance(tr.get(k), dict) and "auto" in tr[k].values() for k in ("size_weight", "size_split")):
        from tools.evaluate import size_edges

        area_auto = size_edges(data["core_split"])[1][0] * data["input_size"][0] * data["input_size"][1]
        for k, f in (("size_weight", "a_ref"), ("size_split", "area_px")):
            if isinstance(tr.get(k), dict) and tr[k].get(f) == "auto":
                tr[k][f] = float(area_auto)
        log(f"small-defect area (auto) = {area_auto:.0f} px at input {data['input_size']}")
        save_yaml(cfg, out_dir / "config.yaml")
    train_ds = SegDataset(data["train_split"], data["input_size"], train=True, aug=tr.get("aug"),
                          mask_resize_threshold=data.get("mask_resize_threshold", 0.5),
                          size_weight=tr.get("size_weight"), size_split=tr.get("size_split"))
    val_ds = SegDataset(data["val_split"], data["input_size"], train=False,
                        mask_resize_threshold=data.get("mask_resize_threshold", 0.5))
    train_loader = make_loader(train_ds, tr["batch_size"], True, cfg, seed, drop_last=True)
    val_loader = make_loader(val_ds, tr.get("eval_batch_size", tr["batch_size"]), False, cfg, seed)
    log(f"train={len(train_ds)} val={len(val_ds)} iters/epoch={len(train_loader)}")

    model = build_model(cfg).to(device)
    if refine:
        ck = torch.load(resolve(cfg["refine"]["init_checkpoint"]), map_location="cpu")
        model.load_state_dict(ck["model"])
        log(f"refine: init from {cfg['refine']['init_checkpoint']} (epoch {ck.get('epoch')})")
    replay = build_replay_loaders(cfg, seed) if refine else None
    if replay:
        log(f"replay: type={cfg['refine']['replay_type']} pos={replay[2]} bg={replay[3]}")
    full_per_replay = cfg.get("refine", {}).get("full_per_replay", 3)
    # refinement without replay (control): the replay slots are filled with extra whole-image batches, so every
    # refinement variant gets the same number of optimisation steps
    extra_full = None
    if refine and not replay and cfg["refine"].get("match_budget", True):
        extra_full = infinite(make_loader(train_ds, tr["batch_size"], True, cfg, seed + 3, drop_last=True))
        log("refine: no replay, budget-matched with extra whole-image steps")

    # second view: native-resolution crops, one crop batch every `every` whole-image batches
    # (zoom.only: native crops only, the regime of the small-target baselines)
    zoom_iter, zc = None, tr.get("zoom")
    if zc:
        zds = NativeCropDataset(train_ds.rows, crop=zc.get("crop", 384), p_defect=zc.get("p_defect", 0.5),
                                aug=tr.get("aug"))
        zoom_iter = infinite(make_loader(zds, zc.get("batch", tr["batch_size"]), True, cfg, seed + 4, drop_last=True))
        log(f"native crops: {zc}")

    groups = [{"params": list(model.encoder_parameters()), "lr": tr["lr"]},
              {"params": list(model.head_parameters()), "lr": tr["lr"] * tr.get("head_lr_mult", 10.0)}]
    groups = [g for g in groups if g["params"]]
    if tr.get("optimizer", "adamw") == "adagrad":  # small-target baselines with their authors' recipe
        opt = torch.optim.Adagrad(groups, lr=tr["lr"])
    else:
        opt = torch.optim.AdamW(groups, lr=tr["lr"], weight_decay=tr.get("weight_decay", 0.01))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]

    accum = tr.get("grad_accum", 1)
    steps_per_epoch = len(train_loader)
    if replay or extra_full is not None:
        steps_per_epoch += steps_per_epoch // full_per_replay
    if zoom_iter is not None and not zc.get("only"):
        steps_per_epoch += len(train_loader) // zc.get("every", 2)
    total_steps = max(1, tr["epochs"] * steps_per_epoch // accum)
    warmup = int(tr.get("warmup_ratio", 0.05) * total_steps)

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        if tr.get("scheduler", "poly") == "cosine":
            return 0.5 * (1 + math.cos(math.pi * p))
        return (1 - p) ** tr.get("poly_power", 1.0)

    amp = tr.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    lam = cfg["model"].get("lambda_aux", 1.0)

    best_ap, best_epoch, it, opt_step = -1.0, -1, 0, 0
    hist = open(out_dir / "metrics.jsonl", "a")
    t0 = time.time()
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        if hasattr(model, "set_epoch"):  # MSHNet: warm-up epochs of its SLS loss
            model.set_epoch(epoch)
        run = {}
        n_full = 0
        for full_batch in train_loader:
            batches = [full_batch]
            n_full += 1
            if zc and zc.get("only"):
                batches = [next(zoom_iter)]
            if replay and n_full % full_per_replay == 0:
                pb, bb = next(replay[0]), next(replay[1])
                batches.append({k: torch.cat([pb[k], bb[k]]) for k in ("image", "mask", "valid")})
            elif extra_full is not None and n_full % full_per_replay == 0:
                batches.append(next(extra_full))
            if zoom_iter is not None and not zc.get("only") and n_full % zc.get("every", 2) == 0:
                batches.append(next(zoom_iter))
            for batch in batches:
                batch = to_device(batch, device)
                if tr.get("random_crop") and "idx" not in batch:  # whole-image batches only
                    batch = random_crop(batch, tr["random_crop"])
                with torch.autocast("cuda", enabled=amp):
                    out = model(batch["image"], return_aux=True)
                spec = ((set(tr["size_split"]["heads"]), batch["small_mask"], batch["small_valid"])
                        if "small_mask" in batch else None)
                if getattr(model, "custom_loss", False):  # the authors' loss of a baseline
                    loss = model.loss_fn(out, batch["mask"].float())
                    logs = {"loss_main": loss.detach(), "loss": loss.detach()}
                else:
                    loss, logs = seg_loss(out, batch["mask"], batch["valid"], lam, batch.get("pos_weight"), spec)
                scaler.scale(loss / accum).backward()
                it += 1
                if it % accum == 0:
                    for g in opt.param_groups:
                        g["lr"] = g["base_lr"] * lr_factor(opt_step)
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("grad_clip", 1.0))
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    opt_step += 1
                for k, v in logs.items():
                    run[k] = run.get(k, 0.0) + float(v)
                run["_n"] = run.get("_n", 0) + 1
                if not math.isfinite(float(loss)):
                    log("non-finite loss, abort")
                    sys.exit(1)
            if args.max_iters and it >= args.max_iters:
                break
        n = run.pop("_n", 1)
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], **{k: v / n for k, v in run.items()},
               "time_min": (time.time() - t0) / 60}

        if epoch % tr.get("eval_interval", 5) == 0 or epoch == tr["epochs"] or args.max_iters:
            evs = evaluate_loader(model, val_loader, val_ds, device, ("main",), amp, keep_per_image=False)
            ap = evs["main"].pixel_ap()
            rec["val_pixel_ap"] = ap
            if ap > best_ap:
                best_ap, best_epoch = ap, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_pixel_ap": ap, "cfg": cfg},
                           out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, out_dir / "last.pt")
        hist.write(json.dumps(rec) + "\n")
        hist.flush()
        log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if args.max_iters and it >= args.max_iters:
            break

    minutes = (time.time() - t0) / 60
    parent = 0.0  # refinement: add the training time of the init checkpoint (total training cost)
    if refine:
        ps = resolve(cfg["refine"]["init_checkpoint"]).parent / "train_summary.json"
        if ps.exists():
            parent = json.load(open(ps)).get("train_minutes_total", 0.0)
    save_json({"experiment_id": exp_id, "run_id": cfg["run_id"], "best_epoch": best_epoch,
               "best_val_pixel_ap": best_ap, "epochs": tr["epochs"], "train_minutes": minutes,
               "train_minutes_total": minutes + parent, "completed": True, **prov}, out_dir / "train_summary.json")
    log(f"done. best val AP={best_ap:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
