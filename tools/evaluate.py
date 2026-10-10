"""Evaluate a trained experiment.

  python tools/evaluate.py --exp outputs/vision_segformer_b0                # validation only
  python tools/evaluate.py --exp outputs/vision_segformer_b0 --test [--tol 3]   # + test split

Protocol
* thresholds (best-F1 and recall>=0.9) are selected on the validation split only and
  applied unchanged to the test split; at the "val R90" threshold the test recall is
  reported next to the FP area because it is not fixed to 0.9 on test;
* size groups use Q33/Q66 of the training (Core) split (image foreground ratio for ratio_*,
  defect-component area for defect_*), identical for every method;
* per-category AP (+ macro mean) when data.category_from_id is true (id prefix before '_');
* one row per head and split is appended to results/experiments.csv with run_id,
  checkpoint hash, split hash and git commit (with --test only test rows are added).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sds.data import SegDataset, read_split  # noqa: E402
from sds.data.dataset import load_mask  # noqa: E402
from sds.engine import evaluate_loader  # noqa: E402
from sds.metrics import metrics_at, select_thresholds  # noqa: E402
from sds.models import build_model  # noqa: E402
from sds.utils import (enable_tf32, append_result, environment_info, file_sha256, load_config, rel, resolve,  # noqa: E402
                       save_json, splits_sha256)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment output dir")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--heads", nargs="*", default=["main"], help="main and/or stage heads (s1..s4)")
    ap.add_argument("--test", action="store_true", help="also evaluate the official test split (final only!)")
    ap.add_argument("--save-probs", action="store_true", help="save float16 probability maps (original res)")
    ap.add_argument("--criterion", default="f1", choices=["f1", "iou"])
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true", help="evaluate a run without train_summary.json")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out-suffix", default="", help="write metrics_<split>_<suffix>.json instead of the main file "
                                                       "(e.g. adding Pd/Fa to old runs without overwriting them)")
    ap.add_argument("--tol", type=int, default=0,
                    help="boundary tolerance in original px (sds.metrics.tolerance_keep), same for every method; "
                         "results go to metrics_<split>_tol<k>.json, rows get eval_head '<head>@tol<k>'")
    return ap.parse_args()


def size_edges(core_split):
    """Q33/Q66 of (a) image foreground ratio and (b) defect-component area ratio of the defect images of the
    training (Core) split, at original resolution."""
    rows = [x for x in read_split(core_split) if x["fg_ratio"] > 0]
    comp = []
    for x in rows:
        g = load_mask(x["mask"], (x["height"], x["width"]))
        lab, n = ndimage.label(g, structure=np.ones((3, 3)))
        comp += (np.asarray(ndimage.sum(g, lab, np.arange(1, n + 1))) / g.size).tolist()
    q = lambda a: (float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3)))  # noqa: E731
    return q(np.array([x["fg_ratio"] for x in rows])), q(np.array(comp))


def main():
    args = parse()
    exp = resolve(args.exp)
    cfg = load_config(exp / "config.yaml")
    summ_path = exp / "train_summary.json"
    if not summ_path.exists() and not args.allow_incomplete:
        raise SystemExit(f"{exp} has no train_summary.json (training incomplete?); use --allow-incomplete to force")
    summ = json.load(open(summ_path)) if summ_path.exists() else {}
    exp_id = cfg.get("experiment_id_resolved", exp.name)
    enable_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt_path = exp / args.ckpt
    ck = torch.load(ckpt_path, map_location="cpu")
    if ck.get("cfg", {}).get("run_id") != cfg.get("run_id"):
        raise SystemExit(f"checkpoint run_id {ck.get('cfg', {}).get('run_id')} != config run_id {cfg.get('run_id')}")
    model.load_state_dict(ck["model"])
    heads = list(args.heads)
    data, tr = cfg["data"], cfg["train"]
    edges, comp_edges = size_edges(data["core_split"])
    cat_fn = (lambda i: str(i).split("_")[0]) if data.get("category_from_id") else None
    (exp / "eval").mkdir(exist_ok=True)
    ck_sha = file_sha256(ckpt_path)
    env = environment_info()
    git, dirty = env.get("git_commit"), env.get("git_dirty")
    if dirty:
        print("[warn] uncommitted changes: commit before producing paper numbers (git_dirty=True is recorded)")

    splits = [("val", data["val_split"])] + ([("test", data["test_split"])] if args.test else [])
    results, thresholds = {}, {}
    for split_name, split_csv in splits:
        ds = SegDataset(split_csv, data["input_size"], mask_resize_threshold=data.get("mask_resize_threshold", 0.5))
        loader = DataLoader(ds, batch_size=tr.get("eval_batch_size", 8), shuffle=False,
                            num_workers=tr.get("num_workers", 4), pin_memory=True)
        hook = None
        if args.save_probs:
            pdir = exp / "predictions" / split_name
            pdir.mkdir(parents=True, exist_ok=True)

            def hook(row, per, gt, pdir=pdir):
                np.savez_compressed(pdir / f"{row['id']}.npz",
                                    **{f"p{h}": p.half().cpu().numpy() for h, p in per.items()})

        evs = evaluate_loader(model, loader, ds, device, heads, tr.get("amp", True), on_image=hook,
                              size_edges=edges, comp_edges=comp_edges, category_fn=cat_fn, tol=args.tol)
        for h, ev in evs.items():
            st = ev.state()
            np.savez_compressed(exp / "eval" / f"{split_name}_{h}{f'_tol{args.tol}' if args.tol else ''}{f'_{args.out_suffix}' if args.out_suffix else ''}.npz", **st)
            if split_name == "val":
                thresholds[h] = select_thresholds(st, args.criterion, 0.9)
            k, k90 = thresholds[h]
            m = metrics_at(st, k, k90, edges, comp_edges)
            results.setdefault(str(h), {})[split_name] = m
            show = ["pixel_ap", "image_ap", "foreground_iou", "precision", "recall", "normal_fp_area",
                    "fp_at_val_r90", "recall_at_val_r90", "defect_small_ap", "cat_macro_ap"]
            print(f"[{exp_id}] head={h} {split_name}: " + " ".join(f"{a}={m[a]:.4f}" for a in show if a in m))

            if args.no_csv or (args.test and split_name == "val"):
                continue  # with --test the val pass only re-derives thresholds
            eff = json.load(open(exp / "efficiency.json")) if (exp / "efficiency.json").exists() else {}
            mc, rc = cfg["model"], cfg.get("refine", {})
            append_result({
                **m, "experiment_id": exp_id, "run_id": cfg.get("run_id"), "project": "SDS",
                "model": mc.get("head", mc.get("type")),
                "backbone": "mit-b0" if mc.get("type", "segformer") == "segformer" else mc.get("encoder"),
                "dataset": data["name"], "subset": data.get("subset", ""), "seed": cfg.get("seed", 42),
                "split_version": data.get("split_version", ""), "train_split_sha": splits_sha256(data["train_split"]),
                "input_height": data["input_size"][0], "input_width": data["input_size"][1],
                "batch_size": tr["batch_size"], "effective_batch_size": tr["batch_size"] * tr.get("grad_accum", 1),
                "optimizer": "AdamW", "learning_rate": tr["lr"], "weight_decay": tr.get("weight_decay", 0.01),
                "epochs": tr["epochs"], "best_epoch": summ.get("best_epoch", ck.get("epoch")), "loss": "bce+dice",
                "replay_type": rc.get("replay_type", "") if rc.get("enabled") else "",
                "eval_head": h + (f"@tol{args.tol}" if args.tol else ""), "eval_split": split_name, "threshold_source": f"val_{args.criterion}",
                "params_m": eff.get("params_m", ""), "flops_g": eff.get("flops_g", ""),
                "latency_ms": eff.get("latency_ms", ""), "train_minutes_total": summ.get("train_minutes_total", ""),
                "checkpoint": rel(ckpt_path), "checkpoint_sha": ck_sha, "git_commit": git, "git_dirty": dirty, "notes": args.notes,
            })
    save_json({"run_id": cfg.get("run_id"), "checkpoint_sha": ck_sha, "size_edges": edges,
               "component_size_edges": comp_edges, "thresholds": {str(k): v for k, v in thresholds.items()},
               "results": results}, exp / ((("metrics_test" if args.test else "metrics_val")
                                            + (f"_tol{args.tol}" if args.tol else "")
                                            + (f"_{args.out_suffix}" if args.out_suffix else "")) + ".json"))


if __name__ == "__main__":
    main()
