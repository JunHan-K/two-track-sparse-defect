"""MagNet (Huynh et al., "Progressive Semantic Segmentation", CVPR 2021; github.com/VinAIResearch/MagNet, AGPL-3.0)
retrained on OUR fixed VISION splits: a direct comparison with a global-local, coarse-to-fine method.

  python tools/external/magnet_seg.py --seed 42                 # backbone + refinement + validation
  python tools/external/magnet_seg.py --seed 42 --eval-only [--tol 3] [--test] [--fast]

The authors' code is used unchanged from third_party/MagNet (not in git: AGPL): ResnetFPN backbone (ImageNet
ResNet50), RefinementMagNet, OhemCrossEntropy, patch geometry (get_patch_coords, ensemble, certainty and
uncertain-point sampling, point_sample) and their inference loop (test.py), re-implemented here only to accept
images of different sizes. Their recipe (DeepGlobe, the setting closest to ours: square images, 3 scales,
factor 2, the backbone trained on patches) is kept:
  backbone   ResnetFPN, CE (no OHEM), SGD lr 1e-3, momentum 0.9, wd 5e-4, poly decay, horizontal flip,
             random patches of a random scale; 100 epochs x 170 iterations (batch 4) ~ their 484 x 38 (batch 12).
  refinement backbone frozen; RefinementMagNet(use_bn=True); OHEM CE (thres 0.7, min_kept 100k); SGD lr 1e-3,
             momentum 0.9, wd 5e-4, MultiStepLR [10, 20, 30, 40, 45] x0.1, 50 epochs, batch 8 (their train.py).
  inference  coarse-to-fine over all scales, all patches refined (MagNet; --fast: 3 most uncertain patches per
             scale = MagNet-Fast), n_points 0.75 (0.9 for fast), median smoothing 11, as their scripts.
Adaptation to VISION (images of 480-3840 px): each image is letterboxed into a square canvas of side C = max(H, W)
(padding = ignore label in training, cropped away in evaluation). Scales: 1024 (= the input size of every other
model) then x2 per level while below C, then C (native resolution). Patch (crop) = input size = 1024, i.e. the
coarsest scale is one patch, as 612 = their coarsest DeepGlobe scale. Two classes (background, defect) with
softmax, as in their code; defect probability = softmax channel 1. Training data: Core A + Mining B (the
labelled data of the other baselines). The last epoch is used (no checkpoint selection), as in the authors' code.
Evaluation: our evaluator (sds.metrics) at the original resolution, thresholds selected on validation.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MAGNET = ROOT / "third_party" / "MagNet"
sys.path.insert(0, str(MAGNET))
from sds.data import read_split  # noqa: E402
from sds.data.dataset import load_mask  # noqa: E402
from sds.metrics import PixelEvaluator, metrics_at, select_thresholds  # noqa: E402
from sds.utils import (append_result, environment_info, file_sha256, prepare_out_dir, rel, resolve,  # noqa: E402
                       save_json, splits_sha256)
from tools.evaluate import size_edges  # noqa: E402

D = dict(core="data/splits/vision/core.csv", mining="data/splits/vision/mining.csv", val="data/splits/vision/val.csv",
         test="data/splits/vision/test.csv", name="VISION", version="vision_seed42_v1")
BASE, IGNORE = 1024, 255
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def scales_for(c):
    """Square scales (side) from the common input size up to the native canvas side c."""
    s = [BASE]
    while s[-1] * 2 < c:
        s.append(s[-1] * 2)
    if c > s[-1]:
        s.append(c)
    return s


def canvas(row):
    """Normalised image (C, C, 3) and label (C, C) with padding = 0 / IGNORE; also returns (H, W)."""
    img = np.asarray(Image.open(resolve(row["image"])).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    c = max(h, w)
    out = np.zeros((c, c, 3), np.float32)
    out[:h, :w] = (img - MEAN) / STD
    lab = np.full((c, c), IGNORE, np.uint8)
    lab[:h, :w] = load_mask(row["mask"], (h, w)) > 0
    return out, lab, (h, w)


def resize_img(x, s):
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    return F.interpolate(t, size=(s, s), mode="bilinear", align_corners=False)[0]


def resize_lab(y, s):
    return np.asarray(Image.fromarray(y).resize((s, s), Image.NEAREST))


class PatchSet(Dataset):
    """Backbone training: a random BASE x BASE patch of a random scale of each image (their backbone recipe)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y, _ = canvas(self.rows[i])
        s = random.choice(scales_for(x.shape[0]))
        xi, yi = resize_img(x, s), torch.from_numpy(resize_lab(y, s).astype(np.int64))
        y0, x0 = random.randint(0, s - BASE), random.randint(0, s - BASE)
        xi, yi = xi[:, y0:y0 + BASE, x0:x0 + BASE], yi[y0:y0 + BASE, x0:x0 + BASE]
        if random.random() < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        return xi, yi


class PairSet(Dataset):
    """Refinement training (their RandomPair): coarse = whole canvas at BASE; fine = a BASE crop of a random finer
    scale, with its box in coarse-image coordinates."""

    def __init__(self, rows):
        self.rows = [r for r in rows if max(int(r["height"]), int(r["width"])) > BASE]  # needs a finer scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y, _ = canvas(self.rows[i])
        sc = scales_for(x.shape[0])
        coarse = resize_img(x, BASE)
        s = random.choice(sc[1:])
        xf, yf = resize_img(x, s), resize_lab(y, s)
        y0, x0 = random.randint(0, s - BASE), random.randint(0, s - BASE)
        fine, lab = xf[:, y0:y0 + BASE, x0:x0 + BASE], torch.from_numpy(yf[y0:y0 + BASE, x0:x0 + BASE].astype(np.int64))
        r = BASE / s
        box = torch.tensor([[x0 * r, y0 * r, (x0 + BASE) * r, (y0 + BASE) * r]], dtype=torch.float32)
        return coarse, fine, lab, box


def dice2(logits, y):
    """Dice on the defect channel over valid pixels: the unified recipe of our other baselines (BCE+Dice)."""
    p = logits.softmax(1)[:, 1]
    v = (y != IGNORE).float()
    t = (y == 1).float() * v
    p = p * v
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def train_backbone(model, rows, epochs, device, log, unified=False):
    from magnet.utils.loss import OhemCrossEntropy  # noqa: F401  (not used: their backbone config has USE_OHEM false)

    dl = DataLoader(PatchSet(rows), batch_size=4, shuffle=True, num_workers=8, drop_last=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    total, it = epochs * len(dl), 0
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    model.train()
    for ep in range(epochs):
        t0, ls = time.time(), []
        for x, y in dl:
            for g in opt.param_groups:
                g["lr"] = 1e-3 * (1 - it / total) ** 0.9  # poly, as their HRNet-based backbone trainer
            out, yy = model(x.to(device)), y.to(device)
            loss = ce(out, yy) + (dice2(out, yy) if unified else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(float(loss))
            it += 1
        log(f"backbone epoch={ep + 1} loss={np.mean(ls):.4f} time_min={(time.time() - t0) / 60:.2f}")


def train_refinement(model, ref, rows, epochs, device, log, unified=False):
    from torchvision.ops import roi_align

    from magnet.utils.loss import OhemCrossEntropy

    dl = DataLoader(PairSet(rows), batch_size=8, shuffle=True, num_workers=8, drop_last=True)
    opt = torch.optim.SGD([p for p in ref.parameters() if p.requires_grad], lr=1e-3, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 20, 30, 40, 45], gamma=0.1)
    crit = OhemCrossEntropy(ignore_label=IGNORE)
    model.eval()
    for ep in range(epochs):
        ref.train()
        t0, ls = time.time(), []
        for coarse, fine, lab, box in dl:
            coarse, fine, lab = coarse.to(device), fine.to(device), lab.to(device)
            with torch.no_grad():
                coarse_pred = model(coarse).softmax(1)
                fine_pred = model(fine).softmax(1)
            crop_preds = roi_align(coarse_pred, [b.to(device) for b in box], output_size=(BASE, BASE))
            out = ref(crop_preds, fine_pred)
            loss = crit(out, lab) + (dice2(out, lab) if unified else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(float(loss))
        sched.step()
        log(f"refinement epoch={ep + 1} loss={np.mean(ls):.4f} lr={opt.param_groups[0]['lr']:.1e} "
            f"time_min={(time.time() - t0) / 60:.2f}")


@torch.no_grad()
def predict(model, ref, row, device, n_patches=-1, n_points=0.75, smooth=11):
    """Their test.py loop for one image; returns the defect probability at the original resolution and the
    number of backbone patch evaluations."""
    from torchvision.ops import roi_align

    from magnet.utils.blur import MedianBlur
    from magnet.utils.geometry import (calculate_certainty, ensemble, get_patch_coords,
                                       get_uncertain_point_coords_on_grid, point_sample)

    x, _, (h, w) = canvas(row)
    sc = scales_for(x.shape[0])
    blur = MedianBlur(kernel_size=(smooth, smooth)).to(device).eval()
    nc, n_eval = 2, 0

    def batch_pred(net, patches, other=None):
        out = []
        for i in range(patches.shape[0]):
            a = patches[i:i + 1]
            out.append(torch.softmax(net(a) if other is None else net(a, other[i:i + 1]), 1))
        return torch.cat(out)

    final = batch_pred(model, resize_img(x, BASE)[None].to(device))
    n_eval += 1
    for idx, s in enumerate(sc[1:], 1):
        if n_patches == 0:
            break
        ratios = torch.tensor(get_patch_coords((s, s), (BASE, BASE)), device=device, dtype=torch.float32)
        final = F.interpolate(final, (s, s), mode="bilinear", align_corners=False)
        coords = ratios.clone() * s
        unc = 1.0 - calculate_certainty(final)
        pu = roi_align(unc, [coords], output_size=(BASE, BASE)).mean((1, 2, 3))
        _, sel = torch.sort(pu)
        if n_patches != -1:
            sel = sel[:n_patches]
        xs = resize_img(x, s).to(device)
        offs = [(min(int(round(float(r[1]) * s)), s - BASE), min(int(round(float(r[0]) * s)), s - BASE)) for r in ratios[sel]]
        patches = torch.stack([xs[:, a:a + BASE, b:b + BASE] for a, b in offs])
        early = batch_pred(model, patches)
        n_eval += len(sel)
        coarse = roi_align(final, [coords[sel]], output_size=(BASE, BASE))
        fine = batch_pred(ref, early, coarse)
        fine, mask = ensemble(fine, ratios[sel], (s, s))
        cert = calculate_certainty(fine)
        if n_patches > 0:  # verbatim from their test.py (only used by MagNet-Fast)
            cert[:, :, mask] = 0.0
        err = cert * F.interpolate(unc, (s, s), mode="bilinear", align_corners=False)
        err = F.interpolate(blur(F.interpolate(err, size=(BASE, BASE))), size=(s, s))
        npts = int(s * s * n_points * len(sel) / len(ratios))
        ei, ec = get_uncertain_point_coords_on_grid(err, npts)
        ei = ei.unsqueeze(1).expand(-1, nc, -1)
        fp = point_sample(fine, ec, align_corners=False)
        if n_patches > 0:
            sm = point_sample(mask.float()[None, None], ec, align_corners=False).bool().squeeze()
            ei, fp = ei[:, :, sm], fp[:, :, sm]
        final = final.reshape(1, nc, s * s).scatter_(2, ei, fp).view(1, nc, s, s)
    c = x.shape[0]
    final = F.interpolate(final, (c, c), mode="bilinear", align_corners=False)
    return final[0, 1, :h, :w], n_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp", default=None)
    ap.add_argument("--backbone-epochs", type=int, default=100)
    ap.add_argument("--refine-epochs", type=int, default=50)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--fast", action="store_true", help="MagNet-Fast: 3 patches per scale, n_points 0.9")
    ap.add_argument("--coarse-only", action="store_true", help="diagnostic: backbone on the coarsest scale only")
    ap.add_argument("--loss", default="authors", choices=["authors", "unified"],
                    help="unified: + Dice on the defect channel (as BCE+Dice of our other baselines)")
    ap.add_argument("--tol", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="also evaluate the official test split (final only!)")
    ap.add_argument("--max-images", type=int, default=0, help="debug only")
    ap.add_argument("--base", type=int, default=1024, help="debug only (CPU smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    global BASE
    BASE = args.base

    from magnet.model.fpn import ResnetFPN
    from magnet.model.refinement import RefinementMagNet

    exp = resolve(args.exp or f"outputs/vision_magnet{'' if args.loss == 'unified' else '_authors_loss'}"
                  f"{'' if args.seed == 42 else f'_s{args.seed}'}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResnetFPN(2).to(device)
    ref = RefinementMagNet(2, use_bn=True).to(device)
    train_csvs = [D["core"], D["mining"]]
    if not args.eval_only:
        prepare_out_dir(exp, args.overwrite)
        logf = open(exp / "train.log", "a")

        def log(m):
            print(m, flush=True)
            logf.write(m + "\n")
            logf.flush()

        rows = read_split(train_csvs)
        if args.max_images:
            rows = rows[:args.max_images]
        save_json({"run_id": time.strftime("%Y%m%d-%H%M%S"), "seed": args.seed, "train_split": train_csvs,
                   "train_split_sha": splits_sha256(train_csvs), "base": BASE,
                   "backbone_epochs": args.backbone_epochs, "refine_epochs": args.refine_epochs, "loss": args.loss,
                   "magnet_commit": _git(MAGNET), "env": environment_info()}, exp / "config.json")
        t0 = time.time()
        train_backbone(model, rows, args.backbone_epochs, device, log, args.loss == "unified")
        torch.save(model.state_dict(), exp / "backbone.pt")
        for p in model.parameters():
            p.requires_grad_(False)
        train_refinement(model, ref, rows, args.refine_epochs, device, log, args.loss == "unified")
        torch.save(ref.state_dict(), exp / "refinement.pt")
        save_json({"train_minutes_total": (time.time() - t0) / 60, "completed": True}, exp / "train_summary.json")
    else:
        model.load_state_dict(torch.load(exp / "backbone.pt", map_location="cpu"))
        ref.load_state_dict(torch.load(exp / "refinement.pt", map_location="cpu"), strict=False)
    model.eval()
    ref.eval()
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in ref.parameters())

    cfg = json.load(open(exp / "config.json"))
    edges, comp_edges = size_edges(D["core"])
    tag = "main" + ("_fast" if args.fast else "") + ("_coarse" if args.coarse_only else "")
    (exp / "eval").mkdir(exist_ok=True)
    results, thr = {}, None
    for name in ["val"] + (["test"] if args.test else []):
        ev = PixelEvaluator(True, size_edges=edges, comp_edges=comp_edges, category_fn=lambda i: str(i).split("_")[0],
                            tol=args.tol)
        rows = read_split(D[name])[:args.max_images or None]
        pdir = exp / "predictions" / f"{name}{'' if tag == 'main' else '_' + tag}"
        pdir.mkdir(parents=True, exist_ok=True)
        t0, n_eval = time.time(), 0
        for r in rows:
            p, k = predict(model, ref, r, device, n_patches=0 if args.coarse_only else 3 if args.fast else -1,
                           n_points=0.9 if args.fast else 0.75)
            n_eval += k
            gt = torch.from_numpy(load_mask(r["mask"], (int(r["height"]), int(r["width"])))).to(device)
            ev.update(p.float(), gt, r["id"], r["label"])
            if not args.tol:
                np.savez_compressed(pdir / f"{r['id']}.npz", pmain=p.half().cpu().numpy())
        st = ev.state()
        np.savez_compressed(exp / "eval" / f"{name}_{tag}{f'_tol{args.tol}' if args.tol else ''}.npz", **st)
        if name == "val":
            thr = select_thresholds(st, "f1", 0.9)
        m = metrics_at(st, thr[0], thr[1], edges, comp_edges)
        m["mean_crops_per_image"] = n_eval / len(rows)  # backbone evaluations at 1024^2 per image
        m["sec_per_image"] = (time.time() - t0) / len(rows)
        m["params_m"] = n_params / 1e6
        results[name] = m
        print(f"[MagNet{' fast' if args.fast else ''}] {name}: " + " ".join(f"{k}={m[k]:.4f}" for k in (
            "pixel_ap", "defect_small_ap", "foreground_iou", "miou", "precision", "recall", "aupro_small",
            "irstd_pd", "mean_crops_per_image") if k in m))
        if (args.test and name == "val") or args.tol or args.max_images:
            continue
        append_result({**m, "experiment_id": exp.name, "run_id": cfg["run_id"], "project": "SDS", "model": "magnet",
                       "backbone": "resnet50_fpn", "dataset": D["name"], "seed": cfg["seed"],
                       "split_version": D["version"], "train_split_sha": cfg["train_split_sha"],
                       "eval_head": tag, "eval_split": name, "threshold_source": "val_f1",
                       "checkpoint": rel(exp / "refinement.pt"), "checkpoint_sha": file_sha256(exp / "refinement.pt"),
                       "git_commit": environment_info().get("git_commit"),
                       "git_dirty": environment_info().get("git_dirty"),
                       "notes": f"external baseline, authors' recipe (DeepGlobe setting), magnet {cfg['magnet_commit']}"})
    stem = "metrics_test" if args.test else "metrics_val"
    stem = stem if tag == "main" else f"metrics_{tag}_{'test' if args.test else 'val'}"
    save_json({"thresholds": {"main": list(thr)}, "size_edges": edges, "component_size_edges": comp_edges,
               "results": {"main": results}}, exp / (stem + (f"_tol{args.tol}" if args.tol else "") + ".json"))


def _git(root):
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
