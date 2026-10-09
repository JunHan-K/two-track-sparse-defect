"""Two-track inference + evaluation (precision and recall modes of the paper).

  python tools/evaluate_zoom.py --exp outputs/vision_ours --mode precision [--test] [--tol 3]
  python tools/evaluate_zoom.py --exp outputs/vision_ours --mode recall    [--test] [--tol 3]

Low-resolution track L: the whole image at the training input size, resampled to the original size (p_L).
Native track S (same network): 384^2 crops at the ORIGINAL resolution.
  precision: the --n-roi strongest 8-connected peaks of p_L > --tau are re-segmented; native evidence outside the
             low-resolution support M = dilate(p_L > tau, --mask-dilate px) is scaled by --outside; on covered
             pixels p = max(p_L, p_S * (1_M + outside * 1_notM)), elsewhere p = p_L.
  recall:    the whole image is tiled at native resolution, p = max(p_L, p_S).
Thresholds are selected on validation and applied unchanged to test (as in tools/evaluate.py).
Writes metrics_<tag>_{val,test}.json (tag: zoom_main_masked_d16_o0.5_t0.02_n32 = precision, zoom_tile = recall).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sds.data import SegDataset  # noqa: E402
from sds.data.dataset import IMAGENET_MEAN, IMAGENET_STD, load_mask  # noqa: E402
from sds.engine import predict_batch, to_device  # noqa: E402
from sds.metrics import PixelEvaluator, metrics_at, select_thresholds  # noqa: E402
from sds.models import build_model  # noqa: E402
from sds.utils import (append_result, enable_tf32, environment_info, file_sha256, load_config, rel,  # noqa: E402
                       resolve, save_json)
from tools.evaluate import size_edges  # noqa: E402


def pick_rois(cand: np.ndarray, tau: float, n_roi: int) -> list[tuple[int, int]]:
    """Peaks of the n_roi strongest connected regions of cand > tau (8-connectivity), strongest first.

    The per-label maximum and its first position in C order (deterministic tie-break) come from one lexsort.
    """
    lab, n = ndimage.label(cand > tau, structure=np.ones((3, 3)))
    if n == 0:
        return []
    idx = np.flatnonzero(lab)
    lv, cv = lab.ravel()[idx], cand.ravel()[idx]
    o = np.lexsort((idx, -cv, lv))  # by label, then value descending, then position
    first = o[np.r_[True, lv[o][1:] != lv[o][:-1]]]  # one entry per label (labels 1..n in order)
    scores, pos = cv[first], idx[first]
    order = np.argsort(-scores)[:n_roi]
    return [tuple(int(v) for v in np.unravel_index(pos[i], cand.shape)) for i in order]


PRESETS = {"precision": dict(roi_source="main", tau=0.02, n_roi=32, merge="masked", mask_dilate=16, outside=0.5),
           "recall": dict(roi_source="tile", merge="max")}


def tile_centres(h: int, w: int, c: int) -> list[tuple[int, int]]:
    """Cover the whole image with crops (stride = crop): recall mode."""
    ys = list(range(c // 2, max(h - c // 2, c // 2) + 1, c)) or [c // 2]
    xs = list(range(c // 2, max(w - c // 2, c // 2) + 1, c)) or [c // 2]
    if ys[-1] + c // 2 < h:
        ys.append(h - c // 2)
    if xs[-1] + c // 2 < w:
        xs.append(w - c // 2)
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def zoom_image(model: torch.nn.Module, row: dict, base: torch.Tensor, cand: np.ndarray, args, device):
    """Two-track inference for one image -> (final probability map at original resolution, number of crops)."""
    if args.roi_source == "tile":
        centres = tile_centres(row["height"], row["width"], args.crop)
    else:
        centres = pick_rois(cand, args.tau, args.n_roi)
    if not centres:
        return base, 0
    # the image stays uint8 and only the crops are converted to float (bit-identical to converting it all first)
    img = np.asarray(Image.open(resolve(row["image"])).convert("RGB"))
    h, w = img.shape[:2]
    c = args.crop
    zoom = torch.zeros((h, w), device=device, dtype=base.dtype)  # float64 in evaluation, float32 when timed
    covered = torch.zeros((h, w), dtype=torch.bool, device=device)
    for cy, cx in centres:
        y0, x0 = int(np.clip(cy - c // 2, 0, max(0, h - c))), int(np.clip(cx - c // 2, 0, max(0, w - c)))
        ch, cw = min(c, h - y0), min(c, w - x0)
        patch = np.zeros((c, c, 3), np.float32)
        patch[:ch, :cw] = (img[y0:y0 + ch, x0:x0 + cw].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(patch.transpose(2, 0, 1))[None].to(device)
        p = torch.sigmoid(model(x, return_aux=False)["logits"].to(base.dtype))[0, 0, :ch, :cw]
        zoom[y0:y0 + ch, x0:x0 + cw] = torch.maximum(zoom[y0:y0 + ch, x0:x0 + cw], p)
        covered[y0:y0 + ch, x0:x0 + cw] = True
    if args.merge == "masked":
        # native evidence is fully accepted only where the low-resolution track also responded (cand > tau,
        # dilated by --mask-dilate px); elsewhere it is down-weighted as a likely context-free false positive.
        # k iterations of a 3x3 dilation == one (2k+1)^2 max filter (zero outside the image), done on the GPU
        k = args.mask_dilate
        c = torch.as_tensor(cand, device=device)
        m = F.max_pool2d((c > args.tau).float()[None, None], 2 * k + 1, stride=1, padding=k)[0, 0] > 0
        # native evidence without low-res support is scaled by --outside (0 = hard mask, 1 = plain max)
        zoom = torch.where(m, zoom, zoom * args.outside)
    if args.merge in ("max", "masked"):  # keep the low-resolution prediction; the native track can only add evidence
        return torch.where(covered, torch.maximum(zoom, base), base), len(centres)
    return torch.where(covered, zoom, base), len(centres)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--mode", choices=["precision", "recall"], default=None,
                    help="paper operating points (override the options below)")
    ap.add_argument("--roi-source", default="main", choices=["main", "tile"])
    ap.add_argument("--n-roi", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--crop", type=int, default=None, help="default: train.zoom.crop of the run (384)")
    ap.add_argument("--mask-dilate", type=int, default=8, help="masked merge: dilation (px, original res)")
    ap.add_argument("--outside", type=float, default=0.0,
                    help="masked merge: weight of native evidence OUTSIDE the low-res support (0 = hard mask)")
    ap.add_argument("--merge", default="max", choices=["max", "replace", "masked"],
                    help="max: keep the low-resolution prediction and add native evidence; replace: native only")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--tol", type=int, default=0, help="boundary tolerance (original px), see evaluate.py")
    ap.add_argument("--save-probs", action="store_true", help="save final (zoomed) maps under predictions/<split>_<tag>")
    ap.add_argument("--out-suffix", default="", help="append to the result tag (e.g. 'last' for a --ckpt last.pt diagnostic), "
                                                     "so the main result files are never overwritten")
    args = ap.parse_args()
    if args.mode:
        for k, v in PRESETS[args.mode].items():
            setattr(args, k, v)
    enable_tf32()
    exp = resolve(args.exp)
    cfg = load_config(exp / "config.yaml")
    args.crop = args.crop or (cfg["train"].get("zoom") or {}).get("crop", 384)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    ck_path = exp / args.ckpt
    model.load_state_dict(torch.load(ck_path, map_location="cpu")["model"])
    data = cfg["data"]
    edges, comp_edges = size_edges(data["core_split"])
    cat_fn = (lambda i: str(i).split("_")[0]) if data.get("category_from_id") else None
    tag = (f"zoom_{args.roi_source}" + ("" if args.merge == "max" else f"_{args.merge}")
           + (f"_d{args.mask_dilate}" if args.merge == "masked" and args.mask_dilate != 8 else "")
           + (f"_o{args.outside:g}" if args.merge == "masked" and args.outside > 0 else "")
           + ("" if (args.tau, args.n_roi) == (0.1, 8) or args.roi_source == "tile" else f"_t{args.tau}_n{args.n_roi}")
           + (f"_tol{args.tol}" if args.tol else "")
           + (f"_{args.out_suffix}" if args.out_suffix else ""))
    results, thr = {}, None
    for split in ["val"] + (["test"] if args.test else []):
        ds = SegDataset(data[f"{split}_split"], data["input_size"], mask_resize_threshold=data.get("mask_resize_threshold", 0.5))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        ev = PixelEvaluator(True, size_edges=edges, comp_edges=comp_edges, category_fn=cat_fn, tol=args.tol)
        t0, n_crops = time.time(), 0
        for batch in loader:
            batch = to_device(batch, device)
            base = predict_batch(model, batch, ["main"], amp=False)[0]["main"]
            row = ds.rows[int(batch["index"][0])]
            final, k = zoom_image(model, row, base, base.cpu().numpy(), args, device)
            n_crops += k
            if args.save_probs:
                pdir = exp / "predictions" / f"{split}_{tag}"
                pdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pdir / f"{row['id']}.npz", pmain=final.half().cpu().numpy())
            gt = torch.from_numpy(load_mask(row["mask"], (row["height"], row["width"]))).to(device)
            ev.update(final, gt, row["id"], row["label"])
        st = ev.state()
        (exp / "eval").mkdir(exist_ok=True)
        np.savez_compressed(exp / "eval" / f"{split}_{tag}.npz", **st)  # metrics can be recomputed without inference
        if split == "val":
            thr = select_thresholds(st, "f1", 0.9)
        m = metrics_at(st, thr[0], thr[1], edges, comp_edges)
        m["mean_crops_per_image"] = n_crops / len(ds)
        m["sec_per_image"] = (time.time() - t0) / len(ds)
        results[split] = m
        print(f"[{exp.name} {tag}] {split}: " + " ".join(f"{k}={m[k]:.4f}" for k in (
            "pixel_ap", "defect_small_ap", "defect_small_cov", "image_ap", "normal_fp_area", "cat_macro_ap",
            "mean_crops_per_image") if k in m))
        if args.no_csv or (args.test and split == "val"):
            continue
        env = environment_info()
        append_result({**m, "experiment_id": cfg.get("experiment_id_resolved", exp.name), "run_id": cfg.get("run_id"),
                       "project": "SDS", "model": cfg["model"].get("head", cfg["model"].get("type")),
                       "dataset": data["name"], "seed": cfg.get("seed", 42), "eval_head": tag, "eval_split": split,
                       "threshold_source": "val_f1", "checkpoint": rel(ck_path), "checkpoint_sha": file_sha256(ck_path),
                       "git_commit": env.get("git_commit"), "git_dirty": env.get("git_dirty"),
                       "notes": f"n_roi={args.n_roi} tau={args.tau} crop={args.crop}"})
    save_json({"thresholds": list(thr), "args": vars(args), "results": results},
              exp / (f"metrics_{tag}_test.json" if args.test else f"metrics_{tag}_val.json"))


if __name__ == "__main__":
    main()
