"""Paired image-level bootstrap of the small-defect AP difference between ours and the competitors (VISION test).

  python tools/analysis/bootstrap_small.py [--n 2000] [--tol 3]

Reads the evaluator states saved by evaluate.py / evaluate_zoom.py (outputs/<exp>/eval/test_<tag>[_tol3].npz).
Small-defect AP is recomputed from the per-component histograms on the threshold grid (8192 logit bins; the tables
use the 262144-bin AP histograms, so the point estimates may differ in the last digit). Each bootstrap sample draws the test images with replacement, the
SAME images for every model and seed; for every model the AP is the mean over its 3 seeds on that sample, so the
interval covers both image sampling and the seed spread. Output: mean difference, 95% percentile interval and the
fraction of samples in which ours is better.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sds.metrics import step_ap  # noqa: E402

OURS = ("Ours (precision)", "vision_ours", "zoom_main_masked_d16_o0.5_t0.02_n32")
OTHERS = [  # every competitor of Table II, plus our own variants (precision mode)
    ("SegFormer-B0", "vision_segformer_b0", "main"),
    ("SegFormer-B0 + tiles", "vision_segformer_b0", "zoom_tile"),
    ("SegFormer-B0, 1536", "vision_segformer_b0_1536", "main"),
    ("SegFormer-B5", "vision_segformer_b5", "main"),
    ("U-Net", "vision_unet", "main"),
    ("DeepLabV3+", "vision_deeplabv3p", "main"),
    ("HRNet-W18-small", "vision_hrnet_w18s", "main"),
    ("BiSeNetV2", "vision_bisenetv2", "main"),
    ("Mask2Former", "vision_mask2former", "main"),
    ("MagNet", "vision_magnet", "main"),
    ("MagNet (authors' loss)", "vision_magnet_authors_loss", "main"),
    ("DNANet", "vision_dnanet", "main"),
    ("MSHNet", "vision_mshnet", "main"),
    ("Ours w/o replay", "vision_ours_stage1", "zoom_main_masked_d16_o0.5_t0.02_n32"),
    ("Ours, refinement w/o replay", "vision_abl_refine_no_replay", "zoom_main_masked_d16_o0.5_t0.02_n32"),
    ("Ours, random-background replay", "vision_abl_refine_random_background", "zoom_main_masked_d16_o0.5_t0.02_n32"),
]


def load(exp, tag, tol):
    out = []
    for s in ("", "_s1", "_s2"):
        f = ROOT / "outputs" / (exp + s) / "eval" / f"test_{tag}{f'_tol{tol}' if tol else ''}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        st = np.load(f, allow_pickle=True)
        d = ROOT / "outputs" / (exp + s)
        m = next(f for f in (d / "metrics_test.json", d / "metrics_val.json") if f.exists())
        q33 = json.load(open(m))["component_size_edges"][0]  # Core component-area edges, the same for every run
        small = st["comp_area_ratio"] < q33
        n_img = len(st["img_ids"])
        pos = np.zeros((n_img, st["comp_hist"].shape[1]), np.int64)  # small-component positives per image
        np.add.at(pos, st["comp_img"][small], st["comp_hist"][small])
        out.append({"ids": list(st["img_ids"]), "pos": pos, "neg": st["img_neg"].astype(np.int64)})
    return out


def ap(runs, idx):
    return float(np.mean([step_ap(r["pos"][idx].sum(0), r["neg"][idx].sum(0)) for r in runs]))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, default=2000)
    a.add_argument("--tol", type=int, default=3)
    a = a.parse_args()
    ours = load(OURS[1], OURS[2], a.tol)
    ids = ours[0]["ids"]
    rng = np.random.default_rng(0)
    samples = [rng.integers(0, len(ids), len(ids)) for _ in range(a.n)]
    full = np.arange(len(ids))
    a_ours = ap(ours, full)
    res = {"tol": a.tol, "n_boot": a.n, "n_images": len(ids), "ours": OURS[0], "ours_ap": a_ours, "vs": {}}
    b_ours = np.array([ap(ours, i) for i in samples])
    for name, exp, tag in OTHERS:
        runs = load(exp, tag, a.tol)
        for r in runs:  # align image order with ours
            if r["ids"] != ids:
                order = {k: j for j, k in enumerate(r["ids"])}
                p = [order[k] for k in ids]
                r["pos"], r["neg"], r["ids"] = r["pos"][p], r["neg"][p], ids
        d = b_ours - np.array([ap(runs, i) for i in samples])
        lo, hi = np.percentile(d, [2.5, 97.5])
        res["vs"][name] = {"ap": ap(runs, full), "diff": a_ours - ap(runs, full), "ci95": [float(lo), float(hi)],
                           "p_better": float((d > 0).mean())}
        print(f"{name:20s} AP={100 * res['vs'][name]['ap']:5.1f}  ours-them={100 * res['vs'][name]['diff']:+5.1f} "
              f"CI95=[{100 * lo:+5.1f}, {100 * hi:+5.1f}]  P(ours better)={res['vs'][name]['p_better']:.3f}", flush=True)
    print(f"ours AP={100 * a_ours:.1f}")
    out = ROOT / "outputs" / f"bootstrap_small_vision_test_tol{a.tol}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
