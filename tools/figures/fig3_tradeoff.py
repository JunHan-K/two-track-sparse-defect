"""Fig. 3: accuracy vs. compute on VISION (validation until the test pass).

  python tools/figures/fig3_tradeoff.py [--metric defect_small_ap --tol 3] [--split val]

x: GFLOPs per image (log scale) = one pass at the training input size, plus, for the native modes, the mean number
of native 384^2 crops x the exact FLOPs of one crop pass (tools/crop_flops.py); MagNet from tools/measure_modes.py. y: small-defect AP (default boundary-tolerant, 3 px),
mean over the seeds that exist, error bar = std; hollow markers = fewer than 3 seeds. Same aggregation as the tables
(tools/paper_tables.py). Runs without results are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paper_tables import ROOT, agg, native_gflops, run_metrics  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BASE, OURS = "#7f7f7f", "#c0392b"
# (label, exp, mode, colour, marker)
POINTS = [
    ("SegFormer-B0", "vision_segformer_b0", "full", BASE, "o"),
    ("B0 + tiles", "vision_segformer_b0", "R", BASE, "s"),
    ("B0, $1536^2$", "vision_segformer_b0_1536", "full", BASE, "^"),
    ("BiSeNetV2", "vision_bisenetv2", "full", BASE, "v"),
    ("HRNet-W18-s", "vision_hrnet_w18s", "full", BASE, "<"),
    ("U-Net", "vision_unet", "full", BASE, "D"),
    ("DeepLabV3+", "vision_deeplabv3p", "full", BASE, ">"),
    ("DNANet", "vision_dnanet", "full", BASE, "p"),
    ("MSHNet", "vision_mshnet", "full", BASE, "h"),
    ("MagNet", "vision_magnet", "full", BASE, "P"),
    ("SegFormer-B5", "vision_segformer_b5", "full", BASE, "8"),
    ("Mask2Former", "vision_mask2former", "full", BASE, "d"),
    ("ours, precision", "vision_ours", "P", OURS, "*"),
    ("ours, recall", "vision_ours", "R", OURS, "X"),
]

# label offsets (points) so that neighbouring labels do not overlap
OFFSET = {"SegFormer-B0": (2, -10, "center"), "HRNet-W18-s": (-5, 0, "right"), "BiSeNetV2": (4, 5, "left"),
          "ours w/o replay, precision": (5, 5, "left"), "ours, precision": (5, 4, "left"),
          "B0 + tiles": (0, -8, "center"), "Mask2Former": (5, 6, "left"), "MSHNet": (-5, 0, "right"),
          "DeepLabV3+": (5, 0, "left"), "DNANet": (5, 0, "left"), "U-Net": (5, 0, "left")}


def gflops(exp, mode, split):
    if exp.startswith("vision_magnet"):  # coarse-to-fine loop: measured by tools/measure_modes.py (patches x FLOPs)
        for f in sorted((ROOT / "outputs").glob("latency_vision*.json"), reverse=True):
            g = json.load(open(f))["models"].get(exp, {}).get("gflops")
            if g:
                return g
        return None
    for s in ("", "_s1", "_s2"):
        f = ROOT / "outputs" / (exp + s) / "efficiency.json"
        if f.exists():
            e = json.load(open(f))
            g = e["flops_g"]
            if mode != "full":
                m = run_metrics(exp + s, mode, 0, split)
                if not m or "mean_crops_per_image" not in m:
                    return None
                g = native_gflops(e, m["mean_crops_per_image"])
            return g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="defect_small_ap")
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="outputs/paper/figures")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    for label, exp, mode, col, mk in POINTS:
        v = agg(exp, mode, a.metric, a.tol, a.split)
        g = gflops(exp, mode, a.split)
        if not v or g is None:
            print("skip", label)
            continue
        mu, sd = np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0)
        full = len(v) >= 3
        ax.errorbar(g, mu, yerr=sd, fmt=mk, ms=7 if mk == "*" else 4.5, color=col, mfc=col if full else "white",
                    mec=col, ecolor=col, elinewidth=0.6, capsize=1.5, zorder=3)
        dx, dy, ha = OFFSET.get(label, (4, 3, "left"))
        ax.annotate(label, (g, mu), xytext=(dx, dy), textcoords="offset points", fontsize=6, color=col, ha=ha,
                    va="center")
        print(f"{label:26s} GFLOPs={g:7.1f} y={mu:5.1f}+-{sd:.1f} (n={len(v)})")
    ax.set_xscale("log")
    ax.set_xlabel("GFLOPs per image (log)")
    ax.set_ylabel("small-defect AP" + (f"$^{{{a.tol}}}$" if a.tol else "") + " (%)")
    ax.grid(True, which="major", lw=0.3, color="#dddddd")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_tradeoff.{ext}", dpi=300)
    print("saved", out / "fig3_tradeoff.pdf")


if __name__ == "__main__":
    main()
