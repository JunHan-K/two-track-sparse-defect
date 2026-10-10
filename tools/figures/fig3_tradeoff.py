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
from tools.final_model import VISION as FINAL_VISION  # noqa: E402
from tools.paper_tables import ROOT, agg, gflops  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BASE, OURS = "#7f7f7f", "#c0392b"
# (label, exp, mode, colour, marker)
POINTS = [
    ("SegFormer-B0", "vision_segformer_b0", "full", BASE, "o"),
    ("B0 + tiles", "vision_segformer_b0", "R", BASE, "s"),
    ("B0, $1536^2$", "vision_segformer_b0_1536", "full", BASE, "^"),
    ("BiSeNetV2", "vision_bisenetv2", "full", BASE, "v"),
    ("HRNet", "vision_hrnet_w18s", "full", BASE, "<"),
    ("U-Net", "vision_unet", "full", BASE, "D"),
    ("DeepLabV3+", "vision_deeplabv3p", "full", BASE, ">"),
    ("DNANet", "vision_dnanet", "full", BASE, "p"),
    ("MSHNet", "vision_mshnet", "full", BASE, "h"),
    ("MagNet", "vision_magnet", "full", BASE, "P"),
    ("SegFormer-B5", "vision_segformer_b5", "full", BASE, "8"),
    ("Mask2Former", "vision_mask2former", "full", BASE, "d"),
    ("Ours (precision)", FINAL_VISION, "P", OURS, "*"),
    ("Ours (recall)", FINAL_VISION, "R", OURS, "X"),
]

# label offsets (points) so that neighbouring labels do not overlap
OFFSET = {"SegFormer-B0": (-4, -10, "right"), "HRNet": (-5, 1, "right"), "BiSeNetV2": (4, 4, "left"),
          "Ours (precision)": (-7, 0, "right"), "Ours (recall)": (6, 0, "left"),
          "B0 + tiles": (-14, 27, "right"), "B0, $1536^2$": (5, 4, "left"), "Mask2Former": (5, 4, "left"),
          "MSHNet": (-16, -10, "right"), "DeepLabV3+": (4, -13, "center"), "DNANet": (6, -1, "left"),
          "U-Net": (-6, 3, "right"), "SegFormer-B5": (5, 3, "left"), "MagNet": (0, -9, "center")}
LEADER = {"B0 + tiles", "MSHNet"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="defect_small_ap")
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="outputs/paper/figures")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(4.10, 1.56))  # printed beside the track-disagreement plot (no scaling)
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
        lead = label in LEADER  # crowded spot: label placed away with a straight leader line
        ax.annotate(label, (g, mu), xytext=(dx, dy), textcoords="offset points", fontsize=7.0, color=col, ha=ha,
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.5, shrinkA=0, shrinkB=3) if lead else None)
        print(f"{label:26s} GFLOPs={g:7.1f} y={mu:5.1f}+-{sd:.1f} (n={len(v)})")
    ax.set_xscale("log")
    ax.set_xlim(left=24)  # room for the labels left of the lightest models
    ax.set_ylim(bottom=-10.5, top=37)
    ax.set_xlabel("GFLOPs per image (log)")
    ax.set_ylabel("small-defect AP" + (f"$^{{{a.tol}}}$" if a.tol else "") + " (%)")
    ax.grid(True, which="major", lw=0.3, color="#dddddd")
    ax.set_title("(c) accuracy vs. compute", fontsize=7.4, pad=2)
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
