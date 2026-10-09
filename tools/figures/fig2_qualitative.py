"""Fig. 2: qualitative comparison on a sparse-defect benchmark .

  python tools/figures/fig2_qualitative.py --dataset vision --split val
  python tools/figures/fig2_qualitative.py --dataset vision --split test     # paper (after the test pass)

Columns: input (GT outline) | SegFormer-B0 | B0 at 1536 | U-Net | ours (precision mode) | ours + confusion replay.
Reads probability maps saved by evaluate.py / evaluate_zoom.py --save-probs
(outputs/<exp>/predictions/<split>[_<tag>]/<id>.npz). Missing maps are drawn as grey "pending" panels.

Rows are picked automatically and reproducibly (rule fixed before the final results): baseline and
ours-final are binarised at their own validation thresholds; coverage = fraction of SMALL-defect GT
pixels predicted positive (components below the Core Q33 component-area edge, as in the metrics);
only images with at least one small component are eligible;
  gain1, gain2  defect images with the largest coverage(ours final) - coverage(baseline)
  loss          among defect images where ours covers part of the small defects (coverage > 0), the one with
                the smallest coverage(ours final) - coverage(baseline) (a weakness, not a total miss)
(no "false alarm" row: VISION has no labelled normal images). Ties are broken by image id.
Each panel is a square crop (original resolution) centred on the smallest GT component, side =
clip(--zoom x its bounding-box side, --min-window, --window), so tiny defects stay visible.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sds.data import read_split  # noqa: E402
from sds.data.dataset import load_mask  # noqa: E402
from sds.metrics import thr_value  # noqa: E402
from sds.utils import resolve  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# column = (title, experiment, prediction sub-dir suffix, npz key, metrics file holding its validation-selected
# threshold: the --test run stores the thresholds it selected on validation on the current threshold grid)
PRESETS = {
    "vision": {
        "splits": {"val": "data/splits/vision/val.csv", "test": "data/splits/vision/test.csv"},
        "baseline": 0, "ours": 4,
        "columns": [
            ("SegFormer-B0", "vision_segformer_b0", "", "pmain", "metrics_test.json"),
            ("B0, $1536^2$", "vision_segformer_b0_1536", "", "pmain", "metrics_test.json"),
            ("U-Net (R34)", "vision_unet", "", "pmain", "metrics_test.json"),
            ("MagNet", "vision_magnet", "", "pmain", "metrics_test.json"),
            ("Twin-SparSight", "vision_ours", "_zoom_main_masked_d16_o0.5_t0.02_n32", "pmain",
             "metrics_zoom_main_masked_d16_o0.5_t0.02_n32_test.json"),
        ],
    },
}
ROW_NAMES = {"gain1": "ours > baseline", "gain2": "ours > baseline", "loss": "ours < baseline"}


def load_pred(col, split, iid):
    _, exp, suf, key, _ = col
    f = resolve("outputs") / exp / "predictions" / f"{split}{suf}" / f"{iid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[key].astype(np.float32) if key in z.files else None


def val_threshold(col):
    f = resolve("outputs") / col[1] / col[4]
    if not f.exists():
        return 0.5
    t = json.load(open(f))["thresholds"]
    return thr_value(t["main"][0] if isinstance(t, dict) else t[0])


def small_mask(g, q33):
    lab, n = ndimage.label(g, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(g)
    areas = np.asarray(ndimage.sum(g, lab, np.arange(1, n + 1))) / g.size
    keep = np.concatenate([[False], areas < q33])
    return keep[lab]


def pick_rows(rows, split, cols, base_i, ours_i, q33):
    tb, to = val_threshold(cols[base_i]), val_threshold(cols[ours_i])
    gains = []
    for r in rows:
        if r["label"] != 1:
            continue
        pb, po = load_pred(cols[base_i], split, r["id"]), load_pred(cols[ours_i], split, r["id"])
        if pb is None or po is None:
            continue
        g = small_mask(load_mask(r["mask"], (r["height"], r["width"])) > 0, q33)
        if not g.any():
            continue
        gains.append(((po[g] >= to).mean() - (pb[g] >= tb).mean(), r["id"], (po[g] >= to).mean()))
    if len(gains) < 3:
        return {}
    gains.sort(key=lambda t: (-t[0], t[1]))
    # loss row: a weakness, not a total miss -- among images where ours covers part of the small defects
    # (coverage > 0), the one with the largest deficit to the baseline
    base_cov = {t[1]: t[2] - t[0] for t in gains}
    print(f"counts: images={len(gains)} ours_more={sum(t[0] > 0 for t in gains)} ours_less={sum(t[0] < 0 for t in gains)} "
          f"ours_misses_all={sum(1 for t in gains if t[2] == 0 and base_cov[t[1]] > 0)}")
    partial = [t for t in gains if t[2] > 0 and t[0] < 0]
    loss = min(partial or gains, key=lambda t: (t[0], t[1]))[1]
    return {"gain1": gains[0][1], "gain2": gains[1][1], "loss": loss}


def focus_of(g):
    """Centre and bounding-box side of the smallest GT component."""
    lab, n = ndimage.label(g, structure=np.ones((3, 3)))
    areas = ndimage.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
    ys, xs = np.nonzero(lab == int(np.argmin(areas)) + 1)
    return int(ys.mean()), int(xs.mean()), int(max(np.ptp(ys), np.ptp(xs)) + 1)


def draw(dataset, split, window, min_window, zoom, out_dir, base_split_for_rows, fixed_rows=None):
    P = PRESETS[dataset]
    cols = P["columns"]
    q33 = json.load(open(resolve("outputs") / cols[P["ours"]][1] / "metrics_val.json"))["component_size_edges"][0]
    rows = {r["id"]: r for r in read_split(P["splits"][split])}
    # fixed_rows: the selection printed by an earlier run of the same rule (redraw without reloading every map)
    sel = fixed_rows or pick_rows(list(rows.values()), base_split_for_rows or split, cols, P["baseline"], P["ours"], q33)
    if not sel:  # preview only: final (zoom) maps not saved yet -> select rows with the plain model
        sel = pick_rows(list(rows.values()), split, cols, P["baseline"], P["ours"] - 1, q33)
        if sel:
            print("PREVIEW: rows selected with", cols[P["ours"] - 1][0], "(final maps missing)")
            out_dir = Path(out_dir) / "preview"
    if not sel:
        raise SystemExit("not enough saved predictions yet for baseline and ours")
    print("rows:", sel)
    ncol, nrow = 1 + len(cols), len(sel)
    pw = 0.95
    fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol + 0.5, pw * nrow + 0.25), squeeze=False)
    plt.subplots_adjust(left=0.06, right=0.91, top=0.9, bottom=0.02, wspace=0.04, hspace=0.06)
    im_obj = None
    for i, (kind, iid) in enumerate(sel.items()):
        r = rows[iid]
        img = np.asarray(Image.open(resolve(r["image"])).convert("RGB"))
        g = load_mask(r["mask"], (r["height"], r["width"])) > 0
        cy, cx, side = focus_of(g)
        h, w = g.shape
        s = int(min(np.clip(zoom * side, min_window, window), h, w))
        y0, x0 = int(np.clip(cy - s // 2, 0, h - s)), int(np.clip(cx - s // 2, 0, w - s))
        sl = (slice(y0, y0 + s), slice(x0, x0 + s))
        panels = [("input", None)] + [(c[0], load_pred(c, split, iid)) for c in cols]
        for j, (title, p) in enumerate(panels):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if j == 0:
                ax.imshow(img[sl], interpolation="lanczos")
            elif p is None:
                ax.set_facecolor("#e6e6e6")
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, fontsize=6.5, color="#777")
            else:
                im_obj = ax.imshow(p[sl], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            ax.contour(g[sl].astype(float), levels=[0.5], colors="#39d353", linewidths=0.6)
            ax.set_xlim(-0.5, s - 0.5)
            ax.set_ylim(s - 0.5, -0.5)
            if i == 0:
                ax.set_title(title, fontsize=7, pad=2)
        axes[i, 0].set_ylabel(f"{ROW_NAMES[kind]}\n({iid.split('_')[0]})", fontsize=6.5, labelpad=2)
        axes[i, 0].text(0.03, 0.03, f"{s}px", transform=axes[i, 0].transAxes, fontsize=5.5, color="white",
                        va="bottom", ha="left", bbox=dict(fc="black", ec="none", alpha=0.5, pad=0.8))
    if im_obj is not None:
        cax = fig.add_axes([0.925, 0.12, 0.012, 0.7])
        cb = fig.colorbar(im_obj, cax=cax)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
        cb.set_label("probability", fontsize=6.5)
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"fig2_qualitative_{dataset}_{split}"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300)
    print("saved", out / f"{stem}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vision", choices=list(PRESETS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--window", type=int, default=256, help="max crop side in original pixels")
    ap.add_argument("--min-window", type=int, default=48)
    ap.add_argument("--zoom", type=float, default=4.0, help="crop side = zoom x defect bounding-box side")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--rows", default=None, help='redraw only: JSON dict printed as "rows:" by a full run')
    a = ap.parse_args()
    draw(a.dataset, a.split, a.window, a.min_window, a.zoom, a.out, None, json.loads(a.rows) if a.rows else None)
