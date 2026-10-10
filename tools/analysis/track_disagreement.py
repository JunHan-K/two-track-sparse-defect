"""Fig. 4: what the two tracks find and where they disagree (component level, original resolution).

  python tools/analysis/track_disagreement.py --exp outputs/vision_abl_stage_heads_size_aware [--split val]

L = low-resolution track (whole image at the training input size, resampled to the original size).
S = native track on all native-resolution tiles (recall mode without the max-merge), same network.
Both binarised at 0.5 (as P_d/F_a). Counted per validation image:
  ground-truth components (8-connected), by size group (Core component-area edges, as in the metrics):
      found by both / S only / L only / missed            (hit = any predicted pixel on the component)
  predicted components of (L or S): true if they touch the ground truth dilated by 3 px, else false positive;
      source = L only / both / S only; S-only ones are further split by whether they lie inside the
      low-resolution support M (p_L > tau dilated by 16 px, the precision-mode mask).
Writes <exp>/analysis/track_disagreement_<split>.json and outputs/paper/figures/fig4_disagreement.{pdf,png}.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sds.data import SegDataset  # noqa: E402
from sds.data.dataset import load_mask  # noqa: E402
from sds.engine import predict_batch, to_device  # noqa: E402
from sds.metrics import GROUPS, _group  # noqa: E402
from sds.models import build_model  # noqa: E402
from sds.utils import enable_tf32, load_config, resolve, save_json  # noqa: E402
from tools.evaluate import size_edges  # noqa: E402
from tools.evaluate_zoom import zoom_image  # noqa: E402

S8 = np.ones((3, 3), bool)


def analyse(exp, split, tau, dilate):
    cfg = load_config(exp / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(exp / "best.pt", map_location="cpu")["model"])
    data = cfg["data"]
    _, comp_edges = size_edges(data["core_split"])
    ds = SegDataset(data[f"{split}_split"], data["input_size"], mask_resize_threshold=data.get("mask_resize_threshold", 0.5))
    zargs = SimpleNamespace(roi_source="tile", crop=(cfg["train"].get("zoom") or {}).get("crop", 384), merge="replace",
                            tau=tau, n_roi=0, mask_dilate=dilate, outside=0.0)
    gt_counts = {g: {"both": 0, "S_only": 0, "L_only": 0, "missed": 0} for g in GROUPS}
    pred = {src: {"true": 0, "false": 0} for src in ("L_only", "both", "S_only")}
    s_only_fp = {"inside_M": 0, "outside_M": 0}
    s_only_tp = {"inside_M": 0, "outside_M": 0}
    n_img = 0
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=1, shuffle=False, num_workers=4):
            batch = to_device(batch, device)
            row = ds.rows[int(batch["index"][0])]
            pl = predict_batch(model, batch, ["main"], amp=False)[0]["main"]
            ps, _ = zoom_image(model, row, pl, pl.cpu().numpy(), zargs, device)
            pl, ps = pl.cpu().numpy(), ps.cpu().numpy()
            g = load_mask(row["mask"], (row["height"], row["width"])) > 0
            L, S = pl >= 0.5, ps >= 0.5
            M = ndimage.binary_dilation(pl > tau, structure=S8, iterations=dilate)
            gd = ndimage.binary_dilation(g, structure=S8, iterations=3)
            lab, n = ndimage.label(g, structure=S8)
            for i, sl in enumerate(ndimage.find_objects(lab), 1):
                c = lab[sl] == i
                grp = _group(c.sum() / g.size, comp_edges)
                hl, hs = bool(L[sl][c].any()), bool(S[sl][c].any())
                gt_counts[grp]["both" if hl and hs else "S_only" if hs else "L_only" if hl else "missed"] += 1
            lab, n = ndimage.label(L | S, structure=S8)
            for i, sl in enumerate(ndimage.find_objects(lab), 1):
                c = lab[sl] == i
                inl, ins = bool(L[sl][c].any()), bool(S[sl][c].any())
                src = "both" if inl and ins else "L_only" if inl else "S_only"
                true = bool(gd[sl][c].any())
                pred[src]["true" if true else "false"] += 1
                if src == "S_only":
                    where = "inside_M" if M[sl][c].any() else "outside_M"
                    (s_only_tp if true else s_only_fp)[where] += 1
            n_img += 1
    return {"exp": str(exp), "split": split, "images": n_img, "threshold": 0.5, "tau": tau, "dilate": dilate,
            "gt_components": gt_counts, "pred_components": pred, "s_only_true": s_only_tp, "s_only_false": s_only_fp}


def plot(r, out):
    # drawn at its printed size (left part of a text-width figure, beside the trade-off plot), type not scaled
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(2.95, 0.92), gridspec_kw={"width_ratios": [1, 1.0], "wspace": 0.62})
    cols = {"both": "#7f7f7f", "S_only": "#7a5bb5", "L_only": "#3f6fb0", "missed": "#e6e6e6"}
    names = {"both": "both", "S_only": "Native only", "L_only": "Global only", "missed": "missed"}
    groups = [g for g in GROUPS if sum(r["gt_components"][g].values())]
    for j, g in enumerate(groups):
        c = r["gt_components"][g]
        tot, left = sum(c.values()), 0.0
        for k in ("both", "S_only", "L_only", "missed"):
            f = c[k] / tot
            a.barh(j, f, left=left, color=cols[k], edgecolor="white", lw=0.4, label=names[k] if j == 0 else None)
            left += f
    a.set_yticks(range(len(groups)))
    a.set_yticklabels([f"{g}\n(n={sum(r['gt_components'][g].values())})" for g in groups], fontsize=7.0,
                      linespacing=1.0)
    a.set_xlim(0, 1)
    a.set_xlabel("fraction of defects", labelpad=1)
    a.set_title("(a) defects found", fontsize=7.4, pad=2)
    a.legend(fontsize=7.0, frameon=False, loc="upper center", bbox_to_anchor=(0.40, -0.40), ncol=2, handlelength=0.8,
             columnspacing=0.8, handletextpad=0.4)
    n = r["images"]
    src = ["L_only", "both", "S_only"]
    tp = [r["pred_components"][s]["true"] / n for s in src]
    fp = [r["pred_components"][s]["false"] / n for s in src]
    y = np.arange(len(src))
    b.barh(y, tp, color="#39a95a", label="on a defect")
    b.barh(y, fp, left=tp, color="#c0392b", label="false positive")
    fo = r["s_only_false"]["outside_M"] / n
    b.barh(2, fo, left=tp[2] + fp[2] - fo, color="none", edgecolor="black", hatch="////", lw=0.4,
           label="outside support $M$")
    b.set_yticks(y)
    b.set_yticklabels([names[s].replace(" ", "\n", 1) for s in src], fontsize=7.0, linespacing=1.0)
    b.set_xlabel("regions per image", labelpad=1)
    b.set_title("(b) predicted regions", fontsize=7.4, pad=2)
    b.legend(fontsize=7.0, frameon=False, loc="upper center", bbox_to_anchor=(0.40, -0.40), ncol=1, handlelength=0.8,
             handletextpad=0.4, labelspacing=0.2)
    for ax in (a, b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig4_disagreement.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved", out / "fig4_disagreement.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--dilate", type=int, default=16)
    ap.add_argument("--out", default="outputs/paper/figures")
    ap.add_argument("--plot-only", action="store_true", help="redraw from the saved json")
    a = ap.parse_args()
    enable_tf32()
    exp = resolve(a.exp)
    js = exp / "analysis" / f"track_disagreement_{a.split}.json"
    if a.plot_only:
        r = json.load(open(js))
    else:
        r = analyse(exp, a.split, a.tau, a.dilate)
        js.parent.mkdir(parents=True, exist_ok=True)
        save_json(r, js)
        print(json.dumps(r, indent=1))
    plot(r, resolve(a.out))


if __name__ == "__main__":
    main()
