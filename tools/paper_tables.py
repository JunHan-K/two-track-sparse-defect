"""Collect validation/test metrics of all paper runs into LaTeX table bodies (outputs/paper/tables/*.tex) + a console summary.

  python tools/paper_tables.py [--split val|test]

Every row = one experiment id evaluated in one mode, aggregated over the seeds that exist
(<id>, <id>_s1, <id>_s2): mean +- sample std in %, a dagger marks rows with fewer than 3 seeds, "--" = not run yet.
Metrics are read from metrics_<tag>_<split>.json; mIoU, when missing there, is computed from the saved evaluator
state (outputs/<exp>/eval/<split>_<tag>.npz) at the validation-selected (max-F1) threshold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sds.metrics import curves  # noqa: E402

P_TAG = "zoom_main_masked_d16_o0.5_t0.02_n32"
R_TAG = "zoom_tile"
MODES = {"full": "main", "P": P_TAG, "R": R_TAG}
SEEDS = ("", "_s1", "_s2")
_cache = {}


def run_metrics(exp, mode, tol, split):
    """Metrics dict of one run (None if not evaluated)."""
    key = (exp, mode, tol, split)
    if key in _cache:
        return _cache[key]
    d = ROOT / "outputs" / exp
    tag = MODES[mode] + (f"_tol{tol}" if tol else "")
    out = None
    js = d / (f"metrics_{split}{f'_tol{tol}' if tol else ''}.json" if mode == "full" else f"metrics_{tag}_{split}.json")
    if js.exists():
        r = json.load(open(js))["results"]
        out = dict(r.get(split) or r.get("main", {}).get(split) or {}) or None
    npz = d / "eval" / f"{split}_{tag}.npz"
    if out is not None and "miou" not in out and npz.exists() and split == "val":
        # cheap: only the pixel curves (no AUPRO) -> mIoU at the validation-selected threshold
        with np.load(npz, allow_pickle=True) as z:
            st = {"img_pos": z["img_pos"], "img_neg": z["img_neg"]}
        c = curves(st)
        out["miou"] = float(c["miou"][int(np.argmax(c["f1"]))])
    if out is not None:  # extras that only exist in the json (crops, timing)
        js = d / f"metrics_{tag}_{split}.json"
        if mode != "full" and js.exists():
            r = json.load(open(js))["results"].get(split, {})
            for k in ("mean_crops_per_image", "sec_per_image"):
                if k in r:
                    out[k] = r[k]
    _cache[key] = out
    return out


def agg(exp, mode, metric, tol=0, split="val", scale=100.0):
    vals = []
    for s in SEEDS:
        m = run_metrics(exp + s, mode, tol, split)
        if m is not None and metric in m and m[metric] == m[metric]:
            vals.append(m[metric] * scale)
    return vals


def cell(vals, digits=1):
    if not vals:
        return "--"
    mu = np.mean(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    s = f"{mu:.{digits}f}" + (r"{\scriptsize$\pm$" + f"{sd:.{digits}f}" + "}" if len(vals) > 1 else "")
    return s + ("" if len(vals) >= 3 else r"$^\dagger$")


# runs without an efficiency.json (external code): parameters counted once in the authors' environment
PARAMS = {"mvtec_supersimplenet": "33.7M"}  # SuperSimpleNet: WideResNet50 (layers 1-3) + adaptor + heads


def params(exp):
    if exp in PARAMS:
        return PARAMS[exp]
    for s in SEEDS:
        f = ROOT / "outputs" / (exp + s) / "efficiency.json"
        if f.exists():
            return f"{json.load(open(f))['params_m']:.1f}M"
    return "--"


# (label, exp id, mode); columns: (header, metric, tol, scale, digits)
VISION_ROWS = [
    ("SegFormer-B0~\\cite{xie2021segformer}", "vision_segformer_b0", "full"),
    ("SegFormer-B0, $1536^2$ input", "vision_segformer_b0_1536", "full"),
    ("SegFormer-B5~\\cite{xie2021segformer}", "vision_segformer_b5", "full"),
    ("SegFormer-B0 + native tiles~\\cite{akyon2022sahi}", "vision_segformer_b0", "R"),
    ("BiSeNetV2~\\cite{yu2021bisenetv2}", "vision_bisenetv2", "full"),
    ("HRNet-W18-small~\\cite{wang2021hrnet}", "vision_hrnet_w18s", "full"),
    ("U-Net (R34)~\\cite{ronneberger2015unet}", "vision_unet", "full"),
    ("DeepLabV3+ (R34)~\\cite{chen2018deeplabv3plus}", "vision_deeplabv3p", "full"),
    ("DNANet~\\cite{li2023dnanet}", "vision_dnanet", "full"),
    ("MSHNet~\\cite{liu2024mshnet}", "vision_mshnet", "full"),
    ("Mask2Former (Swin-T)~\\cite{cheng2022mask2former}", "vision_mask2former", "full"),
    ("MagNet~\\cite{huynh2021magnet} (authors' loss)", "vision_magnet_authors_loss", "full"),
    ("MagNet~\\cite{huynh2021magnet} (unified loss)", "vision_magnet", "full"),
    None,
    ("Ours, precision mode", "vision_ours", "P"),
    ("Ours, recall mode", "vision_ours", "R"),
]
VISION_COLS = [("AP", "pixel_ap", 0, 100, 1), ("AP$^{3}$", "pixel_ap", 3, 100, 1),
               ("AP$_s$", "defect_small_ap", 0, 100, 1), ("AP$_s^{3}$", "defect_small_ap", 3, 100, 1),
               ("AUPRO$_s$", "aupro_small", 0, 100, 1), ("mIoU", "miou", 0, 100, 1),
               ("$P_d$", "irstd_pd", 0, 100, 1), ("$F_a$", "irstd_fa", 0, 1e6, 1)]

MVTEC_ROWS = [
    ("SegFormer-B0~\\cite{xie2021segformer}", "mvtec_segformer_b0", "full"),
    ("SegFormer-B5~\\cite{xie2021segformer}", "mvtec_segformer_b5", "full"),
    ("BiSeNetV2~\\cite{yu2021bisenetv2}", "mvtec_bisenetv2", "full"),
    ("HRNet-W18-small~\\cite{wang2021hrnet}", "mvtec_hrnet_w18s", "full"),
    ("U-Net (R34)~\\cite{ronneberger2015unet}", "mvtec_unet", "full"),
    ("DeepLabV3+ (R34)~\\cite{chen2018deeplabv3plus}", "mvtec_deeplabv3p", "full"),
    ("MSHNet~\\cite{liu2024mshnet}", "mvtec_mshnet", "full"),
    ("SuperSimpleNet~\\cite{rolih2024ssn}", "mvtec_supersimplenet", "full"),
    None,
    ("Ours, precision mode", "mvtec_ours", "P"),
    ("Ours, recall mode", "mvtec_ours", "R"),
]
MVTEC_COLS = [("AP", "pixel_ap", 0, 100, 1), ("AP$^{3}$", "pixel_ap", 3, 100, 1),
              ("AP$_s$", "defect_small_ap", 0, 100, 1), ("AUPRO", "aupro", 0, 100, 1),
              ("AUPRO$_s$", "aupro_small", 0, 100, 1), ("mIoU", "miou", 0, 100, 1)]

# Table III, block A: training components (whole view, native crops, stage heads, size-aware), precision mode.
# The first two rows are the Core+Mining baselines of Table II; all other rows train on Core only.
CK = r"\checkmark"
ABL_A = [
    ((CK, "", "", ""), "SegFormer-B0$^{*}$", "vision_segformer_b0"),
    ((CK, "", "", ""), "SegFormer-B0$^{*}$, update-matched", "vision_abl_b0_update_matched"),
    ((CK, CK, "", ""), "", "vision_abl_b0_two_views"),
    ((CK, "", CK, ""), "", "vision_abl_stage_heads"),
    ((CK, "", CK, CK), "", "vision_abl_stage_heads_size_aware"),
    ((CK, CK, CK, ""), "", "vision_abl_two_views_stage_heads"),
    ((CK, CK, CK, CK), "\\textbf{ours}", "vision_ours_stage1"),
    ((CK, CK, CK, CK), "+ inference-time fusion of $s_1,s_2$", "vision_abl_inference_fusion"),
]
# block B: refinement of the final A+B model (all three: 20 epochs on Core+Mining)
ABL_B = [
    ("none", "vision_ours_stage1"),
    ("extra training, no replay", "vision_abl_refine_no_replay"),
    ("random-background replay", "vision_abl_refine_random_background"),
    ("confusion replay (ours)", "vision_ours"),
]
ABL_COLS = [("AP", "pixel_ap", 0, 100, 1), ("AP$_s$", "defect_small_ap", 0, 100, 1),
            ("AP$_s^{3}$", "defect_small_ap", 3, 100, 1), ("AUPRO$_s$", "aupro_small", 0, 100, 1),
            ("Prec.", "precision", 0, 100, 1)]


def ablation_bodies(split):
    a, b, plain = [], [], []
    for flags, note, exp in ABL_A:
        cells = [cell(agg(exp, "P", m, t, split, sc), d) for _, m, t, sc, d in ABL_COLS]
        a.append(" & ".join(list(flags) + [note] + cells) + r"\\")
        plain.append(f"A {''.join('x' if f else '-' for f in flags)} {note[:30]:30s} [{exp}] " + " ".join(cells))
    for name, exp in ABL_B:
        cells = [cell(agg(exp, "P", m, t, split, sc), d) for _, m, t, sc, d in ABL_COLS]
        b.append(" & ".join([r"\multicolumn{5}{l}{\quad " + name + "}"] + cells) + r"\\")
        plain.append(f"B {name:30s} [{exp}] " + " ".join(cells))
    return "\n".join(a), "\n".join(b), plain


def body(rows, cols, split, with_params=False):
    lines, plain = [], []
    for r in rows:
        if r is None:
            lines.append("\\midrule")
            continue
        label, exp, mode = r
        cells = [cell(agg(exp, mode, m, tol, split, sc), dg) for _, m, tol, sc, dg in cols]
        lead = [label] + ([params(exp)] if with_params else [])
        lines.append(" & ".join(lead + cells) + r"\\")
        n = max(len(agg(exp, mode, cols[0][1], 0, split)), 0)
        plain.append(f"{label[:44]:44s} [{exp} {mode} n={n}] " + " ".join(f"{h}={c}" for (h, *_), c in zip(cols, cells)))
    return "\n".join(lines), plain


def native_gflops(e, crops):
    """GFLOPs of a native mode: one low-resolution pass + crops x the exact FLOPs of one crop pass
    (efficiency.json crop_flops_g, tools/crop_flops.py); falls back to pixel-ratio scaling if missing."""
    h, w = e["input_hw"]
    return e["flops_g"] + crops * e.get("crop_flops_g", e["flops_g"] * 384 * 384 / (h * w))


LATENCY = "outputs/latency_vision.json"  # tools/measure_modes.py: every method in one run on one A100


def cost_rows(split):
    """Params, GFLOPs, end-to-end latency (one run, one GPU) and mean native crops per VISION image."""
    lat_f = ROOT / LATENCY
    lat = json.load(open(lat_f))["models"] if lat_f.exists() else {}
    out = []
    for label, exp, mode in [("SegFormer-B0", "vision_segformer_b0", None),
                             ("SegFormer-B0, $1536^2$", "vision_segformer_b0_1536", None),
                             ("U-Net (R34)", "vision_unet", None),
                             ("SegFormer-B5", "vision_segformer_b5", None),
                             ("Mask2Former (Swin-T)", "vision_mask2former", None),
                             ("Ours, precision mode", "vision_ours", "P"),
                             ("Ours, recall mode", "vision_ours", "R")]:
        f = ROOT / "outputs" / exp / "efficiency.json"
        if not f.exists():
            out.append(f"{label} & -- & -- & -- & --\\\\")
            continue
        e = json.load(open(f))
        gf, crops = e["flops_g"], "--"
        if mode:
            m = run_metrics(exp, mode, 0, split)
            if m and "mean_crops_per_image" in m:
                gf = native_gflops(e, m["mean_crops_per_image"])
                crops = f"{m['mean_crops_per_image']:.1f}"
        key = exp if mode is None else f"{exp}:{'precision' if mode == 'P' else 'recall'}"
        ms = lat.get(key, {}).get("latency_ms")
        out.append(f"{label} & {e['params_m']:.1f}M & {gf:.0f} & {'--' if ms is None else f'{ms:.0f}'} & {crops}\\\\")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="outputs/paper/tables")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, rows, cols, wp in (("vision", VISION_ROWS, VISION_COLS, False), ("mvtec", MVTEC_ROWS, MVTEC_COLS, True)):
        tex, plain = body(rows, cols, a.split, wp)
        (out / f"{name}_body.tex").write_text(tex + "\n")
        print(f"== {name} ({a.split})")
        print("\n".join(plain))
    ta, tb, plain = ablation_bodies(a.split)
    (out / "ablation_a_body.tex").write_text(ta + "\n")
    (out / "ablation_b_body.tex").write_text(tb + "\n")
    print("== ablation (" + a.split + ")\n" + "\n".join(plain))
    (out / "cost_body.tex").write_text(cost_rows(a.split) + "\n")
    print("== cost\n" + cost_rows(a.split))


if __name__ == "__main__":
    main()
