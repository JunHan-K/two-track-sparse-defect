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
from tools.final_model import MVTEC as FINAL_MVTEC, VISION as FINAL_VISION  # noqa: E402
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


def latency_models():
    """Models of the newest tools/measure_modes.py run."""
    f = ROOT / LATENCY
    return json.load(open(f))["models"] if f.exists() else {}


def params(exp):
    if exp in PARAMS:
        return PARAMS[exp]
    if exp.startswith("vision_magnet"):  # one MagNet model (authors' or unified loss): counted by tools/measure_modes.py
        p = latency_models().get("vision_magnet", {}).get("params_m")
        return f"{p:.1f}M" if p else "--"
    for s in SEEDS:
        f = ROOT / "outputs" / (exp + s) / "efficiency.json"
        if f.exists():
            return f"{json.load(open(f))['params_m']:.1f}M"
    return "--"


# (label, exp id, mode); columns: (header, metric, tol, scale, digits)
VISION_ROWS = [  # grouped by the purpose of the comparison
    ("SegFormer-B0~\\cite{xie2021segformer} (our backbone)", "vision_segformer_b0", "full"),
    ("BiSeNetV2~\\cite{yu2021bisenetv2}", "vision_bisenetv2", "full"),
    ("HRNet-W18-small~\\cite{wang2021hrnet}", "vision_hrnet_w18s", "full"),
    ("U-Net (R34)~\\cite{ronneberger2015unet}", "vision_unet", "full"),
    ("DeepLabV3+ (R34)~\\cite{chen2018deeplabv3plus}", "vision_deeplabv3p", "full"),
    None,  # capacity / resolution controls
    ("SegFormer-B0, $1536^2$ input", "vision_segformer_b0_1536", "full"),
    ("SegFormer-B0 + native tiles~\\cite{akyon2022sahi}", "vision_segformer_b0", "R"),
    ("SegFormer-B5~\\cite{xie2021segformer}", "vision_segformer_b5", "full"),
    ("Mask2Former (Swin-T)~\\cite{cheng2022mask2former}", "vision_mask2former", "full"),
    None,  # specialised methods
    ("DNANet~\\cite{li2023dnanet}", "vision_dnanet", "full"),
    ("MSHNet~\\cite{liu2024mshnet}", "vision_mshnet", "full"),
    ("MagNet~\\cite{huynh2021magnet}", "vision_magnet", "full"),  # unified loss; authors' loss in the text
    None,
    ("Twin-SparSight, precision mode", FINAL_VISION, "P"),
    ("Twin-SparSight, recall mode", FINAL_VISION, "R"),
]
VISION_COLS = [("AP", "pixel_ap", 0, 100, 1), ("AP$^{3}$", "pixel_ap", 3, 100, 1),
               ("AP$_s$", "defect_small_ap", 0, 100, 1), ("AP$_s^{3}$", "defect_small_ap", 3, 100, 1),
               ("AUPRO$_s$", "aupro_small", 0, 100, 1), ("mIoU", "miou", 0, 100, 1),
               ("$P_d$", "irstd_pd", 0, 100, 1), ("Prec.", "precision", 0, 100, 1)]

MVTEC_ROWS = [  # grouped by the purpose of the comparison
    ("SegFormer-B0~\\cite{xie2021segformer} (our backbone)", "mvtec_segformer_b0", "full"),
    ("BiSeNetV2~\\cite{yu2021bisenetv2}", "mvtec_bisenetv2", "full"),
    ("HRNet-W18-small~\\cite{wang2021hrnet}", "mvtec_hrnet_w18s", "full"),
    ("U-Net (R34)~\\cite{ronneberger2015unet}", "mvtec_unet", "full"),
    ("DeepLabV3+ (R34)~\\cite{chen2018deeplabv3plus}", "mvtec_deeplabv3p", "full"),
    None,  # capacity control
    ("SegFormer-B5~\\cite{xie2021segformer}", "mvtec_segformer_b5", "full"),
    None,  # specialised methods
    ("MSHNet~\\cite{liu2024mshnet}", "mvtec_mshnet", "full"),
    ("SuperSimpleNet~\\cite{rolih2024ssn}", "mvtec_supersimplenet", "full"),
    None,
    ("Twin-SparSight, precision mode", FINAL_MVTEC, "P"),
    ("Twin-SparSight, recall mode", FINAL_MVTEC, "R"),
]
MVTEC_COLS = VISION_COLS  # same layout as Table I (GFLOPs only for VISION)

# Table III, block A: training components on top of the whole-image view (native crops, stage heads, size-aware
# targets), precision mode. The first two rows are Core+Mining baselines as in Table II; the other rows train on Core.
# (Stage heads without native crops, with/without size-aware targets, are discussed in the text only.)
CK = r"\checkmark"
ABL_A = [
    (("", "", ""), "SegFormer-B0$^{*}$", "vision_segformer_b0"),
    (("", "", ""), "B0$^{*}$, update-matched", "vision_abl_b0_update_matched"),
    ((CK, "", ""), "+ native-view training", "vision_abl_b0_two_views"),
    ((CK, CK, ""), "\\quad + stage supervision", "vision_abl_two_views_stage_heads"),
    ((CK, CK, CK), "\\quad\\quad + size-aware", "vision_abl_stage_heads_size_aware"),
    ((CK, CK, CK), "\\quad\\quad\\quad + head fusion", "vision_ours_stage1"),
]
# block B: 20-epoch refinements of "ours" on Core+Mining with the same number of steps
ABL_B = [
    ("before refinement", "vision_abl_stage_heads_size_aware"),
    ("extra training only", "vision_abl_refine_no_replay"),
    ("own false-pos.\\ replay", "vision_abl_refine_own_false_positives"),
    ("Sparse Defect Replay", "vision_abl_replay_unfused"),
    ("\\quad + head fusion (\\textbf{final})", FINAL_VISION),
]
ABL_COLS = [("AP$_s$", "defect_small_ap", 0, 100, 1),
            ("AP$_s^{3}$", "defect_small_ap", 3, 100, 1), ("AUPRO$_s$", "aupro_small", 0, 100, 1),
            ("$P_d$", "irstd_pd", 0, 100, 1), ("Prec.", "precision", 0, 100, 1)]


def ablation_bodies(split):
    """Best/second best marked within each block (training components; refinements)."""
    a, b, plain = [], [], []
    rk_a = ranks([(None, e, "P") for _, _, e in ABL_A], ABL_COLS, split)
    rk_b = ranks([(None, e, "P") for _, e in ABL_B], ABL_COLS, split)

    def cells_of(exp, rk):
        out = []
        for (_, m, t, sc, d), k in zip(ABL_COLS, rk):
            v = agg(exp, "P", m, t, split, sc)
            out.append(mark(cell(v, d), v, k, d))
        return out

    for flags, note, exp in ABL_A:
        cells = cells_of(exp, rk_a)
        a.append(" & ".join([note] + cells) + r"\\")
        plain.append(f"A {''.join('x' if f else '-' for f in flags)} {note[:30]:30s} [{exp}] " + " ".join(cells))
    for name, exp in ABL_B:
        cells = cells_of(exp, rk_b)
        b.append(" & ".join([name] + cells) + r"\\")
        plain.append(f"B {name:30s} [{exp}] " + " ".join(cells))
    return "\n".join(a), "\n".join(b), plain


LOWER_IS_BETTER = {"irstd_fa"}


def ranks(rows, cols, split):
    """Per column: (best, second-best) rounded mean over all rows; lower is better for F_a."""
    out = []
    for _, m, tol, sc, dg in cols:
        vals = sorted({round(float(np.mean(v)), dg) for r in rows if r for v in [agg(r[1], r[2], m, tol, split, sc)] if v},
                      reverse=m not in LOWER_IS_BETTER)
        out.append((vals[0] if vals else None, vals[1] if len(vals) > 1 else None))
    return out


def mark(text, vals, rank, dg):
    """Bold the best and underline the second-best mean of a column (ties share the mark)."""
    if not vals or text == "--":
        return text
    mu = round(float(np.mean(vals)), dg)
    head, sep, tail = text.partition("{\\scriptsize")
    if mu == rank[0]:
        head = f"\\textbf{{{head}}}"
    elif mu == rank[1]:
        head = f"\\underline{{{head}}}"
    return head + sep + tail


def body(rows, cols, split, with_params=False, with_gflops=False):
    lines, plain = [], []
    rk = ranks(rows, cols, split)
    for r in rows:
        if r is None:
            lines.append("\\midrule")
            continue
        label, exp, mode = r
        cells = []
        for (_, m, tol, sc, dg), k in zip(cols, rk):
            v = agg(exp, mode, m, tol, split, sc)
            cells.append(mark(cell(v, dg), v, k, dg))
        lead = [label] + ([params(exp)] if with_params else [])
        if with_gflops:
            g = gflops(exp, mode, split)
            lead.append("--" if g is None else f"{g:.0f}")
        row = " & ".join(lead + cells) + r"\\"
        lines.append(row)
        n = max(len(agg(exp, mode, cols[0][1], 0, split)), 0)
        plain.append(f"{label[:44]:44s} [{exp} {mode} n={n}] " + " ".join(f"{h}={c}" for (h, *_), c in zip(cols, cells)))
    return "\n".join(lines), plain


def native_gflops(e, crops):
    """GFLOPs of a native mode: one low-resolution pass + crops x the exact FLOPs of one crop pass
    (efficiency.json crop_flops_g, tools/crop_flops.py); falls back to pixel-ratio scaling if missing."""
    h, w = e["input_hw"]
    return e["flops_g"] + crops * e.get("crop_flops_g", e["flops_g"] * 384 * 384 / (h * w))


LATENCY = "outputs/latency_vision.json"


def gflops(exp, mode, split="test"):
    """GFLOPs per image: one pass at the training input size, plus mean native crops x exact crop FLOPs for the
    native modes; MagNet's coarse-to-fine loop as measured by tools/measure_modes.py."""
    if exp.startswith("vision_magnet"):
        for f in sorted((ROOT / "outputs").glob("latency_vision*.json"), reverse=True):
            g = json.load(open(f))["models"].get("vision_magnet", {}).get("gflops")
            if g:
                return g
        return None
    for s in SEEDS:
        f = ROOT / "outputs" / (exp + s) / "efficiency.json"
        if f.exists():
            e = json.load(open(f))
            if mode in ("full", None):
                return e["flops_g"]
            crops = mean_crops(exp, mode, split)  # mean over the 3 seeds, like every other number in the tables
            return None if crops is None else native_gflops(e, crops)
    return None


def mean_crops(exp, mode, split="test"):
    v = [m["mean_crops_per_image"] for s in SEEDS
         if (m := run_metrics(exp + s, mode, 0, split)) and "mean_crops_per_image" in m]
    return float(np.mean(v)) if v else None  # tools/measure_modes.py: every method in one run on one A100


def cost_rows(split):
    """Params, GFLOPs, end-to-end latency (one run, one GPU) and mean native crops per VISION image."""
    lat_f = ROOT / LATENCY
    lat = json.load(open(lat_f))["models"] if lat_f.exists() else {}
    out = []
    for label, exp, mode in [("SegFormer-B0", "vision_segformer_b0", None),
                             ("B0, $1536^2$", "vision_segformer_b0_1536", None),
                             ("U-Net (R34)", "vision_unet", None),
                             ("SegFormer-B5", "vision_segformer_b5", None),
                             ("Mask2Former", "vision_mask2former", None),
                             (r"\ours{} (P)", FINAL_VISION, "P"),
                             (r"\ours{} (R)", FINAL_VISION, "R")]:
        f = ROOT / "outputs" / exp / "efficiency.json"
        if not f.exists():
            out.append(f"{label} & -- & -- & -- & --\\\\")
            continue
        e = json.load(open(f))
        gf, crops = e["flops_g"], "--"
        if mode:
            c = mean_crops(exp, mode, split)
            if c is not None:
                gf = native_gflops(e, c)
                crops = f"{c:.1f}"
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
    for name, rows, cols, wp, wg in (("vision", VISION_ROWS, VISION_COLS, True, True),
                                     ("mvtec", MVTEC_ROWS, MVTEC_COLS, True, False)):
        tex, plain = body(rows, cols, a.split, wp, wg)
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
