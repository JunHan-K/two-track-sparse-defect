"""Pre-compute the real image crops and model maps drawn in Fig. 1 (tools/figures/fig1_framework.py).

  python tools/figures/fig1_data.py --step train|replay|infer      -> outputs/figures/fig1_data/<step>.npz

Training/replay example: VISION Console_train_000086 (Core; text, logos and vents are defect-like patterns);
its native-view false positives are the ones mined from the seed-1 model (data/replay/vision/native_nf_s1.json).
Inference example: VISION Console_train_000082 (validation split): a small scratch that the low-resolution
track scores 0.49 and the precision mode recovers (1.0); maps of the final model (seed 42) saved by evaluate.py /
evaluate_zoom.py --save-probs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sds.data import read_split  # noqa: E402
from sds.data.dataset import size_weight_map  # noqa: E402
from sds.utils import resolve  # noqa: E402

TRAIN_ID, VAL_ID = "Console_train_000086", "Console_train_000082"
EXP = "outputs/vision_ours_stage1"
OUT = Path("outputs/figures/fig1_data")
S8 = np.ones((3, 3), bool)


def row(i):
    return {r["id"]: r for r in read_split(["data/splits/vision/core.csv", "data/splits/vision/val.csv"])}[i]


def load(r):
    img = np.asarray(Image.open(resolve(r["image"])).convert("RGB"))
    m = np.array(Image.open(resolve(r["mask"])))
    return img, (m.max(2) if m.ndim == 3 else m) > 0


def down(a, w, nearest=False):
    h = int(round(a.shape[0] * w / a.shape[1]))
    im = Image.fromarray(a if a.dtype == np.uint8 else (a * 255).astype(np.uint8) if a.dtype == bool else a)
    return np.asarray(im.resize((w, h), Image.NEAREST if nearest else Image.BILINEAR))


def crop(a, cy, cx, s=384):
    h, w = a.shape[:2]
    y0, x0 = int(np.clip(cy - s // 2, 0, h - s)), int(np.clip(cx - s // 2, 0, w - s))
    return a[y0:y0 + s, x0:x0 + s], (y0, x0)


def step_train():
    r = row(TRAIN_ID)
    img, g = load(r)
    q33 = json.load(open(f"{EXP}/metrics_val.json"))["component_size_edges"][0]
    lab, n = ndimage.label(g, structure=S8)
    area = ndimage.sum(g, lab, range(1, n + 1))
    small_ids = [i + 1 for i, a in enumerate(area) if a / g.size < q33]
    small = np.isin(lab, small_ids)
    # crop centred on the cluster of defects (two scratches, one small)
    cy, cx = ndimage.center_of_mass(g)
    c_img, (y0, x0) = crop(img, cy, cx, 768)
    c_g, c_s = g[y0:y0 + 768, x0:x0 + 768], small[y0:y0 + 768, x0:x0 + 768]
    # size-aware weight at the training resolution (letterbox 1024), shown on the native crop
    s = 1024 / max(g.shape)
    a_ref = q33 * 1024 * 1024
    w = size_weight_map(c_g.astype(np.float32), a_ref=a_ref / (s * s), beta=1.0, cap=4.0)
    sy, sx = ndimage.center_of_mass(small) if small.any() else (cy, cx)
    n_img, (ny0, nx0) = crop(img, sy, sx, 384)
    n_g, n_s = g[ny0:ny0 + 384, nx0:nx0 + 384], small[ny0:ny0 + 384, nx0:nx0 + 384]
    n_w = size_weight_map(n_g.astype(np.float32), a_ref=a_ref / (s * s), beta=1.0, cap=4.0)
    np.savez_compressed(OUT / "train.npz", whole=down(img, 512), whole_g=down(g, 512, True), crop=c_img, crop_g=c_g,
                        crop_small=c_s, weight=w, native=n_img, native_g=n_g, native_small=n_s, native_w=n_w,
                        native_box=np.array([ny0, nx0, g.shape[0], g.shape[1]]),
                        box=np.array([y0, x0, 768, g.shape[0], g.shape[1]]))


def step_replay():
    r = row(TRAIN_ID)
    img, g = load(r)
    d = json.load(open("data/replay/vision/native_nf_s1.json"))
    fps = sorted([e for e in d["bg"]["native_fp"] if e["id"] == TRAIN_ID], key=lambda e: -e["score"])
    crops = [crop(img, e["cy"], e["cx"], 384)[0] for e in fps[:3]]
    pos = [e for e in d["pos"] if e["id"].startswith("Console")][:2]
    rows = {x["id"]: x for x in read_split("data/splits/vision/core.csv")}
    pcs = []
    for e in pos:
        pi, _ = load(rows[e["id"]])
        pcs.append(crop(pi, e["cy"], e["cx"], 384)[0])
    sc = 700 / img.shape[1]
    np.savez_compressed(OUT / "replay.npz", whole=down(img, 700), whole_g=down(g, 700, True),
                        fp_xy=np.array([[e["cx"] * sc, e["cy"] * sc, e["lowres_support"]] for e in fps]),
                        fp_crops=np.stack(crops), pos_crops=np.stack(pcs), n_fp_total=len(d["bg"]["native_fp"]),
                        n_pos_total=len(d["pos"]))


def step_infer():
    r = row(VAL_ID)
    img, g = load(r)
    L = np.load(f"{EXP}/predictions/val/{VAL_ID}.npz")["pmain"].astype(np.float32)
    P = np.load(f"{EXP}/predictions/val_zoom_main_masked_d16_o0.5_t0.02_n32/{VAL_ID}.npz")["pmain"].astype(np.float32)
    H, W = g.shape
    y_top = int(0.56 * H)  # lower part: vents and the defect (keeps the printed brand line out of the figure)
    img, g, L, P = img[y_top:], g[y_top:], L[y_top:], P[y_top:]
    sc = 900 / W
    # candidates as in evaluate_zoom (top-32 peaks of p_L above tau), on a max-pooled map to stay light
    k = 4
    Lp = L[: L.shape[0] // k * k, : W // k * k].reshape(L.shape[0] // k, k, W // k, k).max((1, 3))
    lab, n = ndimage.label(Lp > 0.02, structure=S8)
    peaks = ndimage.maximum_position(Lp, lab, range(1, n + 1)) if n else []
    scores = ndimage.maximum(Lp, lab, range(1, n + 1)) if n else []
    order = np.argsort(-np.asarray(scores))[:32]
    cand = np.array([[peaks[i][1] * k * sc, peaks[i][0] * k * sc, scores[i]] for i in order])
    M = ndimage.binary_dilation(Lp > 0.02, structure=S8, iterations=max(1, 16 // k))
    lab, n = ndimage.label(g, structure=S8)
    sizes = ndimage.sum(g, lab, range(1, n + 1))
    cy, cx = ndimage.center_of_mass(lab == (int(np.argmin(sizes)) + 1))
    zi, (zy, zx) = crop(img, cy, cx, 384)
    np.savez_compressed(OUT / "infer.npz", img=down(img, 900), g=down(g, 900, True), L=down(L, 900), P=down(P, 900),
                        M=down(M, 900, True), cand=cand, zoom_img=zi, zoom_g=g[zy:zy + 384, zx:zx + 384],
                        zoom_L=L[zy:zy + 384, zx:zx + 384], zoom_P=P[zy:zy + 384, zx:zx + 384],
                        zoom_box=np.array([zx * sc, zy * sc, 384 * sc]), native_hw=np.array([H - y_top, W]),
                        full_hw=np.array([H, W]), y_top=y_top)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["train", "replay", "infer"])
    OUT.mkdir(parents=True, exist_ok=True)
    {"train": step_train, "replay": step_replay, "infer": step_infer}[ap.parse_args().step]()
    print("ok")
