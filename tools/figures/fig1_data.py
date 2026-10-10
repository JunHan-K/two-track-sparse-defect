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
from tools.final_model import VISION as FINAL_VISION  # noqa: E402
from sds.data import read_split  # noqa: E402
from sds.data.dataset import size_weight_map  # noqa: E402
from sds.utils import resolve  # noqa: E402

TRAIN_ID, VAL_ID = "Console_train_000086", "Console_train_000082"
EXP = "outputs/" + FINAL_VISION  # final model (Sparse Defect Replay)
OUT = Path("outputs/figures/fig1_data")
S8 = np.ones((3, 3), bool)
SPLIT = "test"  # split of the inference example (--split); the paper figure uses the test split


def row(i):
    return {r["id"]: r for r in read_split(["data/splits/vision/core.csv", "data/splits/vision/val.csv", "data/splits/vision/test.csv"])}[i]


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
    d = json.load(open("data/replay/vision/vision_ours_stage1_s1.json"))  # mined for the final model
    fps = sorted([e for e in d["bg"]["native_fp"] if e["id"] == TRAIN_ID], key=lambda e: -e["score"])
    crops = [crop(img, e["cy"], e["cx"], 384)[0] for e in fps[:3]]
    # Sparse Defect Replay: background crops at random locations of the same image (as many as its false positives)
    rnd = [e for e in d["bg"]["random"] if e["id"] == TRAIN_ID]
    # shown: 3 of the image's random background crops, skipping featureless ones (e.g. plain white margin), which
    # would read as empty panels; all of them are used in training
    cands = [crop(img, e["cy"], e["cx"], 384)[0] for e in rnd]
    textured = [c for c in cands if c.astype(np.float32).mean(2).std() > 8.0]
    bg_crops = (textured + [c for c in cands if c.astype(np.float32).mean(2).std() <= 8.0])[:3]
    # small-defect crops of the SAME image (centres of its small GT components, as mined)
    pos = [e for e in d["pos"] if e["id"] == TRAIN_ID]
    pcs = [crop(img, e["cy"], e["cx"], 384)[0] for e in (pos[0], pos[3])]
    sc = 700 / img.shape[1]
    np.savez_compressed(OUT / "replay.npz", whole=down(img, 700), whole_g=down(g, 700, True),
                        fp_xy=np.array([[e["cx"] * sc, e["cy"] * sc, e["lowres_support"]] for e in fps]),
                        pos_xy=np.array([[e["cx"] * sc, e["cy"] * sc] for e in pos]),
                        fp_crops=np.stack(crops), bg_crops=np.stack(bg_crops), pos_crops=np.stack(pcs),
                        n_fp_total=len(d["bg"]["native_fp"]), n_bg_total=len(d["bg"]["random"]),
                        n_pos_total=len(d["pos"]))


def step_pick():
    """Inference example of the final model (test split by default, --split), chosen for visibility: among small defects (< Q33
    area) that the Global Sight Track misses (max p_L < 0.5) and the precision mode finds (max p >= 0.9), in images
    with at most 5 defect components (an uncluttered input), the one with the highest local contrast
    |mean inside - mean of a 2-6 px ring| / std(ring). Rule set on 2026-10-10 after seeing the previous pick
    (largest recovered area: a defect hardly visible in the crop); written to fig1_data/pick.json."""
    q33 = json.load(open(f"{EXP}/metrics_val.json"))["component_size_edges"][0]
    best = None
    for r in read_split(f"data/splits/vision/{SPLIT}.csv"):
        fl = Path(f"{EXP}/predictions/{SPLIT}/{r['id']}.npz")
        fp = Path(f"{EXP}/predictions/{SPLIT}_zoom_main_masked_d16_o0.5_t0.02_n32/{r['id']}.npz")
        if not (fl.exists() and fp.exists()):
            continue
        img, g = load(r)
        lab, n = ndimage.label(g, structure=S8)
        if n == 0 or n > 5:
            continue
        L = np.load(fl)["pmain"].astype(np.float32)
        P = np.load(fp)["pmain"].astype(np.float32)
        gray = img.astype(np.float32).mean(2)
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            m = lab[sl] == k
            if m.sum() / g.size >= q33 or m.sum() < 80 or L[sl][m].max() >= 0.5 or P[sl][m].max() < 0.9:
                continue
            y0, x0 = max(sl[0].start - 12, 0), max(sl[1].start - 12, 0)
            y1, x1 = sl[0].stop + 12, sl[1].stop + 12
            mm = lab[y0:y1, x0:x1] == k
            ring = (ndimage.binary_dilation(mm, iterations=6) & ~ndimage.binary_dilation(mm, iterations=2)
                    & (g[y0:y1, x0:x1] == 0))
            if ring.sum() < 10:
                continue
            gw = gray[y0:y1, x0:x1]
            con = abs(gw[mm].mean() - gw[ring].mean()) / (gw[ring].std() + 1.0)
            if best is None or (con, r["id"]) > (best[0], best[1]):
                best = (float(con), r["id"], k, float(L[sl][m].max()), float(P[sl][m].max()))
    print("pick:", best)
    json.dump({"id": best[1], "component": best[2], "p_L": best[3], "p": best[4]}, open(OUT / "pick.json", "w"))

def native_ps(r, vid, wy, wx, c=384):
    """Raw Native Sight output p_S (before the merge) on the c x c window at (wy, wx) of the full image, exactly as
    tools/evaluate_zoom.py computes it: crops centred on the top-32 peaks of p_L > 0.02, max over overlapping crops.
    Only the crops that overlap the window are run (CPU, a handful of MiT-B0 passes)."""
    import torch
    from sds.data.dataset import IMAGENET_MEAN, IMAGENET_STD
    from sds.models import build_model
    from sds.utils import load_config
    from tools.evaluate_zoom import pick_rois
    L = np.load(f"{EXP}/predictions/{SPLIT}/{vid}.npz")["pmain"].astype(np.float32)
    img = np.asarray(Image.open(resolve(r["image"])).convert("RGB")).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    cfg = load_config(resolve(EXP) / "config.yaml")
    model = build_model(cfg).eval()
    model.load_state_dict(torch.load(resolve(EXP) / "best.pt", map_location="cpu")["model"])
    out = np.zeros((c, c), np.float32)
    for cy, cx in pick_rois(L, 0.02, 32):
        y0, x0 = int(np.clip(cy - c // 2, 0, max(0, h - c))), int(np.clip(cx - c // 2, 0, max(0, w - c)))
        if y0 >= wy + c or y0 + c <= wy or x0 >= wx + c or x0 + c <= wx:
            continue
        ch, cw = min(c, h - y0), min(c, w - x0)
        patch = np.zeros((c, c, 3), np.float32)
        patch[:ch, :cw] = (img[y0:y0 + ch, x0:x0 + cw] - IMAGENET_MEAN) / IMAGENET_STD
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(patch.transpose(2, 0, 1))[None], return_aux=False)["logits"])
        p = p[0, 0, :ch, :cw].numpy()
        ya, yb, xa, xb = max(y0, wy), min(y0 + ch, wy + c), max(x0, wx), min(x0 + cw, wx + c)
        out[ya - wy:yb - wy, xa - wx:xb - wx] = np.maximum(out[ya - wy:yb - wy, xa - wx:xb - wx],
                                                           p[ya - y0:yb - y0, xa - x0:xb - x0])
    return out


def step_infer():
    pk = json.load(open(OUT / "pick.json")) if (OUT / "pick.json").exists() else {"id": VAL_ID, "component": None}
    vid = pk["id"]
    r = row(vid)
    img, g = load(r)
    L = np.load(f"{EXP}/predictions/{SPLIT}/{vid}.npz")["pmain"].astype(np.float32)
    P = np.load(f"{EXP}/predictions/{SPLIT}_zoom_main_masked_d16_o0.5_t0.02_n32/{vid}.npz")["pmain"].astype(np.float32)
    H, W = g.shape
    labf, _ = ndimage.label(g, structure=S8)
    fy, _ = ndimage.center_of_mass(labf == pk["component"]) if pk["component"] else (0.78 * H, 0)
    band = min(H, W)  # the whole image when it is square (the current example), else a square band around the defect
    y_top = int(np.clip(fy - band / 2, 0, H - band))
    img, g, L, P = img[y_top:y_top + band], g[y_top:y_top + band], L[y_top:y_top + band], P[y_top:y_top + band]
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
    if pk["component"]:
        cy, cx = ndimage.center_of_mass((labf == pk["component"])[y_top:y_top + band])
    else:
        lab, n = ndimage.label(g, structure=S8)
        sizes = ndimage.sum(g, lab, range(1, n + 1))
        cy, cx = ndimage.center_of_mass(lab == (int(np.argmin(sizes)) + 1))
    zi, (zy, zx) = crop(img, cy, cx, 384)
    zS = native_ps(r, vid, y_top + zy, zx)
    np.savez_compressed(OUT / "infer.npz", img=down(img, 900), g=down(g, 900, True), L=down(L, 900), P=down(P, 900),
                        M=down(M, 900, True), cand=cand, zoom_img=zi, zoom_g=g[zy:zy + 384, zx:zx + 384],
                        zoom_L=L[zy:zy + 384, zx:zx + 384], zoom_P=P[zy:zy + 384, zx:zx + 384], zoom_S=zS,
                        zoom_box=np.array([zx * sc, zy * sc, 384 * sc]), native_hw=np.array([band, W]),
                        full_hw=np.array([H, W]), y_top=y_top)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["train", "replay", "pick", "infer"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    OUT.mkdir(parents=True, exist_ok=True)
    a = ap.parse_args()
    SPLIT = a.split
    {"train": step_train, "replay": step_replay, "pick": step_pick, "infer": step_infer}[a.step]()
    print("ok")
