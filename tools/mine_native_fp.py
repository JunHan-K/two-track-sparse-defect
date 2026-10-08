"""Mine native-resolution false positives of the two-track model for confusion replay.

  python tools/mine_native_fp.py --exp outputs/vision_ours_stage1 --out data/replay/vision/s42.json

For every image of the refinement training pool (Core + Mining; labels of val/test are never used):
  base   = low-resolution whole-image pass (original resolution)
  native = whole-image native-resolution tiles (recall mode, i.e. every native response is visible)
  native false positives = 8-connected components of native >= --thr that do not touch the GT mask
                           (GT dilated by --gt-dilation px). Each records whether the low-res track also
                           responded (base >= 0.02 anywhere in it): "native_only" = the confusion region.
  positives = centres of SMALL GT components (< Core Q33 component area), the defects the replay must keep.
  random    = random background points (same count as native_fp), control for "any extra background crops".
Replay types written: native_fp, native_only, random. The file carries the provenance that train.py checks.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sds.data import SegDataset  # noqa: E402
from sds.data.dataset import load_mask  # noqa: E402
from sds.engine import predict_batch, to_device  # noqa: E402
from sds.mining import replay_provenance  # noqa: E402
from sds.models import build_model  # noqa: E402
from sds.utils import enable_tf32, load_config, resolve, save_json  # noqa: E402
from tools.evaluate import size_edges  # noqa: E402
from tools.evaluate_zoom import zoom_image  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--gt-dilation", type=int, default=5)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--max-per-image", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    enable_tf32()
    exp = resolve(a.exp)
    cfg = load_config(exp / "config.yaml")
    crop = (cfg["train"].get("zoom") or {}).get("crop", 384)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(dev).eval()
    ck = exp / "best.pt"
    model.load_state_dict(torch.load(ck, map_location="cpu")["model"])
    d = cfg["data"]
    q33 = size_edges(d["core_split"])[1][0]
    rng = np.random.default_rng(a.seed)
    zargs = SimpleNamespace(roi_source="tile", tau=0.02, n_roi=32, crop=crop, merge="replace", mask_dilate=8, outside=1.0)
    pos, fp, only, rand = [], [], [], []
    stats = {}
    for split in ("core_split", "mining_split"):
        ds = SegDataset(d[split], d["input_size"], mask_resize_threshold=d.get("mask_resize_threshold", 0.5))
        n_img = n_fp = 0
        for b in DataLoader(ds, batch_size=1, shuffle=False, num_workers=4):
            b = to_device(b, dev)
            row = ds.rows[int(b["index"][0])]
            base = predict_batch(model, b, ["main"], amp=False)[0]["main"]
            native, _ = zoom_image(model, row, base, base.cpu().numpy(), zargs, dev)
            base, native = base.cpu().numpy(), native.cpu().numpy()
            g = load_mask(row["mask"], (row["height"], row["width"])) > 0 if row["label"] == 1 else np.zeros_like(base, bool)
            gd = ndimage.binary_dilation(g, iterations=a.gt_dilation) if g.any() else g
            lab, n = ndimage.label(native >= a.thr, structure=np.ones((3, 3)))
            found = []
            for j, sl in enumerate(ndimage.find_objects(lab), start=1):
                m = lab[sl] == j
                if m.sum() < a.min_area or gd[sl][m].any():
                    continue
                ys, xs = np.nonzero(m)
                e = {"id": row["id"], "cy": float(ys.mean() + sl[0].start), "cx": float(xs.mean() + sl[1].start),
                     "area": int(m.sum()), "score": float(native[sl][m].max()),
                     "lowres_support": bool((base[sl][m] >= 0.02).any())}
                found.append(e)
            found.sort(key=lambda e: -e["score"])
            found = found[:a.max_per_image]
            fp += found
            only += [e for e in found if not e["lowres_support"]]
            n_fp += len(found)
            if g.any():
                gl, gn = ndimage.label(g, structure=np.ones((3, 3)))
                for j, sl in enumerate(ndimage.find_objects(gl), start=1):
                    m = gl[sl] == j
                    if m.sum() / g.size < q33:
                        ys, xs = np.nonzero(m)
                        pos.append({"id": row["id"], "cy": float(ys.mean() + sl[0].start), "cx": float(xs.mean() + sl[1].start)})
            for _ in range(len(found)):
                for _t in range(50):
                    y, x = int(rng.integers(0, g.shape[0])), int(rng.integers(0, g.shape[1]))
                    if not gd[y, x]:
                        rand.append({"id": row["id"], "cy": float(y), "cx": float(x)})
                        break
            n_img += 1
        stats[split] = {"images": n_img, "native_fp": n_fp}
    rep = {"native": True, "crop": crop, "splits": [d["core_split"], d["mining_split"]],
           "pos": pos, "bg": {"native_fp": fp, "native_only": only, "random": rand},
           "stats": {**stats, "pos_small": len(pos), "native_fp": len(fp), "native_only": len(only), "random": len(rand)},
           "provenance": replay_provenance(cfg, ck, vars(a))}
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(rep, out)
    print(json.dumps(rep["stats"], indent=1))


if __name__ == "__main__":
    main()
