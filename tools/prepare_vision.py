"""VISION Datasets (CVPR 2023 workshop, 14 categories, polygon defect annotations) as a supervised
sparse-defect segmentation benchmark.

  python tools/prepare_vision.py [--root data/VISION] [--seed 42]

* Polygons (COCO) are rasterised into binary PNG masks (union per image) under data/VISION_masks.
* Category selection rule (fixed BEFORE any VISION result): a category is kept if the median
  "rectangularity" of its polygons (polygon area / area of the minimum rotated bounding rectangle) is
  below --max-rect (0.9). Box/diamond-like region annotations (~1.0) do not follow the defect shape and
  are excluded from the main benchmark; all categories are still listed in the report.
* Splits: test = official labelled `val`; official `train` -> Core A / Mining B / Validation V (70/15/15)
  per category. There are no labelled normal images (the `inference` split is unlabelled), so normal-
  image metrics are undefined; false positives are measured on the background of defect images.
"""
import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sds.utils import PROJECT_ROOT  # noqa: E402

FIELDS = ["id", "image", "mask", "label", "fg_ratio", "height", "width"]


def poly_area(p):
    x, y = np.asarray(p[0::2], float), np.asarray(p[1::2], float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rectangularity(p):
    x, y = np.asarray(p[0::2], float), np.asarray(p[1::2], float)
    a = poly_area(p)
    if a <= 0 or len(x) < 3:
        return 1.0
    best = np.inf
    for t in np.deg2rad(np.arange(0, 90, 1.0)):  # minimum rotated bounding rectangle (1 degree steps)
        c, s = np.cos(t), np.sin(t)
        u, v = c * x + s * y, -s * x + c * y
        best = min(best, (u.max() - u.min()) * (v.max() - v.min()))
    return float(a / best) if best > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/VISION")
    ap.add_argument("--out-masks", default="data/VISION_masks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rect", type=float, default=0.9)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root, mroot = PROJECT_ROOT / a.root, PROJECT_ROOT / a.out_masks
    split_dir = PROJECT_ROOT / "data/splits/vision"
    if (split_dir / "core.csv").exists() and not a.force:
        raise SystemExit(f"split exists in {split_dir} (use --force)")
    rng = np.random.default_rng(a.seed)
    cats = sorted(p.name for p in root.iterdir() if (p / "train").is_dir())
    report = {"rule": f"keep category if median polygon rectangularity < {a.max_rect}", "categories": {}}
    rows = {"train": [], "test": []}
    kept = []
    for cat in cats:
        rects, info = [], {}
        for sp, dst in (("train", "train"), ("val", "test")):
            d = json.load(open(glob.glob(str(root / cat / sp / "*.json"))[0]))
            anns = {}
            for an in d["annotations"]:
                anns.setdefault(an["image_id"], []).append(an)
            for im in d["images"]:
                w, h = im["width"], im["height"]
                m = Image.new("L", (w, h), 0)
                dr = ImageDraw.Draw(m)
                for an in anns.get(im["id"], []):
                    for p in an["segmentation"]:
                        if len(p) >= 6:
                            dr.polygon([(p[i], p[i + 1]) for i in range(0, len(p), 2)], fill=255)
                            rects.append(rectangularity(p))
                n_fg = int((np.asarray(m) > 0).sum())
                stem = Path(im["file_name"]).stem
                mp = mroot / cat / sp / f"{stem}.png"
                mp.parent.mkdir(parents=True, exist_ok=True)
                m.save(mp)
                img_p = root / cat / sp / im["file_name"]
                if not img_p.exists():
                    raise SystemExit(f"missing image {img_p}")
                rows[dst].append({"id": f"{cat}_{sp}_{stem}", "image": str(img_p.relative_to(PROJECT_ROOT)),
                                  "mask": str(mp.relative_to(PROJECT_ROOT)) if n_fg else "", "label": int(n_fg > 0),
                                  "fg_ratio": n_fg / (w * h), "height": h, "width": w, "_cat": cat})
        med = float(np.median(rects)) if rects else 1.0
        keep = med < a.max_rect
        info.update({"median_rectangularity": round(med, 3), "n_polygons": len(rects), "kept": bool(keep)})
        report["categories"][cat] = info
        if keep:
            kept.append(cat)
    report["kept"] = kept
    sp = {"core": [], "mining": [], "val": []}
    for cat in kept:
        grp = [x for x in rows["train"] if x["_cat"] == cat]
        idx = rng.permutation(len(grp))
        b1, b2 = int(round(0.70 * len(grp))), int(round(0.85 * len(grp)))
        sp["core"] += [grp[i] for i in idx[:b1]]
        sp["mining"] += [grp[i] for i in idx[b1:b2]]
        sp["val"] += [grp[i] for i in idx[b2:]]
    test = [x for x in rows["test"] if x["_cat"] in kept]
    split_dir.mkdir(parents=True, exist_ok=True)

    def write(rs, name):
        with open(split_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rs, key=lambda x: x["id"]))

    for k, v in sp.items():
        write(v, k)
    write(test, "test")
    write([x for x in rows["train"] if x["_cat"] in kept], "train_all")
    sizes = Counter(f"{x['height']}x{x['width']}" for x in rows["train"] + rows["test"] if x["_cat"] in kept)
    report["counts"] = {k: {"n": len(v), "pos": sum(x["label"] for x in v)} for k, v in {**sp, "test": test}.items()}
    report["image_sizes"] = dict(sizes.most_common(12))
    with open(split_dir / "sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
