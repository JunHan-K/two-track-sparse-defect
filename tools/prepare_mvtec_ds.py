"""MVTec AD split following Defect Spectrum (Yang et al., ECCV 2024), supplementary "Experimental Settings":
"Since there was no train-test split in MVTec AD dataset, ... we employed 5 images for each defective type
per object, which is the same as our segmentation training setting."

  python tools/prepare_mvtec_ds.py [--root data/MVTec_AD] [--seed 42] [--per-type 5] [--val-frac 0.2]

Per category and defect type (seed 42): `per_type` defect images -> training (Core); of the remaining
defect images, `val_frac` (at least one) -> validation, the rest -> test. Defect images only (the protocol
does not use normal images); Mining is empty (kept so that every config reads the same keys).
Ids are "<category>_<defect type>_<stem>".
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sds.utils import PROJECT_ROOT  # noqa: E402

FIELDS = ["id", "image", "mask", "label", "fg_ratio", "height", "width"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MVTec_AD")
    ap.add_argument("--out", default="data/splits/mvtec_ds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()
    root, out = PROJECT_ROOT / a.root, PROJECT_ROOT / a.out
    rng = np.random.default_rng(a.seed)
    sp = {"core": [], "mining": [], "val": [], "test": []}
    issues = []
    for cat in sorted(p.name for p in root.iterdir() if (p / "test").is_dir()):
        for dtype in sorted(p.name for p in (root / cat / "test").iterdir() if p.is_dir() and p.name != "good"):
            rows = []
            for img in sorted((root / cat / "test" / dtype).glob("*.png")):
                m = root / cat / "ground_truth" / dtype / f"{img.stem}_mask.png"
                if not m.exists():
                    issues.append(f"no mask {img}")
                    continue
                g = np.asarray(Image.open(m)) > 0
                if not g.any():
                    issues.append(f"empty mask {m}")
                    continue
                rows.append({"id": f"{cat}_{dtype}_{img.stem}", "image": str(img.relative_to(PROJECT_ROOT)),
                             "mask": str(m.relative_to(PROJECT_ROOT)), "label": 1, "fg_ratio": float(g.mean()),
                             "height": g.shape[0], "width": g.shape[1]})
            idx = rng.permutation(len(rows))
            k = min(a.per_type, len(rows))
            rest = [rows[i] for i in idx[k:]]
            nv = max(1, int(round(a.val_frac * len(rest)))) if len(rest) > 1 else 0
            sp["core"] += [rows[i] for i in idx[:k]]
            sp["val"] += rest[:nv]
            sp["test"] += rest[nv:]
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in list(sp.items()) + [("train_all", sp["core"])]:
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(sorted(rs, key=lambda x: x["id"]))
    rep = {"protocol": f"Defect Spectrum (ECCV 2024): {a.per_type} defect images per defect type per object for "
                       f"training; remaining defect images: {a.val_frac:.0%} validation, rest test (seed {a.seed}); "
                       "defect images only", "issues": issues,
           "counts": {k: len(v) for k, v in sp.items()},
           "defect_types": len({r["id"].rsplit("_", 1)[0] for r in sp["core"]}),
           "per_category": {k: dict(Counter(r["id"].split("_")[0] for r in v)) for k, v in sp.items() if v}}
    with open(out / "sanity_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in ("protocol", "counts", "defect_types", "issues")}, indent=1))


if __name__ == "__main__":
    main()
