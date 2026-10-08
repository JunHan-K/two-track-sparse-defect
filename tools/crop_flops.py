"""Exact FLOPs of one native 384^2 crop pass, stored as efficiency.json["crop_flops_g"] (used by the cost table and
Fig. 3 instead of scaling the full-input FLOPs by the pixel ratio, which overestimates MiT: 12.9 vs 8.9 GFLOPs).

  python tools/crop_flops.py --exp outputs/vision_ours [--crop 384]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sds.models import build_model  # noqa: E402
from sds.utils import load_config, resolve  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--crop", type=int, default=384)
    a = ap.parse_args()
    for exp in a.exp:
        exp = resolve(exp)
        m = build_model(load_config(exp / "config.yaml")).eval()
        with FlopCounterMode(display=False) as fc:
            m(torch.zeros(1, 3, a.crop, a.crop), return_aux=False)
        f = exp / "efficiency.json"
        e = json.load(open(f))
        e["crop_flops_g"], e["crop_hw"] = fc.get_total_flops() / 1e9, [a.crop, a.crop]
        json.dump(e, open(f, "w"), indent=1)
        print(exp.name, f"{e['crop_flops_g']:.2f} GFLOPs per {a.crop}^2 crop (full input {e['flops_g']:.2f})")


if __name__ == "__main__":
    main()
