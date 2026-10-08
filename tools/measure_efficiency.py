"""Params / FLOPs / batch-1 latency of the INFERENCE graph (aux heads excluded).

  python tools/measure_efficiency.py --exp outputs/vision_ours
  python tools/measure_efficiency.py --config configs/vision/baselines/segformer_b0.yaml   # no checkpoint needed

Latency protocol: batch 1, same resolution, 20 warm-up + 100 timed iterations,
CUDA synchronize around timing, mean (and std) reported. FLOPs counted with
torch.utils.flop_counter (multiply-add = 2 FLOPs); MACs = FLOPs / 2.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sds.models import build_model  # noqa: E402
from sds.utils import enable_tf32, load_config, resolve, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--amp", action="store_true", help="measure latency under fp16 autocast")
    args = ap.parse_args()
    cfg = load_config(resolve(args.exp) / "config.yaml" if args.exp else args.config)
    enable_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()

    params_inf = sum(p.numel() for m in model.inference_modules() for p in m.parameters())
    params_all = sum(p.numel() for p in model.parameters())
    H, W = cfg["data"]["input_size"]
    x = torch.randn(1, 3, H, W, device=device)

    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        def grid_sample_flops(inp_shape, grid_shape, *args, out_shape=None, **kw):
            # not counted by default (Mask2Former's deformable attention): bilinear = 4 x (mul + add) per output value
            return inp_shape[0] * inp_shape[1] * grid_shape[1] * grid_shape[2] * 8

        with torch.no_grad(), FlopCounterMode(display=False,
                                              custom_mapping={torch.ops.aten.grid_sampler_2d: grid_sample_flops}) as fc:
            model(x, return_aux=False)
        flops = fc.get_total_flops()
    except Exception as e:  # older torch
        print(f"[warn] FLOP counting unavailable: {e}")

    times = []
    with torch.no_grad(), torch.autocast("cuda", enabled=args.amp and device.type == "cuda"):
        for i in range(args.warmup + args.iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            model(x, return_aux=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= args.warmup:
                times.append((time.perf_counter() - t) * 1000)
    res = {
        "params_m": round(params_inf / 1e6, 4), "params_train_m": round(params_all / 1e6, 4),
        "flops_g": round(flops / 1e9, 3) if flops else None, "macs_g": round(flops / 2e9, 3) if flops else None,
        "latency_ms": round(float(np.mean(times)), 3), "latency_std_ms": round(float(np.std(times)), 3),
        "input_hw": [H, W], "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp": args.amp, "batch_size": 1, "warmup": args.warmup, "iters": args.iters,
    }
    print(res)
    if args.exp:
        save_json(res, resolve(args.exp) / "efficiency.json")


if __name__ == "__main__":
    main()
