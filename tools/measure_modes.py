"""End-to-end latency per VISION image of every model / operating point, on ONE GPU (Table IV).

  python tools/measure_modes.py [--n 100] [--split test] [--out outputs/latency_vision.json]

Protocol: --n images evenly spaced over the split, decoded once into memory (disk I/O excluded for every method); batch 1;
fp32; 10 warm-up images, then torch.cuda.synchronize() around each image; mean and std over images.
"Per image" = everything from the decoded original-resolution image in memory to the original-resolution probability
map, for every method (the letterbox resize + normalisation of sds/data/dataset.py is timed too, see prep()):
  single-pass models   letterbox to the input size, forward, resize back (predict_batch, as in evaluation)
  ours precision       low-resolution pass + candidate crops + merge (tools/evaluate_zoom.zoom_image)
  ours / B0 recall     low-resolution pass + all native tiles + max merge
  MagNet               the coarse-to-fine loop of tools/external/magnet_seg.predict
The image cache replaces PIL.Image.open inside those functions, so they run unchanged.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sds.data.dataset as DS  # noqa: E402
from sds.data import SegDataset  # noqa: E402
from sds.engine import predict_batch, to_device  # noqa: E402
from sds.models import build_model  # noqa: E402
from sds.utils import enable_tf32, load_config, resolve  # noqa: E402

SINGLE = ["vision_segformer_b0", "vision_segformer_b5", "vision_mask2former", "vision_segformer_b0_1536", "vision_unet", "vision_hrnet_w18s",
          "vision_bisenetv2", "vision_deeplabv3p", "vision_dnanet", "vision_mshnet"]
P = SimpleNamespace(roi_source="main", tau=0.02, n_roi=32, merge="masked", mask_dilate=16, outside=0.5)
R = SimpleNamespace(roi_source="tile", tau=0.02, n_roi=32, merge="max", mask_dilate=16, outside=0.5)


class Cache:
    """PIL.Image.open replacement serving decoded images from memory."""

    def __init__(self, paths):
        self.img = {}
        for p in paths:
            with Image.open(p) as im:
                self.img[str(p)] = im.convert("RGB").copy()
        self.orig = Image.open

    def __call__(self, p, *a, **k):
        q = str(p)
        return self.img[q].copy() if q in self.img else self.orig(p, *a, **k)


def timed(fn, items, warm=10):
    ts = []
    for i, it in enumerate(items):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(it)
        torch.cuda.synchronize()
        if i >= warm:
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), statistics.pstdev(ts)


def load_model(exp, device):
    cfg = load_config(ROOT / "outputs" / exp / "config.yaml")
    m = build_model(cfg).to(device).eval()
    m.load_state_dict(torch.load(ROOT / "outputs" / exp / "best.pt", map_location="cpu")["model"])
    return cfg, m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/latency_vision.json")
    a = ap.parse_args()
    enable_tf32()
    dev = torch.device("cuda")
    import tools.evaluate_zoom as EZ
    import tools.external.magnet_seg as MG

    base_cfg = load_config(ROOT / "outputs" / "vision_segformer_b0" / "config.yaml")
    # --n images evenly spaced over the split (the split is ordered by category, so the first n are not representative)
    all_rows = SegDataset(base_cfg["data"][f"{a.split}_split"], base_cfg["data"]["input_size"]).rows
    pick = np.linspace(0, len(all_rows) - 1, a.n).round().astype(int)
    rows_all = [all_rows[j] for j in pick]
    cache = Cache([resolve(r["image"]) for r in rows_all])
    EZ.Image.open = cache
    MG.Image.open = cache
    res = {"gpu": torch.cuda.get_device_name(0), "n_images": a.n, "split": a.split, "warmup_images": 10, "sampling": "evenly spaced",
           "precision": "fp32", "batch": 1, "models": {}}

    DS.Image.open = cache

    def prep(i, input_size):
        """The image half of SegDataset.load_resized + to_tensor (same ops, no mask), then to the GPU."""
        img = DS.load_image(rows_all[i]["image"])
        w, h = img.size
        nh, nw, _ = DS.letterbox_params(h, w, input_size)
        x = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        H, W = input_size
        p = np.zeros((H, W, 3), np.float32)
        p[:nh, :nw] = (x - DS.IMAGENET_MEAN) / DS.IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1)))[None].to(dev)
        return {"image": t, "orig_hw": torch.tensor([[h, w]]), "resized_hw": torch.tensor([[nh, nw]])}

    def check_prep(cfg):
        """prep() must give exactly the tensor the evaluation DataLoader gives."""
        ds = SegDataset(cfg["data"][f"{a.split}_split"], cfg["data"]["input_size"])
        for i in range(3):
            assert torch.equal(prep(i, cfg["data"]["input_size"])["image"].cpu(), ds[int(pick[i])]["image"][None]), \
                "prep mismatch"

    for exp in SINGLE:
        if not (ROOT / "outputs" / exp / "best.pt").exists():
            continue
        cfg, m = load_model(exp, dev)
        check_prep(cfg)
        size = cfg["data"]["input_size"]
        mu, sd = timed(lambda i: predict_batch(m, prep(i, size), ["main"], amp=False, dtype=torch.float32), range(a.n))
        res["models"][exp] = {"latency_ms": mu, "std_ms": sd, "passes": 1}
        print(exp, f"{mu:.1f} ms", flush=True)
        del m
    for exp, modes in (("vision_segformer_b0", {"tiles": R}),
                       ("vision_ours", {"precision": P, "recall": R})):
        cfg, m = load_model(exp, dev)
        check_prep(cfg)
        size = cfg["data"]["input_size"]
        for name, z in modes.items():
            z.crop = 384
            crops = []

            def run(i, z=z, crops=crops):
                base = predict_batch(m, prep(i, size), ["main"], amp=False, dtype=torch.float32)[0]["main"]
                _, k = EZ.zoom_image(m, rows_all[i], base, base.cpu().numpy(), z, dev)
                crops.append(k)
            mu, sd = timed(run, range(a.n))
            res["models"][f"{exp}:{name}"] = {"latency_ms": mu, "std_ms": sd, "mean_crops": float(np.mean(crops))}
            print(exp, name, f"{mu:.1f} ms, {np.mean(crops):.1f} crops", flush=True)
        del m
    # MagNet (unified loss, seed 42), authors' inference loop
    exp = ROOT / "outputs" / "vision_magnet"
    if (exp / "refinement.pt").exists():
        sys.path.insert(0, str(MG.MAGNET))
        from magnet.model.fpn import ResnetFPN
        from magnet.model.refinement import RefinementMagNet
        bb, rf = ResnetFPN(2).to(dev).eval(), RefinementMagNet(2, use_bn=True).to(dev).eval()
        bb.load_state_dict(torch.load(exp / "backbone.pt", map_location="cpu"))
        rf.load_state_dict(torch.load(exp / "refinement.pt", map_location="cpu"), strict=False)
        evals = []

        def run_mg(i):
            _, k = MG.predict(bb, rf, rows_all[i], dev)
            evals.append(k)
        mu, sd = timed(run_mg, range(a.n))
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.zeros(1, 3, MG.BASE, MG.BASE, device=dev)
        with FlopCounterMode(display=False) as fc:
            bb(x)
        g_bb = fc.get_total_flops() / 1e9
        with FlopCounterMode(display=False) as fc:
            rf(torch.zeros(1, 2, MG.BASE, MG.BASE, device=dev), torch.zeros(1, 2, MG.BASE, MG.BASE, device=dev))
        g_rf = fc.get_total_flops() / 1e9
        k = float(np.mean(evals))
        res["models"]["vision_magnet"] = {
            "latency_ms": mu, "std_ms": sd, "mean_patches": k,
            "params_m": (sum(p.numel() for p in bb.parameters()) + sum(p.numel() for p in rf.parameters())) / 1e6,
            "gflops": g_bb * k + g_rf * (k - 1)}
        print("MagNet", f"{mu:.1f} ms, {k:.1f} patches", flush=True)
    out = ROOT / a.out
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
