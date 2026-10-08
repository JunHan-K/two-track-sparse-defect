"""Sanity check of our small-target wrappers on the infrared small-target benchmarks their authors report.

  python tools/external/irstd_sanity.py --model dnanet [--epochs 1500]
  python tools/external/irstd_sanity.py --model mshnet [--epochs 400]

Question: do sds/models/dnanet.py and sds/models/mshnet.py (the authors' networks + their losses, as used on VISION)
reach the authors' accuracy in the authors' setting? If yes, their failure on VISION is not an implementation bug.
Everything except the model wrappers is the authors' code: DNANet's TrainSetLoader/TestSetLoader (random scale
0.5-2.0, flip, blur, 256^2 crops; test resize 256^2) and mIoU metric (third_party/DNANet/model/{utils,metric}.py),
on NUAA-SIRST with DNANet's 50/50 split (DNANet) and MSHNet's IRSTD_Dataset / metric on IRSTD-1k (MSHNet). Recipes: DNANet = Adagrad 0.05, cosine to 1e-5, 1500 epochs, batch 16,
SoftIoU on the 4 deep-supervision outputs; MSHNet = Adagrad 0.05 constant, 400 epochs, batch 4, SLS loss, warm-up 5.
References (best test epoch, as the authors' training scripts save): DNANet-ResNet18 on NUAA-SIRST mIoU 77.47;
MSHNet on IRSTD-1k mIoU 67.16. Data (not in git): github.com/YimianDai/sirst; IRSTD-1k from the ISNet release
(github.com/RuiZhang97/ISNet, MIT).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DNA = ROOT / "third_party" / "DNANet"
MSH = ROOT / "third_party" / "MSHNet"
# each model is checked on the dataset its authors report with that model's own data pipeline and metric:
# DNANet on NUAA-SIRST (mIoU 77.47, ResNet18), MSHNet on IRSTD-1k (mIoU 67.16, CVPR'24 release; README update 67.87)
RECIPES = {"dnanet": dict(epochs=1500, batch=16, lr=0.05, cosine=True, ref=77.47, data="NUAA-SIRST"),
           "mshnet": dict(epochs=400, batch=4, lr=0.05, cosine=False, ref=67.16, data="IRSTD-1k")}


def load(path, name):
    """Import one of the authors' files by path (both repos have a top-level package called 'model')."""
    import importlib.util
    import types

    if "skimage" not in sys.modules:  # their metric files import scikit-image only for PD_FA (unused here)
        try:
            import skimage  # noqa: F401
        except ImportError:
            stub = types.ModuleType("skimage")
            stub.measure = types.ModuleType("skimage.measure")
            sys.modules["skimage"], sys.modules["skimage.measure"] = stub, stub.measure
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare(name):
    """images/<id>.png and masks/<id>.png (+ split lists) as the authors' loaders expect; links, no copies."""
    data = ROOT / "data" / "irstd" / name
    for d in ("images", "masks"):
        (data / d).mkdir(parents=True, exist_ok=True)
    if name == "NUAA-SIRST":
        src = ROOT / "data" / "irstd" / "sirst"
        pairs = [(f, src / "masks" / f"{f.stem}_pixels0.png") for f in (src / "images").glob("*.png")]
    else:
        src = ROOT / "data" / "irstd" / "IRSTD-1k_raw" / "IRSTD-1k"
        pairs = [(f, src / "IRSTD1k_Label" / f.name) for f in (src / "IRSTD1k_Img").glob("*.png")]
        for t in ("trainval.txt", "test.txt"):
            if not (data / t).exists():
                os.symlink(src / t, data / t)
    for img, msk in pairs:
        for d, s in (("images", img), ("masks", msk)):
            t = data / d / img.name
            if not t.exists():
                os.symlink(s, t)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RECIPES))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    R = RECIPES[a.model]
    epochs = a.epochs or R["epochs"]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    data = prepare(R["data"])
    if a.model == "dnanet":
        U = load(DNA / "model" / "utils.py", "dna_utils")
        mIoU = load(DNA / "model" / "metric.py", "dna_metric").mIoU
        split = DNA / "dataset" / "NUAA-SIRST" / "50_50"
        tr_ids = [l.strip() for l in open(split / "train.txt") if l.strip()]
        te_ids = [l.strip() for l in open(split / "test.txt") if l.strip()]
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trs = U.TrainSetLoader(str(data), tr_ids, base_size=256, crop_size=256, transform=tf)
        tes = U.TestSetLoader(str(data), te_ids, base_size=256, crop_size=256, transform=tf)
    else:
        D = load(MSH / "utils" / "data.py", "msh_data")
        mIoU = load(MSH / "utils" / "metric.py", "msh_metric").mIoU
        args = argparse.Namespace(dataset_dir=str(data), base_size=256, crop_size=256)
        trs, tes = D.IRSTD_Dataset(args, mode="train"), D.IRSTD_Dataset(args, mode="val")
    tr = DataLoader(trs, R["batch"], shuffle=True, num_workers=4, drop_last=True)
    te = DataLoader(tes, 8 if a.model == "dnanet" else 1, num_workers=4)
    dev = torch.device("cuda")
    if a.model == "dnanet":
        from sds.models.dnanet import DNANetSDS
        model = DNANetSDS(loss="softiou").to(dev)
    else:
        from sds.models.mshnet import MSHNetSDS
        model = MSHNetSDS(warm_epoch=5, loss="sls").to(dev)
    opt = torch.optim.Adagrad(model.parameters(), lr=R["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5) if R["cosine"] else None
    out = ROOT / "outputs" / f"SANITY_IRSTD_{a.model.upper()}_{R['data'].replace('-', '')}"
    out.mkdir(parents=True, exist_ok=True)
    metric, hist, best = mIoU(1), [], (-1.0, -1)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(ep)
        ls = []
        for img, mask in tr:
            img, mask = img.to(dev), mask.to(dev)
            o = model(img)
            loss = model.loss_fn(o, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(float(loss))
        if sched is not None:
            sched.step()
        if (ep + 1) % a.eval_every == 0 or ep + 1 == epochs:
            model.eval()
            metric.reset()
            with torch.no_grad():
                for img, mask in te:
                    metric.update(model(img.to(dev))["logits"].cpu(), mask)
            iou = float(metric.get()[1]) * 100
            best = max(best, (iou, ep + 1))
            hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "mIoU": iou})
            print(f"[{a.model}] epoch={ep + 1} loss={np.mean(ls):.4f} test_mIoU={iou:.2f} best={best[0]:.2f}@{best[1]} "
                  f"min={(time.time() - t0) / 60:.1f}", flush=True)
    res = {"model": a.model, "dataset": R["data"], "epochs": epochs, "recipe": R,
           "last_mIoU": hist[-1]["mIoU"], "best_mIoU": best[0], "best_epoch": best[1], "reference_mIoU": R["ref"],
           "history": hist}
    json.dump(res, open(out / "sanity.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}))


if __name__ == "__main__":
    main()
