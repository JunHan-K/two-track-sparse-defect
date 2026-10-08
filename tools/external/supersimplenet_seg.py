"""SuperSimpleNet (Rolih et al., ICPR 2024 / JIMS 2025) retrained on our fixed splits (MVTec AD, Defect Spectrum).

  python tools/external/supersimplenet_seg.py --dataset mvtec_ds [--seed 42]            # train (authors' recipe)
  python tools/external/supersimplenet_seg.py --dataset mvtec_ds --eval-only --test [--tol 3]

Generalised from supersimplenet_ksdd2.py (2026-10-04): the authors' supervised recipe is kept; per dataset
the image size (= our input size, plain resize as in the authors' code) and batch size (memory) change.
For VISION, which has no labelled normal images, the sampler uses only the annotated defect images
instead of the authors' balanced normal/defect sampling. Masks are read from our split CSVs (0/255 PNG).

Faithful to the authors' supervised KSDD2 recipe (train.py: run_sup / main_ksdd2): same config
(WideResNet50-2 layer2+3, 640x232 resize, flips, dilate 7, distance transform (3, 2), batch 32,
300 epochs, AdamW + MultiStepLR, focal + truncated-L1 loss) and their own `train()` function.
Only the data differ: training on Core A + Mining B (= the labelled data of our final model)
instead of the full official train set. The authors' code evaluates on the official test split
during training; here that slot is the validation split, so the official test is untouched until
`--test`. Like the authors, the LAST epoch is used (no checkpoint selection).

Evaluation uses OUR evaluator (sds.metrics) at the original resolution: pixel probability =
sigmoid(anomaly map) (no test-set min-max normalisation), thresholds selected on validation.
image_ap = max pixel probability (as for our models); image_ap_cls = the authors' classification
head (their reported AP-det).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sds.data import read_split  # noqa: E402
from sds.data.dataset import load_mask  # noqa: E402
from sds.metrics import PixelEvaluator, ap_from_scores, auroc_from_scores, metrics_at, select_thresholds  # noqa: E402
from sds.utils import (append_result, environment_info, file_sha256, prepare_out_dir, rel, resolve,  # noqa: E402
                       save_json, splits_sha256)

DATASETS = {
    # the authors' supervised recipe was written for KSDD2; its dataset class is reused with our split files
    "ksdd2": dict(root="data/KSDD2", size=(640, 232), batch=32, name="KSDD2", version="ksdd2_seed42_v1",
                  out="ksdd2_supersimplenet"),
    "vision": dict(root="data/VISION", size=(1024, 1024), batch=8, name="VISION", version="vision_seed42_v1",
                   out="vision_supersimplenet"),
    # Defect Spectrum protocol (tools/prepare_mvtec_ds.py): 5 defect images per defect type, defect images only
    "mvtec_ds": dict(root="data/MVTec_AD", size=(512, 512), batch=16, name="MVTDS", version="mvtec_ds_seed42_v1",
                     out="mvtec_supersimplenet"),
}
for _k, _d in DATASETS.items():
    _d.update({s: f"data/splits/{_k}/{s}.csv" for s in ("core", "mining", "val", "test")})
KSDD2 = DATASETS["ksdd2"]


def ssn_config(seed, out_dir, batch=32):
    """The authors' run_sup() config for KSDD2 (N246 = fully segmented), verbatim except paths/seed."""
    return {
        "wandb_project": "ssn", "datasets_folder": None, "num_workers": 1, "setup_name": "superSimpleNet",
        "dt": (3, 2), "dilate": 7, "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
        "patch_size": 3, "noise": True, "perlin": True, "no_anomaly": "empty", "bad": True, "overlap": False,
        "adapt_cls_feat": False, "noise_std": 0.015, "perlin_thr": 0.6, "seed": seed, "batch": batch,
        "epochs": 300, "flips": True, "seg_lr": 0.0002, "dec_lr": 0.0002, "adapt_lr": 0.0001, "gamma": 0.4,
        "stop_grad": False, "clip_grad": True, "eval_step_size": 300, "results_save_path": out_dir,
        "ratio": 246, "dataset": "ksdd2", "category": "ksdd2",
    }


def make_datasets(ssn, splits_train, split_monitor, D=KSDD2):
    """Authors' KSDD2 dataset class, with make_dataset() reading our split CSVs."""
    import pandas as pd
    from datamodules.base import Supervision
    from datamodules.base.datamodule import SSNDataModule
    from datamodules.ksdd2 import KSDD2Dataset

    def frame(csvs):
        rows = read_split(csvs)
        df = pd.DataFrame([{
            "path": str(resolve(D["root"])), "sample_id": r["id"], "split": "train",
            "image_path": str(resolve(r["image"])),
            # masks from our CSVs; normal images (label 0) are given an all-zero mask by the dataset class
            "mask_path": str(resolve(r["mask"])) if r["mask"] else "",
            "is_segmented": True, "label_index": int(r["label"]),
        } for r in rows])
        return (df[df.label_index == 0].reset_index(), df[df.label_index == 1].reset_index())

    class OurSplit(KSDD2Dataset):
        def __init__(self, csvs, **kw):
            self.csvs = csvs
            super().__init__(**kw)

        def make_dataset(self):
            return frame(self.csvs)

        def _setup(self):
            # VISION has labelled defect images only. The author's balanced
            # sampler otherwise samples from an empty normal-image collection.
            # Keep every real label; do not invent normal images or empty masks.
            neg, pos = self.make_dataset()
            self.positive_only = len(neg) == 0 and len(pos) > 0
            if not self.positive_only:
                return super()._setup()
            self.counter = 0
            self._normal_samples, self._anomalous_samples = neg, pos
            self.num_neg, self.num_pos = 0, len(pos)
            self.generated_num_pos = self.num_pos * (4 if self.flips else 1)
            self.neg_retrieval_freq = np.zeros(0)
            self.neg_imgs_permutation = np.zeros(0, dtype=int)
            self.replace = False

        def __len__(self):
            if getattr(self, "positive_only", False) and self.split == Split.TRAIN:
                return self.generated_num_pos
            return super().__len__()

        def generate_permutation(self):
            if getattr(self, "positive_only", False):
                return
            return super().generate_permutation()

    from anomalib.data.utils import Split
    from datamodules import ksdd2

    dm = SSNDataModule.__new__(SSNDataModule)
    SSNDataModule.__init__(dm, root=resolve(D["root"]), supervision=Supervision.MIXED_SUPERVISION,
                           image_size=D["size"], train_batch_size=ssn["batch"],
                           eval_batch_size=ssn["batch"], num_workers=ssn["num_workers"], seed=ssn["seed"],
                           flips=ssn["flips"])
    common = dict(root=resolve(D["root"]), supervision=Supervision.MIXED_SUPERVISION, normal_flips=False)
    dm.train_data = OurSplit(splits_train, transform=dm.transform_train, split=Split.TRAIN, flips=ssn["flips"],
                             dt=ssn["dt"], dilate=ssn["dilate"], **common)
    dm.test_data = OurSplit(split_monitor, transform=dm.transform_eval, split=Split.TEST, flips=False, **common)
    dm.setup()
    return dm


@torch.no_grad()
def predict(model, transform, csv, device):
    """Yields (row, prob map at original resolution, image score of the cls head)."""
    from anomalib.data.utils import read_image

    model.eval()
    for r in read_split(csv):
        img = read_image(str(resolve(r["image"])))
        x = transform(image=img)["image"][None].to(device)
        amap, score = model(x)
        p = torch.sigmoid(amap.float()).reshape(1, 1, *amap.shape[-2:])
        p = F.interpolate(p, size=(r["height"], r["width"]), mode="bilinear", align_corners=False)[0, 0]
        yield r, p.clamp(0, 1), float(torch.sigmoid(score.float()).flatten()[0])


def size_edges(D=KSDD2):
    from tools.evaluate import size_edges as se  # same Q33/Q66 edges as every other method

    return se(D["core"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssn-root", default="third_party/SuperSimpleNet", help="clone of github.com/blaz-r/SuperSimpleNet")
    ap.add_argument("--dataset", default="mvtec_ds", choices=list(DATASETS))
    ap.add_argument("--tol", type=int, default=0, help="boundary tolerance (see evaluate.py); suffixes the metrics file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp", default=None, help="output dir (default outputs/mvtec_supersimplenet[_s<seed>])")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--test", action="store_true", help="also evaluate the official test split (final only!)")
    ap.add_argument("--epochs", type=int, default=None, help="debug only; the paper uses the authors' 300")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.ssn_root).resolve()))
    from pytorch_lightning import seed_everything

    from model.supersimplenet import SuperSimpleNet
    from datamodules import ksdd2
    import train as ssn_train

    D = DATASETS[args.dataset]
    exp = resolve(args.exp or f"outputs/{D['out']}{'' if args.seed == 42 else f'_s{args.seed}'}")
    if not args.eval_only:
        prepare_out_dir(exp, args.overwrite)
    ssn = ssn_config(args.seed, exp, D["batch"])
    if args.epochs:
        ssn["epochs"] = ssn["eval_step_size"] = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(ssn["seed"], workers=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = SuperSimpleNet(image_size=D["size"], config=ssn)
    train_csvs = [D["core"], D["mining"]]
    dm = make_datasets(ssn, train_csvs, D["val"], D)

    if not args.eval_only:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_json({"ssn_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in ssn.items()},
                   "sampling_protocol": "positive_only" if dm.train_data.positive_only else "authors_balanced",
                   "run_id": run_id, "train_split": train_csvs, "train_split_sha": splits_sha256(train_csvs),
                   "ssn_commit": _git(args.ssn_root), "env": environment_info()}, exp / "config.json")
        from anomalib.utils.metrics import AUROC
        from torchmetrics import AveragePrecision

        t0 = time.time()
        ssn_train.train(model=model, epochs=ssn["epochs"], datamodule=dm, device=device,
                        image_metrics={"I-AUROC": AUROC(), "AP-det": AveragePrecision(num_classes=1)},
                        pixel_metrics={"AP-loc": AveragePrecision(num_classes=1)},  # monitor = validation
                        clip_grad=ssn["clip_grad"], eval_step_size=ssn["eval_step_size"])
        model.save_model(exp)  # weights.pt (frozen backbone excluded, as in the authors' code)
        save_json({"train_minutes_total": (time.time() - t0) / 60, "epochs": ssn["epochs"], "completed": True,
                   "run_id": run_id}, exp / "train_summary.json")
    else:
        model.load_model(exp / "weights.pt")
    model.to(device)

    cfg = json.load(open(exp / "config.json"))
    summ = json.load(open(exp / "train_summary.json"))
    edges, comp_edges = size_edges(D)
    cat_fn = (lambda i: str(i).split("_")[0]) if args.dataset in ("vision", "mvtec", "mvtec_ds") else None
    (exp / "eval").mkdir(exist_ok=True)
    splits = [("val", D["val"])] + ([("test", D["test"])] if args.test else [])
    results, thr = {}, None
    for name, csv in splits:
        ev = PixelEvaluator(True, size_edges=edges, comp_edges=comp_edges, category_fn=cat_fn, tol=args.tol)
        cls_y, cls_s = [], []
        pdir = exp / "predictions" / name
        pdir.mkdir(parents=True, exist_ok=True)
        for r, p, s in predict(model, dm.transform_eval, csv, device):
            gt = torch.from_numpy(load_mask(r["mask"], (r["height"], r["width"]))).to(p.device)
            ev.update(p, gt, r["id"], r["label"])
            if not args.tol:
                np.savez_compressed(pdir / f"{r['id']}.npz", pmain=p.half().cpu().numpy())  # same format as ours
            cls_y.append(r["label"])
            cls_s.append(s)
        st = ev.state()
        np.savez_compressed(exp / "eval" / f"{name}_main.npz", **st)
        if name == "val":
            thr = select_thresholds(st, "f1", 0.9)
        m = metrics_at(st, thr[0], thr[1], edges, comp_edges)
        m["image_ap_cls"], m["image_auroc_cls"] = ap_from_scores(cls_y, cls_s), auroc_from_scores(cls_y, cls_s)
        results[name] = m
        print(f"[SuperSimpleNet] {name}: " + " ".join(f"{k}={m[k]:.4f}" for k in (
            "pixel_ap", "image_ap", "image_ap_cls", "foreground_iou", "precision", "recall", "normal_fp_area",
            "defect_small_ap") if k in m))
        if (args.test and name == "val") or args.tol:
            continue
        w = exp / "weights.pt"
        append_result({**m, "experiment_id": exp.name, "run_id": cfg["run_id"], "project": "SDS",
                       "model": "supersimplenet", "backbone": "wide_resnet50_2 (frozen)", "dataset": D["name"],
                       "seed": args.seed, "split_version": D["version"], "train_split_sha": cfg["train_split_sha"],
                       "input_height": D["size"][0], "input_width": D["size"][1], "batch_size": D["batch"],
                       "effective_batch_size": D["batch"],
                       "optimizer": "AdamW", "epochs": cfg["ssn_config"]["epochs"], "best_epoch": "last",
                       "loss": "focal+trunc_l1", "eval_head": "main", "eval_split": name,
                       "threshold_source": "val_f1", "train_minutes_total": summ.get("train_minutes_total"),
                       "checkpoint": rel(w), "checkpoint_sha": file_sha256(w),
                       "git_commit": environment_info().get("git_commit"),
                       "git_dirty": environment_info().get("git_dirty"),
                       "notes": f"external baseline, authors' recipe, ssn {cfg['ssn_commit']}"})
    save_json({"thresholds": {"main": list(thr)}, "size_edges": edges, "component_size_edges": comp_edges,
               "results": {"main": results}},
              exp / (("metrics_test" if args.test else "metrics_val") + (f"_tol{args.tol}" if args.tol else "") + ".json"))


def _git(root):
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root).decode().strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
