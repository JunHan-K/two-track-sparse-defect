import copy
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
def _deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path, overrides=None):
    """Load YAML config. `_base_` (path relative to the file) is merged first.

    overrides: list of "a.b.c=value" strings (value parsed as YAML).
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = _deep_update(load_config(path.parent / base), cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve(path):
    """Resolve a path relative to the project root (absolute paths unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def enable_tf32():
    """TF32 matmul/conv on Ampere+: fp32 range (no overflow) at near-fp16 speed. Used everywhere."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(seed, deterministic=False):
    import torch

    enable_tf32()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def worker_init_fn(worker_id):
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def environment_info():
    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available()
        else None,
        "numpy": np.__version__,
    }
    for mod in ["transformers", "segmentation_models_pytorch", "scipy", "PIL"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        # uncommitted changes mean the commit hash alone does not identify the code
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).strip())
    except Exception:
        info["git_commit"] = info["git_dirty"] = None
    return info


# ----------------------------------------------------------------------------
# Provenance: every number must be traceable to a config, checkpoint and split
# ----------------------------------------------------------------------------
def file_sha256(path, n=16):
    import hashlib

    h = hashlib.sha256()
    with open(resolve(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def splits_sha256(split):
    """Hash of one split CSV or a list of them (order-sensitive)."""
    files = split if isinstance(split, (list, tuple)) else [split]
    return "+".join(file_sha256(f, 12) for f in files)


def rel(path):
    """Project-relative path string (absolute paths outside the project are kept)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def new_run_id():
    import time
    import uuid

    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def prepare_out_dir(out_dir, overwrite=False):
    """Refuse to mix artefacts of different runs in one experiment directory.

    Existing non-empty directory: error, or with overwrite=True it is renamed to
    <dir>.old-<timestamp> (never deleted). Resuming is not supported.
    """
    import time

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out_dir} already contains a run. Use a new --exp-id, or --overwrite to move "
                             f"the old run to {out_dir.name}.old-<timestamp> (resume is not supported).")
        backup = out_dir.with_name(f"{out_dir.name}.old-{time.strftime('%Y%m%d-%H%M%S')}")
        out_dir.rename(backup)
        print(f"[info] previous run moved to {backup}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ----------------------------------------------------------------------------
# Results CSV (one row per run, head and split)
# ----------------------------------------------------------------------------
RESULT_COLUMNS = [
    "experiment_id", "run_id", "project", "model", "backbone", "dataset", "subset", "seed", "split_version",
    "train_split_sha", "input_height", "input_width", "batch_size", "effective_batch_size", "optimizer",
    "learning_rate", "weight_decay", "epochs", "best_epoch", "loss",
    "replay_type", "eval_head", "eval_split", "threshold_source", "threshold",
    "pixel_ap", "image_ap", "image_auroc", "image_ap_cls", "foreground_iou", "precision", "recall", "f1", "normal_fp_area",
    "max_recall", "threshold_val_r90", "fp_at_val_r90", "recall_at_val_r90",
    "ratio_small_ap", "ratio_medium_ap", "ratio_large_ap", "ratio_small_recall", "ratio_medium_recall",
    "ratio_large_recall", "defect_small_ap", "defect_medium_ap", "defect_large_ap", "defect_det",
    "defect_small_det", "defect_medium_det", "defect_large_det", "defect_small_cov", "defect_medium_cov",
    "defect_large_cov", "cat_macro_ap", "params_m", "flops_g", "latency_ms", "train_minutes_total",
    "checkpoint", "checkpoint_sha", "git_commit", "git_dirty", "notes",
]


def append_result(row, path="results/experiments.csv"):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


class Logger:
    def __init__(self, path=None):
        self.f = open(path, "a") if path else None

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()
