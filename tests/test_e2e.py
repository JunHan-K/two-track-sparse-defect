"""Synthetic end-to-end run of our method (CPU, random-init encoder, a few minutes):
step 1 (two views, size-aware, stage heads) -> mine native false positives -> refinement with Sparse Defect Replay
-> precision / recall evaluation, plus the guards (run-dir reuse, replay/checkpoint mismatch).

Skipped unless SDS_E2E=1:  SDS_E2E=1 python -m pytest tests/test_e2e.py -q
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("SDS_E2E") != "1", reason="set SDS_E2E=1 to run")


def _make_split(root, name, n, rng):
    rows = []
    for i in range(n):
        h, w = 96, 128
        img = (rng.random((h, w, 3)) * 60 + 100).astype(np.uint8)
        m = np.zeros((h, w), np.uint8)
        y, x, k = rng.integers(5, h - 10), rng.integers(5, w - 10), 2 + i % 5
        img[y:y + k, x:x + k] = 250  # defect of varying size (the small ones define the small-defect group)
        m[y:y + k, x:x + k] = 255
        y, x = rng.integers(5, h - 6), rng.integers(5, w - 6)
        img[y:y + 3, x:x + 3] = 215  # defect-like background blob
        iid = f"cat_{name}{i:02d}"
        Image.fromarray(img).save(root / f"{iid}.png")
        Image.fromarray(m).save(root / f"{iid}_GT.png")
        rows.append({"id": iid, "image": str(root / f"{iid}.png"), "mask": str(root / f"{iid}_GT.png"), "label": 1,
                     "fg_ratio": (m > 0).mean(), "height": h, "width": w})
    with open(root / f"{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return str(root / f"{name}.csv")


def _run(*args, ok=True):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    else:
        assert r.returncode != 0, "expected failure"
    return r.stdout + r.stderr


def _config(path, base, extra):
    import yaml

    cfg = yaml.safe_load(open(ROOT / base))
    for k, v in extra.items():
        node = cfg
        *ks, last = k.split(".")
        for p in ks:
            node = node.setdefault(p, {})
        node[last] = v
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    d.mkdir()
    sp = {n: _make_split(d, n, k, rng) for n, k in [("core", 10), ("mining", 6), ("val", 4), ("test", 4)]}
    out = tmp_path / "out"
    common = {"output_root": str(out), "model.pretrained": None, "data.core_split": sp["core"],
              "data.mining_split": sp["mining"], "data.val_split": sp["val"], "data.test_split": sp["test"],
              "data.input_size": [64, 64], "train.epochs": 1, "train.batch_size": 2, "train.eval_batch_size": 2,
              "train.num_workers": 0, "train.zoom.crop": 32, "train.zoom.batch": 2}
    s1 = _config(tmp_path / "s1.yaml", "configs/vision/ours_stage1.yaml", {**common, "data.train_split": sp["core"]})
    _run("tools/train.py", "--config", s1)
    st1 = out / "vision_ours_stage1"
    assert "already contains a run" in _run("tools/train.py", "--config", s1, ok=False)

    rep = tmp_path / "replay.json"
    _run("tools/mine_native_fp.py", "--exp", str(st1), "--out", str(rep), "--min-area", "1")
    r = json.load(open(rep))
    assert len(r["pos"]) > 0 and r["provenance"]["source_checkpoint_sha"]
    # an untrained model may have no clean false positives: add two points to the false positives and to the random
    # background crops allocated by them (the final replay uses the random ones)
    for key in ("native_fp", "random"):
        if len(r["bg"][key]) < 2:
            r["bg"][key] += [{"id": e["id"], "cy": 2.0, "cx": 2.0} for e in r["pos"][:2]]
    json.dump(r, open(rep, "w"))
    r2 = _config(tmp_path / "r2.yaml", "configs/vision/ours.yaml",
                 {**common, "data.train_split": [sp["core"], sp["mining"]], "refine.init_checkpoint": str(st1 / "best.pt"),
                  "refine.replay_file": str(rep), "refine.replay_batch_size": 2})
    _run("tools/train.py", "--config", r2)
    ours = out / "vision_ours"
    for mode in ("precision", "recall"):
        _run("tools/evaluate_zoom.py", "--exp", str(ours), "--mode", mode, "--test", "--no-csv")
    m = json.load(open(ours / "metrics_zoom_main_masked_d16_o0.5_t0.02_n32_test.json"))["results"]["test"]
    assert 0.0 <= m["pixel_ap"] <= 1.0 and "defect_small_ap" in m

    # refinement from a checkpoint the replay file was NOT mined from must be refused
    msg = _run("tools/train.py", "--config", r2, "--exp-id", "bad", "--opts", f"refine.init_checkpoint={ours}/best.pt",
               ok=False)
    assert "replay provenance mismatch" in msg
