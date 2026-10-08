"""Regression tests for paths that could silently corrupt paper results."""
import json

import pytest
import torch

from sds.mining import check_replay_provenance, replay_provenance, select_tau
from sds.models.segformer import SegFormerSDS
from sds.utils import prepare_out_dir


def test_select_tau_does_not_assume_lower_threshold_gives_more_components():
    # reviewer case: 5 components at 0.5 but regions merge into 1 at lower thresholds
    counts = {0.5: 5, 0.3: 3, 0.1: 1, 0.01: 1}
    assert select_tau(counts, target=400) == 0.5
    assert select_tau({0.5: 2, 0.3: 7, 0.1: 7}, target=400) == 0.3  # tie -> higher tau
    assert select_tau({0.5: 10, 0.3: 500, 0.1: 900}, target=400) == 0.3  # highest tau reaching target


def test_prepare_out_dir_refuses_to_mix_runs(tmp_path):
    d = tmp_path / "exp"
    prepare_out_dir(d)
    (d / "best.pt").write_text("old")
    with pytest.raises(SystemExit):
        prepare_out_dir(d)
    prepare_out_dir(d, overwrite=True)  # old run is moved aside, never deleted
    assert not (d / "best.pt").exists()
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("exp.old-")]
    assert len(backups) == 1 and (backups[0] / "best.pt").read_text() == "old"


def test_missing_pretrained_weights_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        SegFormerSDS(pretrained=str(tmp_path / "does_not_exist"))
    SegFormerSDS(pretrained=None)  # random init only when explicitly requested


def _cfg(tmp_path, input_size=(64, 32)):
    for name in ("core", "mining"):
        (tmp_path / f"{name}.csv").write_text("id,image,mask,label,fg_ratio,height,width\n")
    return {"data": {"input_size": list(input_size), "mask_resize_threshold": 0.5,
                     "core_split": str(tmp_path / "core.csv"), "mining_split": str(tmp_path / "mining.csv")}}


def test_replay_provenance_detects_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"x": 1}, a)
    torch.save({"x": 2}, b)
    replay = {"provenance": replay_provenance(cfg, a)}
    check_replay_provenance(replay, cfg, a)  # consistent -> ok
    with pytest.raises(SystemExit):  # refinement from a different checkpoint
        check_replay_provenance(replay, cfg, b)
    with pytest.raises(SystemExit):  # input resolution changed (patch centres are in resized coords)
        check_replay_provenance(replay, _cfg(tmp_path, (128, 64)), a)
    (tmp_path / "mining.csv").write_text("id,image,mask,label,fg_ratio,height,width\nx,y,,0,0,1,1\n")
    with pytest.raises(SystemExit):  # split file changed after mining
        check_replay_provenance(replay, cfg, a)
    with pytest.raises(SystemExit):  # legacy replay without provenance
        check_replay_provenance({}, cfg, a)

