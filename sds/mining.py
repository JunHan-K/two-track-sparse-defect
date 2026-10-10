"""Replay-provenance helpers shared by tools/mine_native_fp.py and tools/train.py."""
import numpy as np
from scipy import ndimage

from .utils import environment_info, file_sha256, rel, splits_sha256


def components(cand, score, min_area):
    """8-connected components of a boolean candidate map -> list of dicts (centroid, area, mean score)."""
    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    ones = np.ones_like(score)
    area = ndimage.sum(ones, lab, idx)
    mean = ndimage.mean(score, lab, idx)
    cms = ndimage.center_of_mass(ones, lab, idx)
    return [{"cy": float(c[0]), "cx": float(c[1]), "area": int(a), "score": float(s)}
            for a, s, c in zip(area, mean, cms) if a >= min_area]


def gt_components(mask):
    lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3)))
    out = []
    for j, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = np.nonzero(lab[sl] == j)
        out.append({"cy": float(ys.mean() + sl[0].start), "cx": float(xs.mean() + sl[1].start), "area": int(len(ys))})
    return out


def select_tau(counts, target):
    """counts: {tau: #components}. Highest tau reaching `target`; otherwise the tau with the
    MOST components (ties -> higher tau). Lowering tau can merge regions and reduce the count,
    so the lowest tau is never assumed to maximise the candidates."""
    taus = sorted(counts, reverse=True)
    for t in taus:
        if counts[t] >= target:
            return t
    best = max(counts[t] for t in taus)
    return next(t for t in taus if counts[t] == best)


def replay_provenance(cfg, ckpt_path, args=None):
    data = cfg["data"]
    return {
        "source_checkpoint": rel(ckpt_path),
        "source_checkpoint_sha": file_sha256(ckpt_path),
        "source_run_id": cfg.get("run_id"),
        "input_size": list(data["input_size"]),
        "mask_resize_threshold": data.get("mask_resize_threshold", 0.5),
        "core_split": data["core_split"], "core_split_sha": splits_sha256(data["core_split"]),
        "mining_split": data["mining_split"], "mining_split_sha": splits_sha256(data["mining_split"]),
        "git_commit": environment_info().get("git_commit"),
        "args": args or {},
    }


def check_replay_provenance(replay, cfg, init_checkpoint):
    """Raise if the replay set was not mined from `init_checkpoint` with the current data settings."""
    pv = replay.get("provenance")
    if pv is None:
        raise SystemExit("replay file has no provenance; re-run tools/mine_native_fp.py")
    data = cfg["data"]
    problems = []
    if pv["source_checkpoint_sha"] != file_sha256(init_checkpoint):
        problems.append(f"mined from {pv['source_checkpoint']} (sha {pv['source_checkpoint_sha']}) but "
                        f"refinement starts from {rel(init_checkpoint)} (sha {file_sha256(init_checkpoint)})")
    if list(pv["input_size"]) != list(data["input_size"]):
        problems.append(f"input_size {pv['input_size']} != {data['input_size']} (patch centres are in resized coords)")
    if pv["mask_resize_threshold"] != data.get("mask_resize_threshold", 0.5):
        problems.append("mask_resize_threshold differs")
    for k in ("core_split", "mining_split"):
        if pv[f"{k}_sha"] != splits_sha256(data[k]):
            problems.append(f"{k} file changed since mining")
    if problems:
        raise SystemExit("replay provenance mismatch:\n  - " + "\n  - ".join(problems))
