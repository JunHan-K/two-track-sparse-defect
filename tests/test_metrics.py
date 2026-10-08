import numpy as np
import torch
from scipy import ndimage

from sds.metrics import (THR_BINS, PixelEvaluator, logit_bin, ap_from_scores, ap_from_state, metrics_at, select_thresholds,
                         thr_value)

SIZE_EDGES = (0.003, 0.006)
COMP_EDGES = (0.003, 0.01)  # 4-px components (0.002) are "small"


def _synthetic(seed=0, **kw):
    rng = np.random.default_rng(seed)
    ev, probs, gts, ids = PixelEvaluator(size_edges=SIZE_EDGES, comp_edges=COMP_EDGES, **kw), [], [], []
    for i in range(8):
        g = np.zeros((40, 50), np.uint8)
        if i < 4:
            g[10:12 + 3 * i, 20:25] = 1
            g[30:32, 40:42] = 1  # second, small component
        p = np.clip(rng.random((40, 50)) * 0.6 + g * 0.35, 0, 1).astype(np.float32)
        iid = f"{'catA' if i % 2 else 'catB'}_{i}"
        ev.update(torch.tensor(p), torch.tensor(g), iid, int(g.any()))
        probs.append(p), gts.append(g), ids.append(iid)
    return ev.state(), probs, gts, ids


def _bruteforce_ap(y, s):
    """Reference AP with ties grouped (sklearn average_precision_score definition)."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    ap, prev_r = 0.0, 0.0
    for t in np.unique(s)[::-1]:
        pred = s >= t
        tp = (pred & (y == 1)).sum()
        r, p = tp / y.sum(), tp / pred.sum()
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def test_pixel_ap_matches_bruteforce():
    st, probs, gts, _ = _synthetic()
    P = np.concatenate([p.ravel() for p in probs])
    G = np.concatenate([g.ravel() for g in gts])
    # logit-space histogram: equal to the exact AP on the raw scores up to the bin width
    assert abs(ap_from_state(st) - _bruteforce_ap(G, P)) < 1e-4


def test_pixel_ap_separates_saturated_scores():
    # confident models put many pixels above 0.9999: a positive at 0.999999 must still rank above
    # negatives at 0.99999 (uniform probability bins merged them)
    ev = PixelEvaluator(size_edges=(1.0, 2.0), comp_edges=(1.0, 2.0))
    g = np.zeros((10, 10), np.uint8)
    g[0, 0] = 1
    p = np.full((10, 10), 0.99999, np.float32)
    p[0, 0] = 0.999999
    ev.update(torch.tensor(p), torch.tensor(g), "a", 1)
    m = metrics_at(ev.state(), 0, None, (1.0, 2.0), (1.0, 2.0))
    assert m["pixel_ap"] == 1.0 and m["defect_small_ap"] == 1.0


def test_image_ap_is_tie_invariant():
    # reviewer case: equal scores must not depend on the input order
    assert ap_from_scores([1, 0], [0.5, 0.5]) == ap_from_scores([0, 1], [0.5, 0.5]) == 0.5
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)  # many ties
    perm = rng.permutation(200)
    assert abs(ap_from_scores(y, s) - ap_from_scores(y[perm], s[perm])) < 1e-12
    assert abs(ap_from_scores(y, s) - _bruteforce_ap(y, s)) < 1e-12


def test_group_ap_has_full_precision_at_low_probabilities():
    # reviewer case: positive 0.0009 vs negatives 0.0001 must stay separable in the size-group AP
    ev = PixelEvaluator(size_edges=(1.0, 2.0), comp_edges=(1.0, 2.0))
    g = np.zeros((10, 10), np.uint8)
    g[0, 0] = 1
    p = np.full((10, 10), 0.0001, np.float32)
    p[0, 0] = 0.0009
    ev.update(torch.tensor(p), torch.tensor(g), "a", 1)
    m = metrics_at(ev.state(), 0, None, (1.0, 2.0), (1.0, 2.0))
    assert m["pixel_ap"] == 1.0 and m["ratio_small_ap"] == 1.0 and m["defect_small_ap"] == 1.0


def test_thresholded_metrics_match_bruteforce():
    st, probs, gts, _ = _synthetic()
    k, k90 = select_thresholds(st)
    m = metrics_at(st, k, k90, SIZE_EDGES, COMP_EDGES)
    t = thr_value(k)
    P = np.concatenate([p.ravel() for p in probs])
    G = np.concatenate([g.ravel() for g in gts])
    pr = P >= t
    tp, fp, fn = (pr & (G == 1)).sum(), (pr & (G == 0)).sum(), (~pr & (G == 1)).sum()
    assert abs(m["precision"] - tp / (tp + fp)) < 1e-9
    assert abs(m["recall"] - tp / (tp + fn)) < 1e-9
    assert abs(m["foreground_iou"] - tp / (tp + fp + fn)) < 1e-9
    assert abs(m["normal_fp_area"] - np.mean([(probs[i] >= t).mean() for i in range(4, 8)])) < 1e-9
    assert m["recall_at_val_r90"] >= 0.9  # on the split the threshold was selected on


def test_defect_level_metrics_and_categories():
    st, probs, gts, ids = _synthetic(category_fn=lambda i: i.split("_")[0])
    k = int(logit_bin(torch.tensor([0.5]), THR_BINS)[0])  # grid index of the bin holding 0.5
    t = thr_value(k)
    m = metrics_at(st, k, None, SIZE_EDGES, COMP_EDGES)
    det, cov, areas = [], [], []
    for p, g in zip(probs, gts):
        lab, n = ndimage.label(g, structure=np.ones((3, 3)))
        for j in range(1, n + 1):
            det.append(p[lab == j].max() >= t)
            cov.append((p[lab == j] >= t).mean())
            areas.append((lab == j).sum() / g.size)
    assert len(st["comp_max"]) == len(det) == 8
    assert abs(m["defect_det"] - np.mean(det)) < 1e-9
    small = np.array(areas) < COMP_EDGES[0]
    assert small.sum() == 4
    assert abs(m["defect_small_cov"] - np.mean(np.array(cov)[small])) < 1e-9
    assert set(k for k in m if k.startswith("cat_")) == {"cat_catA_ap", "cat_catB_ap", "cat_macro_ap"}
    assert abs(m["cat_macro_ap"] - (m["cat_catA_ap"] + m["cat_catB_ap"]) / 2) < 1e-12


def test_unreachable_target_recall_is_nan_not_threshold_zero():
    # 30% of defect pixels scored exactly 0 -> recall 0.9 only at threshold 0 (all pixels positive)
    ev = PixelEvaluator(size_edges=(0.01, 0.1), comp_edges=(0.01, 0.1))
    gt = torch.zeros(20, 20)
    gt[:10, :10] = 1
    prob = torch.zeros(20, 20)
    prob[:7, :10] = 0.8
    ev.update(prob, gt, "a", 1)
    ev.update(torch.zeros(20, 20), torch.zeros(20, 20), "b", 0)
    st = ev.state()
    k, k90 = select_thresholds(st)
    assert k90 is None
    m = metrics_at(st, k, k90)
    assert np.isnan(m["fp_at_val_r90"]) and abs(m["max_recall"] - 0.7) < 1e-6
