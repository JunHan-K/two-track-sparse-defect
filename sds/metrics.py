"""Histogram-based segmentation metrics.

All metrics are computed at the ORIGINAL image resolution (padding removed, probability
map upsampled) against the original binary mask.

Threshold-free (2^16 score bins, resolution 1.5e-5, accumulated during evaluation):
  pixel_ap               AP over all pixels of the split
  ratio_{g}_ap           AP of the defect pixels of images whose foreground RATIO falls in
                         group g (small/medium/large); negatives = all background pixels
  defect_{g}_ap          AP of the pixels of individual defect COMPONENTS whose area falls
                         in group g; negatives = all background pixels
  cat_{c}_ap, cat_macro_ap  per-category AP and their unweighted mean (e.g. VisA PCB1-4)
  image_ap, image_auroc  image score = max pixel probability; ties are handled as one
                         threshold (independent of input order)

Threshold-based (validation-selected threshold, 1/THR_BINS grid, from per-image histograms):
  precision, recall, f1, foreground_iou, normal_fp_area (mean FP area on defect-free images),
  fp_at_val_r90 + recall_at_val_r90 (FP area / recall on THIS split at the threshold that
  gave recall >= 0.9 on validation — recall on the test split is NOT fixed to 0.9; NaN when
  no threshold > 0 reaches 0.9 on validation, see max_recall),
  ratio_{g}_recall       mean per-image pixel recall in image-ratio group g
  defect_{g}_det         fraction of defect components whose max probability >= threshold
                         ("touched at least once" — a lenient detection criterion)
  defect_{g}_cov         mean per-component pixel coverage (fraction of its pixels >= threshold)
Size-group edges are passed in (Q33/Q66 of the training split) so all methods share them.
"""
import numpy as np
import torch
from scipy import ndimage

AP_BINS = 262144
# AP histograms are binned in LOGIT space. Uniform bins in probability space merged the saturated
# scores near 1 (and near 0) of confident models: with 65536 bins the top bin alone held ~19% of the small-defect
# pixels of the native-resolution modes, and the small-defect AP still moved by 3 points per 4x finer bins.
# Logits of float32 probabilities lie in [LOGIT_MIN, LOGIT_MAX] (sigmoid saturates to 1.0 above ~16.6); 262144 bins
# over this range resolve 4e-4 in logit, i.e. the AP equals the AP on the raw float32 scores up to ties.
LOGIT_MIN, LOGIT_MAX = -104.0, 17.0


def ap_bin(prob):
    """Monotone map of probabilities (torch tensor) to AP histogram bins (logit space)."""
    p = prob.double()
    z = torch.log(p) - torch.log1p(-p)  # +inf at p == 1, -inf at p == 0: clamped to the end bins
    z = z.clamp(LOGIT_MIN, LOGIT_MAX)
    return ((z - LOGIT_MIN) / (LOGIT_MAX - LOGIT_MIN) * AP_BINS).long().clamp_(0, AP_BINS - 1)
THR_BINS = 4096
GROUPS = ("small", "medium", "large")


def thr_value(k):
    """Threshold value of grid index k: pixel predicted positive iff prob >= k / THR_BINS."""
    return k / THR_BINS


def _group(x, edges):
    q33, q66 = edges
    return "small" if x < q33 else ("medium" if x < q66 else "large")


def step_ap(pos, neg):
    """Step-wise AP from score histograms (index = ascending score bin); ties share a bin."""
    pos = np.asarray(pos, dtype=np.float64)[::-1]
    neg = np.asarray(neg, dtype=np.float64)[::-1]
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    if tp[-1] == 0:
        return float("nan")
    keep = (pos + neg) > 0
    tp, fp = tp[keep], fp[keep]
    prec, rec = tp / np.maximum(tp + fp, 1), tp / tp[-1]
    return float(np.sum((rec - np.concatenate([[0.0], rec[:-1]])) * prec))


def ap_from_scores(y, s):
    """Exact step-wise AP for (label, score) pairs; equal scores form one threshold."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    uniq, inv = np.unique(s, return_inverse=True)  # ascending
    pos = np.bincount(inv, weights=y, minlength=len(uniq))
    neg = np.bincount(inv, weights=1 - y, minlength=len(uniq))
    return step_ap(pos, neg)


def auroc_from_scores(y, s):
    from scipy.stats import rankdata

    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)  # average ranks for ties
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tolerance_keep(g, tol):
    """Boundary tolerance (crack-segmentation practice; same for every method). Returns the mask of
    BACKGROUND pixels that still count as negatives: background within `tol` px of a GT defect is ignored
    (a prediction there is not a false positive). Every GT pixel stays a positive; its score is the max
    prediction within `tol` px (recall side, see PixelEvaluator). The first version also
    dropped an inner GT band, which removed well-detected pixels of mid-size defects and could LOWER AP."""
    st = np.ones((3, 3), bool)
    outer = ndimage.binary_dilation(g, st, iterations=tol) & ~g
    return ~outer


def pd_fa_counts(pred, g, comp_ratio, comp_edges, dist=3.0):
    """IRSTD target-level protocol (DNANet / MSHNet utils/metric.py PD_FA), at a fixed threshold:
    predicted 8-connected components are matched greedily to GT components whose centroid lies
    < `dist` px away (each prediction used once). Pd = matched / GT components; Fa = pixel area of
    unmatched predicted components / all pixels. Also returns matched/total for SMALL GT components."""
    st = np.ones((3, 3), bool)
    pl, pn = ndimage.label(pred, structure=st)
    pc = ndimage.center_of_mass(np.ones_like(pl), pl, np.arange(1, pn + 1)) if pn else []
    pa = ndimage.sum(np.ones_like(pl), pl, np.arange(1, pn + 1)) if pn else []
    gl, gn = ndimage.label(g, structure=st)
    gc = ndimage.center_of_mass(np.ones_like(gl), gl, np.arange(1, gn + 1)) if gn else []
    used = np.zeros(pn, bool)
    matched, small_tot, small_hit = 0, 0, 0
    for i in range(gn):
        small = comp_edges is not None and comp_ratio[i] < comp_edges[0]
        small_tot += int(small)
        for m in range(pn):
            if not used[m] and np.hypot(pc[m][0] - gc[i][0], pc[m][1] - gc[i][1]) < dist:
                used[m] = True
                matched += 1
                small_hit += int(small)
                break
    fa_area = float(np.sum(np.asarray(pa)[~used])) if pn else 0.0
    return gn, matched, fa_area, small_tot, small_hit


class PixelEvaluator:
    def __init__(self, keep_per_image=True, size_edges=None, comp_edges=None, category_fn=None, tol=0):
        self.tol = tol
        self.keep = keep_per_image
        self.size_edges, self.comp_edges, self.category_fn = size_edges, comp_edges, category_fn
        self.ap_pos = self.ap_neg = None
        self.ratio_pos = {g: np.zeros(AP_BINS, np.int64) for g in GROUPS}
        self.defect_pos = {g: np.zeros(AP_BINS, np.int64) for g in GROUPS}
        self.cat_pos, self.cat_neg = {}, {}
        self.img_pos, self.img_neg = [], []
        self.img_label, self.img_fg_ratio, self.img_ids, self.img_max = [], [], [], []
        self.comp_img, self.comp_area_ratio, self.comp_max, self.comp_hist = [], [], [], []
        self.pdfa = np.zeros(6)  # targets, matched, fa_pixels, pixels, small targets, small matched (thr 0.5)

    @torch.no_grad()
    def update(self, prob, gt, image_id=None, label=None):
        """prob: float tensor (h, w) in [0, 1]; gt: {0,1} tensor (h, w)."""
        assert prob.dim() == 2 and prob.shape == gt.shape
        shape = tuple(prob.shape)
        prob = prob.float().clamp(0, 1)
        gt2 = gt > 0
        keep_m = None
        if self.tol and bool(gt2.any()):
            g_np = gt2.cpu().numpy()
            keep_m = torch.from_numpy(tolerance_keep(g_np, self.tol).ravel()).to(prob.device)
            # recall side: a GT pixel is hit if the prediction is high anywhere within tol px of it
            prob_det = torch.from_numpy(ndimage.maximum_filter(prob.cpu().numpy(), size=2 * self.tol + 1)).flatten().to(prob.device)
        prob = prob.flatten()
        gt = gt2.flatten()
        full_prob = prob
        pos_prob = prob_det if keep_m is not None else prob  # positives: tolerant score
        if keep_m is not None:  # negatives: background outside the tolerance band
            prob, gt = prob[keep_m], gt[keep_m]
        qa = ap_bin(prob)
        qa_pos = ap_bin(pos_prob)
        pa = torch.bincount(qa_pos[gt2.flatten()], minlength=AP_BINS)
        na = torch.bincount(qa[~gt], minlength=AP_BINS)
        if self.ap_pos is None:
            self.ap_pos, self.ap_neg = pa.clone(), na.clone()
        else:
            self.ap_pos += pa
            self.ap_neg += na
        if not self.keep:
            return
        n_fg = int(gt2.sum())  # full GT (tolerance only removes pixels from the counts below)
        pa_np = pa.cpu().numpy()
        if self.category_fn is not None:
            c = self.category_fn(image_id)
            if c not in self.cat_pos:
                self.cat_pos[c], self.cat_neg[c] = np.zeros(AP_BINS, np.int64), np.zeros(AP_BINS, np.int64)
            self.cat_pos[c] += pa_np
            self.cat_neg[c] += na.cpu().numpy()
        if n_fg and self.size_edges is not None:
            self.ratio_pos[_group(n_fg / gt2.numel(), self.size_edges)] += pa_np

        qt = (prob * THR_BINS).long().clamp_(max=THR_BINS - 1)
        qt_pos = (pos_prob * THR_BINS).long().clamp_(max=THR_BINS - 1)
        self.img_pos.append(torch.bincount(qt_pos[gt2.flatten()], minlength=THR_BINS).cpu().numpy().astype(np.int32))
        self.img_neg.append(torch.bincount(qt[~gt], minlength=THR_BINS).cpu().numpy().astype(np.int32))
        self.img_label.append(int(label) if label is not None else int(n_fg > 0))
        self.img_fg_ratio.append(n_fg / gt2.numel())
        self.img_ids.append(image_id)
        self.img_max.append(float(prob.max()))
        g_full = gt2.cpu().numpy()
        if g_full.any():
            lab0, n0 = ndimage.label(g_full, structure=np.ones((3, 3)))
            cr = np.asarray(ndimage.sum(g_full, lab0, np.arange(1, n0 + 1))) / g_full.size
        else:
            cr = np.zeros(0)
        t, mt, fa, stt, sh = pd_fa_counts(full_prob.view(shape).cpu().numpy() > 0.5, g_full, cr, self.comp_edges)
        self.pdfa += np.array([t, mt, fa, g_full.size, stt, sh], dtype=float)
        if n_fg:
            g = gt2.cpu().numpy()  # full GT (components are defined on the unmasked mask)
            lab, n = ndimage.label(g, structure=np.ones((3, 3)))
            lab = lab.ravel()
            fq = pos_prob  # tolerant positive scores (= full_prob when tol == 0)
            qa_np = ap_bin(fq).cpu().numpy()
            qt_np = (fq * THR_BINS).long().clamp_(max=THR_BINS - 1).cpu().numpy()
            pm = fq.cpu().numpy()
            km = np.ones(lab.shape, bool)  # every GT pixel stays a positive
            k_img = len(self.img_ids) - 1
            for j in range(1, n + 1):
                sel = lab == j
                ratio = sel.sum() / g.size  # size group from the FULL component area (unchanged by tol)
                ev = sel & km
                self.comp_img.append(k_img)
                self.comp_area_ratio.append(float(ratio))
                self.comp_max.append(float(pm[sel].max()))
                self.comp_hist.append(np.bincount(qt_np[ev], minlength=THR_BINS).astype(np.int32))
                if self.comp_edges is not None:
                    self.defect_pos[_group(ratio, self.comp_edges)] += np.bincount(qa_np[ev], minlength=AP_BINS)

    def pixel_ap(self):
        return step_ap(self.ap_pos.cpu().numpy(), self.ap_neg.cpu().numpy())

    def state(self):
        st = {"ap_pos": self.ap_pos.cpu().numpy(), "ap_neg": self.ap_neg.cpu().numpy(), "pdfa": self.pdfa.copy()}
        if not self.keep:
            return st
        st.update({
            "img_pos": np.stack(self.img_pos), "img_neg": np.stack(self.img_neg),
            "img_label": np.array(self.img_label), "img_fg_ratio": np.array(self.img_fg_ratio),
            "img_ids": np.array(self.img_ids, dtype=object), "img_max": np.array(self.img_max),
            "comp_img": np.array(self.comp_img, dtype=np.int64),
            "comp_area_ratio": np.array(self.comp_area_ratio), "comp_max": np.array(self.comp_max),
            "comp_hist": np.stack(self.comp_hist) if self.comp_hist else np.zeros((0, THR_BINS), np.int32),
        })
        for g in GROUPS:
            st[f"ratio_pos_{g}"] = self.ratio_pos[g]
            st[f"defect_pos_{g}"] = self.defect_pos[g]
        for c in self.cat_pos:
            st[f"cat_pos_{c}"], st[f"cat_neg_{c}"] = self.cat_pos[c], self.cat_neg[c]
        return st


# ----------------------------------------------------------------------
# Metrics from a saved state
# ----------------------------------------------------------------------
def _tail(h):
    """tail[k] = sum(h[k:]) along the last axis -> #pixels with score >= thr_value(k)."""
    return np.cumsum(h[..., ::-1], axis=-1)[..., ::-1]


def aupro(state, fpr_limit=0.3, comp_mask=None):
    """Area under the per-region-overlap curve up to FPR = fpr_limit, normalised to [0, 1]
    (Bergmann et al., IJCV 2021, the MVTec AD segmentation metric). PRO(t) = mean over GT connected
    components of the fraction of their pixels with score >= t; FPR(t) = background pixels >= t / all
    background pixels. Every defect counts equally, regardless of its size. comp_mask selects components
    (e.g. the small group) for a size-restricted AUPRO."""
    ch = state["comp_hist"].astype(np.float64)
    if comp_mask is not None:
        ch = ch[comp_mask]
    if len(ch) == 0:
        return float("nan")
    pro = (_tail(ch) / np.maximum(ch.sum(1, keepdims=True), 1)).mean(0)       # per threshold index k
    neg = _tail(state["img_neg"].sum(0).astype(np.float64))
    fpr = neg / max(neg[0], 1)
    order = np.argsort(fpr)                                                    # fpr decreases with k
    f, p_ = fpr[order], pro[order]
    keep = f <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    f, p_ = f[keep], p_[keep]
    if f[-1] < fpr_limit:  # extend to the limit with the last PRO value
        f, p_ = np.append(f, fpr_limit), np.append(p_, p_[-1])
    return float(np.trapz(p_, f) / fpr_limit)


def curves(state):
    pos = _tail(state["img_pos"].sum(0).astype(np.float64))
    neg = _tail(state["img_neg"].sum(0).astype(np.float64))
    P = pos[0]
    tp, fp, fn = pos, neg, P - pos
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    iou = tp / np.maximum(tp + fp + fn, 1)
    # mIoU over {background, defect} as in mmsegmentation / Defect Spectrum (dataset-level counts):
    # background IoU = TN / (TN + FP + FN) with TN = N - FP
    N = neg[0]
    iou_bg = (N - fp) / np.maximum(N + fn, 1)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "miou": (iou + iou_bg) / 2}


def select_thresholds(val_state, criterion="f1", target_recall=0.9):
    """Choose thresholds on VALIDATION only. Returns grid indices (see thr_value).

    k_r is None when no threshold > 0 reaches target_recall: threshold 0 marks every pixel
    positive (recall 1 trivially), which would report FP area = 1.0 as if it were a result.
    """
    c = curves(val_state)
    k_best = int(np.argmax(c[criterion]))
    ok = np.where(c["recall"][1:] >= target_recall)[0] + 1
    k_r = int(ok.max()) if len(ok) else None
    return k_best, k_r


def threshold_free(state):
    out = {"pixel_ap": step_ap(state["ap_pos"], state["ap_neg"])}
    if "img_max" not in state:
        return out
    neg = state["ap_neg"]
    for g in GROUPS:
        out[f"ratio_{g}_ap"] = step_ap(state[f"ratio_pos_{g}"], neg)
        out[f"defect_{g}_ap"] = step_ap(state[f"defect_pos_{g}"], neg)
    cats = sorted(k[len("cat_pos_"):] for k in state if k.startswith("cat_pos_"))
    if cats:
        aps = [step_ap(state[f"cat_pos_{c}"], state[f"cat_neg_{c}"]) for c in cats]
        out.update({f"cat_{c}_ap": a for c, a in zip(cats, aps)})
        out["cat_macro_ap"] = float(np.nanmean(aps))
    y, s = state["img_label"], state["img_max"]
    out["image_ap"] = ap_from_scores(y, s)
    out["image_auroc"] = auroc_from_scores(y, s)
    return out


def metrics_at(state, k, k_r90=None, size_edges=None, comp_edges=None):
    """Threshold-based metrics at grid index k (selected on validation) + threshold-free ones."""
    c = curves(state)
    pos, neg = state["img_pos"].astype(np.int64), state["img_neg"].astype(np.int64)
    n_pix = pos.sum(1) + neg.sum(1)
    normal = state["img_label"] == 0

    def normal_fp(kk):
        return float((_tail(neg[normal])[:, kk] / n_pix[normal]).mean()) if normal.any() else float("nan")

    out = threshold_free(state)
    out.update({
        "threshold": thr_value(k), "precision": float(c["precision"][k]), "recall": float(c["recall"][k]),
        "f1": float(c["f1"][k]), "foreground_iou": float(c["iou"][k]), "miou": float(c["miou"][k]),
        "normal_fp_area": normal_fp(k),
    })
    out["max_recall"] = float(c["recall"][1])  # highest recall at any threshold > 0
    if k_r90 is not None:
        out["threshold_val_r90"] = thr_value(k_r90)
        out["fp_at_val_r90"] = normal_fp(k_r90)
        out["recall_at_val_r90"] = float(c["recall"][k_r90])
    else:  # target recall unreachable on validation: report as missing, never as FP = 1
        out["threshold_val_r90"] = out["fp_at_val_r90"] = out["recall_at_val_r90"] = float("nan")
    if size_edges is not None:
        r = state["img_fg_ratio"]
        has_fg = pos.sum(1) > 0
        img_rec = np.where(has_fg, _tail(pos)[:, k] / np.maximum(pos.sum(1), 1), np.nan)
        for g in GROUPS:
            m = has_fg & np.array([_group(x, size_edges) == g for x in r])
            out[f"ratio_{g}_recall"] = float(np.nanmean(img_rec[m])) if m.any() else float("nan")
            out[f"n_ratio_{g}"] = int(m.sum())
    if "comp_hist" in state and len(state["comp_hist"]):
        out["aupro"] = aupro(state)
        if comp_edges is not None:
            sm = np.array([_group(x, comp_edges) == "small" for x in state["comp_area_ratio"]])
            out["aupro_small"] = aupro(state, comp_mask=sm) if sm.any() else float("nan")
    if comp_edges is not None and len(state["comp_max"]):
        a = state["comp_area_ratio"]
        hit = state["comp_max"] >= thr_value(k)
        ch = state["comp_hist"].astype(np.int64)
        cov = _tail(ch)[:, k] / np.maximum(ch.sum(1), 1)
        out["defect_det"] = float(hit.mean())
        for g in GROUPS:
            m = np.array([_group(x, comp_edges) == g for x in a])
            out[f"defect_{g}_det"] = float(hit[m].mean()) if m.any() else float("nan")
            out[f"defect_{g}_cov"] = float(cov[m].mean()) if m.any() else float("nan")
            out[f"n_defect_{g}"] = int(m.sum())
    if "pdfa" in state:  # IRSTD protocol at threshold 0.5 (independent of the val-selected threshold)
        t, mt, fa, npx, stt, sh = state["pdfa"]
        out["irstd_pd"] = float(mt / t) if t else float("nan")
        out["irstd_fa"] = float(fa / npx) if npx else float("nan")
        out["irstd_small_pd"] = float(sh / stt) if stt else float("nan")
    return out


def ap_from_state(state):
    return step_ap(state["ap_pos"], state["ap_neg"])
