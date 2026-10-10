"""Fig. 1: overview of the method with real data (final method, 2026-10-05).

  python tools/figures/fig1_data.py --step train|replay|infer     # once: real crops / maps -> fig1_data/*.npz
  python tools/figures/fig1_framework.py --out outputs/paper/figures

(a) Training on two views: the same image enters as a downscaled whole view (letterbox 1024^2) and as native
    384^2 crops; ONE shared MiT-B0 segmenter (encoder stages s1-s4, all-MLP decoder, main head). Size-aware
    supervision: the main head's BCE weight w(a) (real map) and stage heads trained only during training, s1/s2 on
    small components only (real target), s3/s4 on all components.
(b) Sparse Defect Replay: the trained model on native tiles of the training images; its false positives (real boxes,
    red) set how many random background crops each image receives; these are replayed with small-defect crops (real
    crops) in a 20-epoch refinement.
(c) Two-track inference on a validation image: low-resolution track -> p_L (real), candidates and support M;
    native track on the candidate crops -> p_S; merge rule; output (real). Recall mode = all native tiles.
Numbers (crops / GFLOPs) are the validation means of the final model (tools/paper_tables.py).
The previous box diagram is kept as fig1_framework_v3_boxes.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sds.utils import resolve  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 6.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED = "#1f1f1f", "#6b6b6b"
LOW, NAT, SIZE, REP, GT = "#2f66b3", "#7a4fc4", "#e07b1a", "#c8323c", "#2ca25f"
ALL = "#8c510a"  # s3,4 targets (all defects); kept apart from the GT green
NETF = "#eceef1"
NETC = "#4d5563"  # the one shared network: neutral, identical everywhere (paths carry the view colour)
DATA = "outputs/figures/fig1_data"


def arrow(ax, p, q, c=INK, lw=0.9, rad=0.0, ls="-", hw=0.22, z=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length=0.42,head_width={hw}", mutation_scale=8, color=c,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def rbox(ax, x, y, w, h, fc="white", ec=INK, lw=0.7, ls="-", r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))


def img(ax, a, x, y, w, h=None, ec=None, lw=0.8, cmap=None, vmin=None, vmax=None, z=3, interp="lanczos"):
    h = h if h is not None else w * a.shape[0] / a.shape[1]
    ax.imshow(a, extent=(x, x + w, y, y + h), cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp, zorder=z,
              aspect="auto")
    if ec:
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=z + 1))
    return h


def contour(ax, m, x, y, w, h, c=GT, lw=0.6, z=5):
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    ax.contour(x + (xs + 0.5) * w / m.shape[1], y + h - (ys + 0.5) * h / m.shape[0], m.astype(float), levels=[0.5],
               colors=c, linewidths=lw, zorder=z)


def overlay(rgb, mask, color, alpha=0.75):
    out = rgb.astype(np.float32).copy()
    c = np.array(matplotlib.colors.to_rgb(color)) * 255
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    return out.astype(np.uint8)


def heat(p, gamma=1.0):
    return plt.get_cmap("inferno")(np.clip(p, 0, 1) ** gamma)[..., :3]


def net(ax, x, y, s=1.0, c=NETC, label=True, stages=True):
    """Shared segmenter icon: 4 encoder stages (shrinking), all-MLP decoder, main head."""
    hs = [0.62, 0.50, 0.38, 0.28]
    xx = x
    for i, h in enumerate(hs):
        h *= s
        ax.add_patch(Polygon([(xx, y - h / 2), (xx + 0.11 * s, y - h / 2 + 0.05 * s), (xx + 0.11 * s, y + h / 2 + 0.05 * s),
                              (xx, y + h / 2)], closed=True, fc=NETF, ec=c, lw=0.7, zorder=4))
        if stages:
            ax.text(xx + 0.055 * s, y - h / 2 - 0.035, f"$s_{i + 1}$", ha="center", va="top", fontsize=7.4, color=c)
        xx += 0.17 * s
    ax.add_patch(Polygon([(xx, y - 0.22 * s), (xx + 0.30 * s, y - 0.32 * s), (xx + 0.30 * s, y + 0.32 * s), (xx, y + 0.22 * s)],
                         closed=True, fc=NETF, ec=c, lw=0.7, zorder=4))
    if label:
        ax.text(xx + 0.15 * s, y, "MLP\ndec.", ha="center", va="center", fontsize=5.2, color=c, zorder=5)
    return xx + 0.30 * s


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="#fbfbfb", ec="#c9c9c9",
                                lw=0.6, zorder=0))
    ax.text(x + 0.07, y + h - 0.06, title, ha="left", va="top", fontsize=7.6, weight="bold", color=INK)


def blend(rgb, p, gamma=0.6, amax=0.9):
    """Probability map over a dimmed grey image (inferno), so weak responses stay visible."""
    g = rgb.astype(np.float32).mean(2, keepdims=True) / 255.0 * 0.75
    a = np.clip(p * 2.0, 0, amax)[..., None]
    return np.clip(g * (1 - a) + heat(p, gamma) * a, 0, 1)


def wblend(rgb, w, m):
    g = np.repeat(rgb.astype(np.float32).mean(2, keepdims=True) / 255.0 * 0.6, 3, 2)
    c = plt.get_cmap("magma")((np.clip(w, 1, 5) - 1) / 4 * 0.62 + 0.38)[..., :3]
    return np.where(m[..., None], c, g)


def clipped_box(ax, x0, y0, w, h, bx, by, bw, bh, **kw):
    """Rectangle (x0, y0, w, h) clipped to the image area (bx, by, bw, bh)."""
    xa, ya = max(x0, bx), max(y0, by)
    xb, yb = min(x0 + w, bx + bw), min(y0 + h, by + bh)
    if xb > xa and yb > ya:
        ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fc="none", **kw))


def seg(ax, pts, c=INK, lw=0.9, head=True, z=8):
    """Straight, right-angled connector through pts; arrowhead on the last segment."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=c, lw=lw, solid_capstyle="butt", zorder=z)
    if head:
        arrow(ax, pts[-2], pts[-1], c=c, lw=lw, z=z)
    else:
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=c, lw=lw, zorder=z)


def ring(ax, cx, cy, r, c="white", lw=0.9, z=9):
    t = np.linspace(0, 2 * np.pi, 80)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=c, lw=lw, zorder=z)


def centre(m):
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean()


def draw(out_dir):
    T = np.load(resolve(DATA) / "train.npz")
    R = np.load(resolve(DATA) / "replay.npz")
    I = np.load(resolve(DATA) / "infer.npz")
    plt.rcParams.update({"font.size": 7.2})
    LAB, SMALL, TITLE = 7.9, 7.4, 8.8  # printed at 0.95: >= 7 pt
    W, H, Y0 = 7.16, 3.72, 0.34  # inches; [Y0, H] shown; (c) is drawn compact and moved up by DC (below)
    DC = 0.14
    fig = plt.figure(figsize=(W, H - Y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")

    # ======================================================================= (a) training on two views
    ax_, ay_, aw_, ah_ = 0.03, 1.92, 4.38, 1.77
    panel(ax, ax_, ay_, aw_, ah_, "(a) Twin-View SparSight Training")
    plt.setp(ax.texts[-1], fontsize=TITLE)
    wx, wy, ww = 0.12, 2.66, 0.88
    wh = img(ax, overlay(T["whole"], T["whole_g"] > 0, GT, 0.9), wx, wy, ww, ec=LOW, lw=1.0)
    ax.text(wx, wy + wh + 0.04, "whole image ($1024^2$)", fontsize=LAB, color=LOW, va="bottom")
    nb = T["native_box"]
    sx = ww / nb[3]
    bx, by, bs = wx + nb[1] * sx, wy + wh - (nb[0] + 384) * sx, 384 * sx
    ax.add_patch(Rectangle((bx, by), bs, bs, fc="none", ec=NAT, lw=0.9, zorder=6))
    cs_n, cy_n = 0.50, 1.99
    cx_n = bx + bs / 2 - cs_n / 2  # the crop sits right under its box: one straight arrow
    img(ax, T["native"], cx_n, cy_n, cs_n, cs_n, ec=NAT, lw=1.0)
    contour(ax, T["native_g"], cx_n, cy_n, cs_n, cs_n, c=GT, lw=0.6)
    seg(ax, [(bx + bs / 2, by), (bx + bs / 2, cy_n + cs_n + 0.01)], c=NAT, lw=0.8)
    ax.text(bx + bs / 2 + 0.05, (wy + cy_n + cs_n) / 2, "every 3rd batch", fontsize=SMALL, color=NAT, va="center")
    ax.text(cx_n + cs_n + 0.05, cy_n + 0.02, "native\n$384^2$ crop", fontsize=LAB, color=NAT, va="bottom",
            linespacing=1.0)
    # one shared network
    nx, ny = 1.55, 2.98
    xe = net(ax, nx, ny, 0.95, label=False)
    ax.text((nx + xe) / 2, ny + 0.38, "one MiT-B0 segmenter for both views", ha="center", va="bottom", fontsize=LAB, color=NETC)
    seg(ax, [(wx + ww, wy + wh / 2), (nx - 0.03, wy + wh / 2)], c=LOW)
    seg(ax, [(cx_n + cs_n, cy_n + cs_n * 0.8), (1.43, cy_n + cs_n * 0.8), (1.43, ny - 0.12), (nx - 0.03, ny - 0.12)],
        c=NAT)
    # main head (logit z) -> gated logit fusion with the stage-head logits z1-z4 -> size-aware loss on the fused logit
    hb_x, hb_w = xe + 0.08, 0.30
    rbox(ax, hb_x, ny - 0.11, hb_w, 0.22, fc=NETF, ec=NETC)
    ax.text(hb_x + hb_w / 2, ny, "head", ha="center", va="center", fontsize=SMALL, color=NETC, zorder=5)
    seg(ax, [(xe, ny), (hb_x - 0.01, ny)], c=NETC)
    fb_x, fb_w = hb_x + hb_w + 0.13, 0.56
    rbox(ax, fb_x, ny - 0.15, fb_w, 0.30, fc="white", ec=NETC, lw=0.9)
    ax.text(fb_x + fb_w / 2, ny, "gated logit\nfusion", ha="center", va="center", fontsize=SMALL, color=NETC, zorder=5,
            linespacing=0.95)
    seg(ax, [(hb_x + hb_w, ny), (fb_x - 0.01, ny)], c=NETC)
    ax.text(hb_x + hb_w + 0.065, ny + 0.03, "$z$", ha="center", va="bottom", fontsize=SMALL, color=NETC)
    tx, ty, ts = 3.70, 2.44, 0.66
    img(ax, wblend(T["native"], T["native_w"], T["native_g"]), tx, ty, ts, ts, ec=SIZE, lw=1.0)
    seg(ax, [(fb_x + fb_w, ny), (tx - 0.02, ny)], c=SIZE)
    ax.text(tx + ts, ty + ts + 0.04, "BCE$\\cdot w(a)$ + Dice", ha="right", va="bottom", fontsize=SMALL, color=SIZE)
    ax.text(tx + ts, ty - 0.04, "small defects: $w\\leq\\!5\\times$", ha="right", va="top", fontsize=SMALL,
            color=SIZE)
    # stage heads, training only
    # stage heads (training only), right under the encoder stages: s1, s2 -> small-defect targets, s3, s4 -> all
    sy, st = 1.98, 0.34
    sxs = [nx + 0.17 * 0.95 * i + 0.055 * 0.95 for i in range(4)]  # stage centres (as drawn by net())
    tb = 0.27  # thumbnail side; centred under (s1, s2) and (s3, s4); straight arrows converge on each
    cA, cB = (sxs[0] + sxs[1]) / 2, (sxs[2] + sxs[3]) / 2
    # each thumbnail shows only its own target mask, solid in the frame colour, over a faded crop
    faded = np.repeat((255 - (255 - T["native"].astype(np.float32).mean(2, keepdims=True)) * 0.3), 3, 2)
    for cc, m, c in ((cA, T["native_small"], SIZE), (cB, T["native_g"], ALL)):
        img(ax, overlay(faded, ndimage.binary_dilation(m > 0, iterations=3), c, 1.0), cc - tb / 2, sy + 0.02, tb, tb,
            ec=c, lw=0.8)
    for i, x in enumerate(sxs):
        cc = cA if i < 2 else cB
        seg(ax, [(x, 2.57), (cc + (x - cc) * 0.2, sy + 0.02 + tb + 0.01)], c=SIZE if i < 2 else ALL, lw=0.7)
    lx = cB + tb / 2 + 0.06
    ax.text(lx, sy + st, "stage-head targets", fontsize=SMALL, color=MUTED, va="top")
    # the stage-head logits z1-z4 enter the gated fusion (used at inference too)
    fx0, fy0, hx_ = sxs[3] + 0.10, ny - 0.40, fb_x + fb_w / 2
    seg(ax, [(fx0, fy0), (hx_, fy0), (hx_, ny - 0.16)], c=NETC, lw=0.8)
    ax.text(fx0 + 0.04, fy0 + 0.02, "$z_1$\u2013$z_4$", fontsize=SMALL, color=NETC, va="bottom", ha="left")
    ax.text(hx_ - 0.05, fy0 + 0.15, "$z_{1,2}$: evidence\n$z_{3,4}$: gate", fontsize=SMALL, color=NETC, va="center",
            ha="right", linespacing=1.0)
    ax.text(lx, sy + 0.17, "$s_{1,2}$: small defects", fontsize=SMALL, color=SIZE, va="center")
    ax.text(lx, sy + 0.01, "$s_{3,4}$: all defects", fontsize=SMALL, color=ALL, va="bottom")

    # ======================================================================= (b) Sparse Defect Replay
    bx_, by_, bw_, bh_ = 4.48, 1.92, 2.65, 1.77
    panel(ax, bx_, by_, bw_, bh_, "(b) Sparse Defect Replay")
    plt.setp(ax.texts[-1], fontsize=TITLE)
    rw = R["whole"]
    rx, ry, rwid = 4.57, 2.52, 1.00
    rh = img(ax, rw, rx, ry, rwid)
    s_ = rwid / rw.shape[1]
    k_fp = len(R["fp_xy"])
    # orange: small defects (centres of the small-defect crops); green: other defects (GT, as in (a), (c));
    # red: native false positives (their count k sets the number of random background crops)
    lab_, n_ = ndimage.label(R["whole_g"] > 0)
    small_ids = {lab_[min(int(round(y)), lab_.shape[0] - 1), min(int(round(x)), lab_.shape[1] - 1)]
                 for x, y in R["pos_xy"]} - {0}
    for j, sl in enumerate(ndimage.find_objects(lab_)):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        w_, h_ = max((x1 - x0) * s_, 0.05), max((y1 - y0) * s_, 0.05)
        cxp, cyp = rx + (x0 + x1) / 2 * s_, ry + rh - (y0 + y1) / 2 * s_
        ax.add_patch(Rectangle((cxp - w_ / 2 - 0.01, cyp - h_ / 2 - 0.01), w_ + 0.02, h_ + 0.02, fc="none",
                               ec=SIZE if j + 1 in small_ids else GT, lw=0.8, zorder=6))
    for fx, fy, _ in R["fp_xy"]:
        ax.add_patch(Rectangle((rx + fx * s_ - 0.03, ry + rh - fy * s_ - 0.03), 0.06, 0.06, fc="none", ec=REP, lw=0.7,
                               zorder=6))
    ax.text(rx, ry + rh + 0.04, "false positives", fontsize=LAB, color=REP, va="bottom")
    ax.text(rx + rwid / 2, ry - 0.03, "count FPs per image", fontsize=SMALL, color=REP, ha="center", va="top")
    # the two crop sets, each framed as one group
    cs, gp, gx0 = 0.30, 0.03, 5.86
    bgy, psy = 3.00, 2.50
    for i, c in enumerate(R["bg_crops"]):
        img(ax, c, gx0 + i * (cs + gp), bgy, cs, cs, ec=REP, lw=0.7)
    for i, c in enumerate(R["pos_crops"]):
        img(ax, c, gx0 + i * (cs + gp), psy, cs, cs, ec=SIZE, lw=0.7)
    pad = 0.035
    bg_x1 = gx0 + 3 * cs + 2 * gp + pad
    ps_x1 = gx0 + 2 * cs + gp + pad
    rbox(ax, gx0 - pad, bgy - pad, bg_x1 - gx0 + pad, cs + 2 * pad, fc="none", ec=REP, lw=0.9, r=0.03, z=5)
    rbox(ax, gx0 - pad, psy - pad, ps_x1 - gx0 + pad, cs + 2 * pad, fc="none", ec=SIZE, lw=0.9, r=0.03, z=5)
    ax.text(bg_x1, bgy + cs + pad + 0.03, "random background crops", fontsize=SMALL, color=REP, va="bottom",
            ha="right")
    ax.text(ps_x1 + 0.05, psy + cs / 2, "small-\ndefect\ncrops", fontsize=SMALL, color=SIZE, va="center",
            linespacing=1.0)
    # image -> groups, groups -> replay batch (straight, right-angled)
    seg(ax, [(rx + rwid, bgy + cs / 2), (gx0 - pad - 0.01, bgy + cs / 2)], c=REP)
    seg(ax, [(rx + rwid, psy + cs / 2), (gx0 - pad - 0.01, psy + cs / 2)], c=SIZE)
    rbox(ax, 4.57, 1.98, 2.49, 0.36, fc="white", ec=INK, lw=0.8)
    ax.text(5.815, 2.25, "replay batch: $\\frac{1}{2}$ small-defect + $\\frac{1}{2}$ background crops", ha="center",
            va="center", fontsize=SMALL, color=INK)
    ax.text(5.815, 2.08, "confused images sampled more often, not at FP locations", ha="center", va="center", fontsize=SMALL,
            color=INK)
    seg(ax, [((gx0 - pad + ps_x1) / 2, psy - pad), ((gx0 - pad + ps_x1) / 2, 2.35)], c=SIZE)
    seg(ax, [(bg_x1, bgy + cs / 2), (7.03, bgy + cs / 2), (7.03, 2.35)], c=REP)

    # ======================================================================= (c) inference
    before_c = {id(a) for a in ax.patches + ax.texts + ax.lines + ax.images + ax.collections}
    panel(ax, 0.03, 0.23, 7.10, 1.62 - DC, "(c) Twin-Track SparSight Inference")
    plt.setp(ax.texts[-1], fontsize=TITLE)
    im_ = I["img"]
    iH, iW = im_.shape[:2]
    zs = 0.62
    zy0 = 0.76
    iw = min(zs * iW / iH, 1.15)  # input and global prediction keep the image aspect (height zs)
    ih = iw * iH / iW
    iy = zy0 + (zs - ih) / 2
    # centre the row in the panel: row width = images, nets (0.49 wide at scale 0.5), merge column and the gaps below
    nw = 4 * 0.17 * 0.5 + 0.30 * 0.5
    row_w = iw + 0.36 + nw + 0.25 + iw + 0.50 + zs + 0.24 + nw + 0.26 + (zs - 0.04) / 2 + 0.36 + 0.10 + 0.24 + zs
    ix = 0.03 + (7.10 - row_w) / 2
    img(ax, im_, ix, iy, iw, ih)
    contour(ax, I["g"] > 0, ix, iy, iw, ih, c=GT, lw=0.6)
    ax.text(ix, iy + ih + 0.04, "input image", fontsize=LAB, va="bottom")
    ly = iy + ih / 2
    gx0n = ix + iw + 0.36
    xe = net(ax, gx0n, ly, 0.5, NETC, label=False, stages=False)
    seg(ax, [(ix + iw, ly), (gx0n - 0.02, ly)], c=LOW)
    ax.text((gx0n + xe) / 2, iy - 0.05, "Global Sight (L)\nsame MiT-B0", fontsize=LAB, color=LOW, ha="center", va="top",
            linespacing=1.0)
    # every prediction is drawn as in Fig. 2: probability map (inferno), ground truth in green
    px, pw = xe + 0.25, iw
    # shown at ~0.6 in: block-max pooled for display so thin predicted defects stay visible (averaging interpolation
    # erases them); no GT outline here, it would cover them (the ground truth is outlined on the input image)
    k_ = 5  # block maximum 5x5 (900 px -> 180 px, about the printed resolution), drawn without interpolation
    Lb = I["L"][: I["L"].shape[0] // k_ * k_, : I["L"].shape[1] // k_ * k_]
    Lb = Lb.reshape(Lb.shape[0] // k_, k_, Lb.shape[1] // k_, k_).max((1, 3))
    imb = im_[: Lb.shape[0] * k_ : k_, : Lb.shape[1] * k_ : k_]  # the same image, dimmed, under p_L (context)
    img(ax, blend(imb, Lb), px, iy, pw, ih, ec=LOW, lw=0.8, interp="nearest")
    contour(ax, I["L"] > 0.5, px, iy, pw, ih, c="#ffd23f", lw=0.9)  # predicted regions, outlined like the GT
    seg(ax, [(xe, ly), (px - 0.02, ly)], c=LOW)
    ax.text(px + pw / 2, iy - 0.04, "thin: candidates\nbold: re-inspected", fontsize=SMALL,
            color=MUTED, ha="center", va="top", linespacing=1.0)
    b = 384 * 900 / I["full_hw"][1] * pw / iW
    for cx_, cy_, _ in I["cand"]:  # all candidates; thin, so the predicted outlines stay visible
        clipped_box(ax, px + cx_ * pw / iW - b / 2, iy + ih - cy_ * pw / iW - b / 2, b, b, px, iy, pw, ih, ec=NAT,
                    lw=0.6, zorder=7)
    # native track on the candidate holding the example defect
    zb = I["zoom_box"]
    zx0 = px + pw + 0.50
    zi = I["zoom_img"]
    zg = I["zoom_g"] > 0
    gx_, gy_ = centre(I["zoom_g"])

    Z = 128  # crop panels are zoomed to a Z x Z window of the 384 x 384 native crop around the example defect
    wy0, wx0 = (int(np.clip(round(v - Z / 2), 0, 384 - Z)) for v in (gy_, gx_))

    def crop_panel(a, x, ec, lab, col, y=zy0, sz=zs, below=False):
        img(ax, a[wy0:wy0 + Z, wx0:wx0 + Z], x, y, sz, sz, ec=ec, lw=1.0)
        contour(ax, zg[wy0:wy0 + Z, wx0:wx0 + Z], x, y, sz, sz, c=GT, lw=0.6 if sz > 0.4 else 0.3)
        if below:
            ax.text(x + sz / 2, y - 0.04, lab, ha="center", va="top", fontsize=LAB, color=col)
            return
        ax.text(x + sz / 2, y + sz + 0.04, lab, ha="center", va="bottom", fontsize=LAB, color=col)

    crop_panel(zi, zx0, NAT, "native crop", NAT)  # centred label, wording as in the paper
    cbx = px + (zb[0] + zb[2]) * pw / iW
    # the window shown in the zoomed panels, boxed on the input and the global prediction (bold), and joined to the
    # native-crop panel by two straight zoom lines
    sc_ = zb[2] / 384  # 900-px panel scale per original pixel
    wx_, wy_, ws_ = zb[0] + wx0 * sc_, zb[1] + wy0 * sc_, Z * sc_
    def wbox(x0_, w_):
        s_ = max(ws_ * w_ / iW, 0.07)  # at least 0.07 in so it is visible
        cx_, cy_ = x0_ + (wx_ + ws_ / 2) * w_ / iW, iy + ih - (wy_ + ws_ / 2) * w_ / iW
        ax.add_patch(Rectangle((cx_ - s_ / 2, cy_ - s_ / 2), s_, s_, fc="none", ec=NAT, lw=1.6, zorder=8))
        return cx_ + s_ / 2, cy_ - s_ / 2, cy_ + s_ / 2
    wbox(ix, iw)
    bx1, by0, by1 = wbox(px, pw)
    for yb_, yz_ in ((by1, zy0 + zs), (by0, zy0)):
        ax.plot([bx1, zx0], [yb_, yz_], color=NAT, lw=0.7, zorder=8)
    xe2 = net(ax, zx0 + zs + 0.24, zy0 + zs / 2, 0.5, NETC, label=False, stages=False)  # same network icon/colour
    seg(ax, [(zx0 + zs, zy0 + zs / 2), (zx0 + zs + 0.22, zy0 + zs / 2)], c=NAT)
    ax.text((zx0 + zs + 0.24 + xe2) / 2, iy - 0.05, "Native Sight (S)\nsame MiT-B0", fontsize=LAB, color=NAT, ha="center",
            va="top", linespacing=1.0)
    # the merge (Eq. 3) on the same window: global p_L (top, from the global prediction) and native p_S (bottom)
    cz = (zs - 0.04) / 2
    ox = xe2 + 0.26
    ctr = zy0 + zs / 2
    yt, yb = ctr + 0.02, ctr - 0.02 - cz
    crop_panel(heat(I["zoom_L"]), ox, LOW, "", LOW, y=yt, sz=cz)
    crop_panel(heat(I["zoom_S"]), ox, NAT, "native $p_S$", NAT, y=yb, sz=cz, below=True)
    seg(ax, [(xe2, ctr), (ox - 0.07, ctr), (ox - 0.07, yb + cz / 2), (ox - 0.02, yb + cz / 2)], c=NAT)
    yr = 1.70 - DC
    # p_L of the same window: from the top edge of the global prediction (right end) straight up; the crop's guide
    # lines start at a candidate in the lower half (fig1_data), so they never cross it
    rx_ = px + pw - 0.05
    seg(ax, [(rx_, iy + ih), (rx_, yr), (ox + cz / 2, yr), (ox + cz / 2, yt + cz + 0.01)], c=LOW)
    ax.text(px, iy + ih + 0.04, "global $p_L$", fontsize=LAB, color=LOW, va="bottom", ha="left")
    ax.text(ox + cz / 2 - 0.04, yr + 0.01, "global $p_L$, same window", ha="right", va="bottom", fontsize=SMALL,
            color=LOW)
    mx, my_, mr = ox + cz + 0.36, ctr, 0.125
    ring(ax, mx, my_, mr, c=INK, lw=0.8)
    ax.text(mx, my_, "max", ha="center", va="center", fontsize=SMALL, color=INK)
    ax.text(mx, my_ - mr - 0.03, "pixel-wise (4);\n$p_S$: $\\times1$ in $M$,\n$\\times0.5$ outside", ha="center", va="top",
            fontsize=SMALL, color=MUTED, linespacing=1.0)
    seg(ax, [(ox + cz, yt + cz / 2), (mx - mr * 0.75, my_ + mr * 0.66)], c=LOW)
    seg(ax, [(ox + cz, yb + cz / 2), (mx - mr * 0.75, my_ - mr * 0.66)], c=NAT)
    fx_ = mx + mr + 0.24
    seg(ax, [(mx + mr, my_), (fx_ - 0.02, my_)], c=INK)
    crop_panel(heat(I["zoom_P"]), fx_, INK, "final prediction $p$", INK)
    # operating points, compact: what each mode re-inspects on the same image
    my, mt = 0.26, 0.30
    th = mt * iH / iW
    t = 384 * 900 / I["full_hw"][1] * mt / iW
    # mode strip spans the same width as the row: precision at its left edge, recall ending at its right edge
    modes_x = ix
    img(ax, im_, modes_x, my, mt, th)
    for cx_, cy_, _ in I["cand"]:
        clipped_box(ax, modes_x + cx_ * mt / iW - t / 2, my + th - cy_ * mt / iW - t / 2, t, t, modes_x, my, mt, th,
                    ec=NAT, lw=0.7, zorder=7)
    ax.text(modes_x + mt + 0.06, my + th / 2, "precision mode: selective, candidate crops", fontsize=SMALL, color=NAT,
            va="center")
    rt = ax.text(ix + row_w, my + th / 2, "recall mode: dense, all native tiles", fontsize=SMALL, color=NAT,
                 va="center", ha="right")
    bb = rt.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    rx2 = bb.x0 - 0.06 - mt
    img(ax, im_, rx2, my, mt, th)
    k = 1
    while k * t < mt:
        ax.plot([rx2 + k * t] * 2, [my, my + th], color=NAT, lw=0.4, zorder=6)
        k += 1
    k = 1
    while k * t < th:
        ax.plot([rx2, rx2 + mt], [my + th - k * t] * 2, color=NAT, lw=0.4, zorder=6)
        k += 1

    # move panel (c) up by DC so it sits right under (a), (b) (patches carry their own patch transform)
    sh = matplotlib.transforms.Affine2D().translate(0, DC) + ax.transData
    for a in ax.patches + ax.texts + ax.lines + ax.images + ax.collections:
        if id(a) not in before_c:
            a.set_transform(sh)
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_framework.{ext}", dpi=300)
    print("saved", out / "fig1_framework.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/paper/figures")
    draw(ap.parse_args().out)
