"""Fig. 1: overview of the method with real data .

  python tools/figures/fig1_data.py --step train|replay|infer     # once: real crops / maps -> fig1_data/*.npz
  python tools/figures/fig1_framework.py --out outputs/paper/figures

(a) Training on two views: the same image enters as a downscaled whole view (letterbox 1024^2) and as native
    384^2 crops; ONE shared MiT-B0 segmenter (encoder stages s1-s4, all-MLP decoder, main head). Size-aware
    supervision: the main head's BCE weight w(a) (real map) and stage heads trained only during training, s1/s2 on
    small components only (real target), s3/s4 on all components.
(b) Confusion replay: the trained model on native tiles of the training images; its false positives (real mined
    boxes, red) are replayed with small-defect crops (real crops) in a 20-epoch refinement.
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
NETF = "#e3ecf8"
DATA = "outputs/figures/fig1_data"


def arrow(ax, p, q, c=INK, lw=0.9, rad=0.0, ls="-", hw=0.22, z=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length=0.42,head_width={hw}", mutation_scale=8, color=c,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def rbox(ax, x, y, w, h, fc="white", ec=INK, lw=0.7, ls="-", r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))


def img(ax, a, x, y, w, h=None, ec=None, lw=0.8, cmap=None, vmin=None, vmax=None, z=3):
    h = h if h is not None else w * a.shape[0] / a.shape[1]
    ax.imshow(a, extent=(x, x + w, y, y + h), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos", zorder=z,
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


def net(ax, x, y, s=1.0, c=LOW, label=True, stages=True):
    """Shared segmenter icon: 4 encoder stages (shrinking), all-MLP decoder, main head."""
    hs = [0.62, 0.50, 0.38, 0.28]
    xx = x
    for i, h in enumerate(hs):
        h *= s
        ax.add_patch(Polygon([(xx, y - h / 2), (xx + 0.11 * s, y - h / 2 + 0.05 * s), (xx + 0.11 * s, y + h / 2 + 0.05 * s),
                              (xx, y + h / 2)], closed=True, fc=NETF, ec=c, lw=0.7, zorder=4))
        if stages:
            ax.text(xx + 0.055 * s, y - h / 2 - 0.035, f"$s_{i + 1}$", ha="center", va="top", fontsize=5.6, color=c)
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


def draw(out_dir):
    T = np.load(resolve(DATA) / "train.npz")
    R = np.load(resolve(DATA) / "replay.npz")
    I = np.load(resolve(DATA) / "infer.npz")
    W, H, Y0 = 7.16, 4.45, 0.22  # drawing coordinates in inches; [Y0, H] is shown
    fig = plt.figure(figsize=(W, H - Y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")

    # ======================================================================= (a) training on two views
    panel(ax, 0.03, 2.30, 4.62, 2.12, "(a) Training one lightweight segmenter on two views")
    whole = overlay(T["whole"], T["whole_g"] > 0, GT, 0.9)
    h = img(ax, whole, 0.13, 3.38, 0.95, ec=LOW, lw=1.0)
    ax.text(0.13, 3.38 + h + 0.03, "whole image, letterbox $1024^2$", fontsize=5.6, color=LOW, va="bottom")
    nb = T["native_box"]
    sx = 0.95 / nb[3]
    ax.add_patch(Rectangle((0.13 + nb[1] * sx, 3.38 + h - (nb[0] + 384) * sx), 384 * sx, 384 * sx, fc="none", ec=NAT,
                           lw=0.8, zorder=6))
    nat = overlay(T["native"], T["native_g"], GT, 0.0)
    img(ax, nat, 0.40, 2.45, 0.66, 0.66, ec=NAT, lw=1.0)
    contour(ax, T["native_g"], 0.40, 2.45, 0.66, 0.66, c=GT, lw=0.5)
    arrow(ax, (0.13 + (nb[1] + 192) * sx, 3.38 + h - (nb[0] + 384) * sx), (0.73, 3.13), c=NAT, lw=0.6, ls=(0, (2, 1.2)))
    ax.text(0.37, 2.78, "native\n$384^2$ crop\n(defect-\ncentred,\n$p{=}0.5$)", fontsize=5.0, color=NAT, ha="right",
            va="center", linespacing=1.0)
    nx, ny = 1.42, 3.36
    xe = net(ax, nx, ny, 1.0)
    ax.text((nx + xe) / 2, ny + 0.47, "shared segmenter (MiT-B0, 3.8M)\nsame weights for both views", ha="center",
            va="bottom", fontsize=5.6, color=LOW, linespacing=1.05)
    arrow(ax, (1.10, 3.70), (nx - 0.02, ny + 0.12), c=LOW)
    arrow(ax, (1.08, 2.80), (nx - 0.02, ny - 0.15), c=NAT, rad=0.2)
    rbox(ax, xe + 0.08, ny - 0.12, 0.32, 0.24, fc=NETF, ec=LOW)
    ax.text(xe + 0.24, ny, "main\nhead", ha="center", va="center", fontsize=5.2, color=LOW, linespacing=0.95, zorder=5)
    arrow(ax, (xe, ny), (xe + 0.08, ny), c=LOW)
    # main-head supervision: size-aware weight on the native crop
    wx, wy, ws = 3.18, 3.02, 0.80
    img(ax, wblend(T["native"], T["native_w"], T["native_g"]), wx, wy, ws, ws, ec=SIZE, lw=1.0)
    ax.text(wx + ws / 2, wy + ws + 0.03, "target with size-aware weight $w(a)$", ha="center", va="bottom", fontsize=5.6,
            color=SIZE)
    cb = fig.add_axes([(wx + ws + 0.04) / W, (wy - Y0) / (H - Y0), 0.05 / W, ws / (H - Y0)])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(1, 5),
                                              cmap=matplotlib.colors.ListedColormap(plt.get_cmap("magma")(np.linspace(0.38, 1, 64)))),
                        cax=cb, ticks=[1, 3, 5])
    cbar.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    cbar.outline.set_linewidth(0.4)
    arrow(ax, (xe + 0.40, ny), (wx - 0.03, ny), c=SIZE)
    ax.text((xe + 0.40 + wx) / 2, ny + 0.05, "BCE$\cdot w$\n+ Dice", ha="center", va="bottom", fontsize=5.2, color=SIZE,
            linespacing=1.0)
    ax.text(wx + ws / 2 + 0.05, wy - 0.04, "small defects weigh up to $5\\times$\n(background 1)", ha="center", va="top",
            fontsize=5.0, color=SIZE, linespacing=1.0)
    # stage heads (training only)
    rbox(ax, 1.30, 2.38, 1.78, 0.56, fc="none", ec=MUTED, ls=(0, (2.5, 1.5)), lw=0.6)
    ts = 0.40
    img(ax, overlay(T["native"], T["native_small"], SIZE, 0.95), 1.36, 2.44, ts, ts, ec=SIZE, lw=0.8)
    img(ax, overlay(T["native"], T["native_g"], GT, 0.95), 1.84, 2.44, ts, ts, ec=GT, lw=0.8)
    for i in range(4):
        x0 = nx + 0.17 * i + 0.055
        top = ny - [0.62, 0.50, 0.38, 0.28][i] / 2 - 0.13
        arrow(ax, (x0, top), (1.56 if i < 2 else 2.04, 2.86), c=SIZE if i < 2 else GT, lw=0.5, hw=0.15)
    ax.text(2.30, 2.80, "stage heads (training only)", fontsize=5.2, color=MUTED, va="center")
    ax.text(2.30, 2.64, "$s_1,s_2$: small defects only", fontsize=5.2, color=SIZE, va="center")
    ax.text(2.30, 2.50, "$s_3,s_4$: all defects", fontsize=5.2, color=GT, va="center")
    ax.text(3.18, 2.48, "$\\mathcal{L}=\\mathcal{L}_{main}(w)+\\frac{\\lambda}{4}\\sum_k\\mathcal{L}_{s_k}$", fontsize=6.2,
            va="center", color=INK)

    # ======================================================================= (b) confusion replay
    panel(ax, 4.72, 2.30, 2.41, 2.12, "(b) Replaying the model's own confusions")
    rw = R["whole"]
    rx, ry, rwid = 4.82, 3.10, 1.12
    rh = img(ax, rw, rx, ry, rwid)
    contour(ax, R["whole_g"] > 0, rx, ry, rwid, rh, c=GT, lw=0.6)
    s = rwid / rw.shape[1]
    for fx, fy, sup in R["fp_xy"]:
        b = 0.06
        ax.add_patch(Rectangle((rx + fx * s - b / 2, ry + rh - fy * s - b / 2), b, b, fc="none", ec=REP, lw=0.6, zorder=6))
    ax.text(rx, ry + rh + 0.03, "native-tile prediction, training image", fontsize=5.2, color=INK, va="bottom")
    ax.text(rx, ry - 0.03, "□ false positives (print, vents, edges)", fontsize=5.0, color=REP, va="top")
    ax.text(rx, ry - 0.16, "— ground truth", fontsize=5.0, color=GT, va="top")
    cx0, cy0, cs = 6.06, 3.62, 0.33
    for i, c in enumerate(R["fp_crops"]):
        img(ax, c, cx0 + (i % 3) * (cs + 0.03), cy0, cs, cs, ec=REP, lw=0.8)
    ax.text(cx0, cy0 + cs + 0.03, f"mined FP crops ({int(R['n_fp_total']):,})", fontsize=5.2, color=REP, va="bottom")
    for i, c in enumerate(R["pos_crops"]):
        img(ax, c, cx0 + i * (cs + 0.03), cy0 - 0.50, cs, cs, ec=SIZE, lw=0.8)
    ax.text(cx0 + 2 * (cs + 0.03), cy0 - 0.50 + cs / 2, f"small-\ndefect\ncrops\n({int(R['n_pos_total'])})",
            fontsize=5.0, color=SIZE, va="center", linespacing=1.0)
    arrow(ax, (rx + rwid + 0.02, ry + rh * 0.75), (cx0 - 0.03, cy0 + cs / 2), c=REP)
    rbox(ax, 5.30, 2.40, 1.78, 0.46, fc="white", ec=REP, lw=0.8)
    ax.text(6.19, 2.63, "refine the same network for 20 epochs;\n1 of 4 batches is a replay batch\n"
            "(50% FP crops, 50% small-defect crops)", ha="center", va="center", fontsize=5.0, color=INK, linespacing=1.05)
    arrow(ax, (6.30, cy0 - 0.52), (6.30, 2.87), c=REP)

    # ======================================================================= (c) two-track inference
    panel(ax, 0.03, 0.26, 7.10, 1.98, "(c) Two-track inference with the same network")
    im_ = I["img"]
    iH, iW = im_.shape[:2]
    ix, iy, iw = 0.13, 1.28, 1.45
    ih = img(ax, im_, ix, iy, iw)
    contour(ax, I["g"] > 0, ix, iy, iw, ih, c=GT, lw=0.6)
    ax.text(ix, iy + ih + 0.03, "test image ($3840{\\times}2748$, lower part)", fontsize=5.4, va="bottom")
    ly = iy + ih / 2
    xe = net(ax, 1.98, ly, 0.5, LOW, label=False, stages=False)
    arrow(ax, (ix + iw + 0.02, ly), (1.96, ly), c=LOW)
    ax.text(1.88, ly + 0.25, "$\\downarrow1024^2$", fontsize=5.4, color=LOW, ha="center")
    ax.text((1.98 + xe) / 2, ly - 0.24, "track L", fontsize=5.6, color=LOW, ha="center", va="top")
    px, pw = xe + 0.18, 1.45
    ph = img(ax, blend(im_, I["L"]), px, iy, pw)
    contour(ax, I["M"] > 0, px, iy, pw, ph, c="#58c4dd", lw=0.5)
    arrow(ax, (xe, ly), (px - 0.02, ly), c=LOW)
    b = 384 * 900 / I["full_hw"][1] * pw / iW
    for cx_, cy_, _ in I["cand"]:
        ax.add_patch(Rectangle((px + cx_ * pw / iW - b / 2, iy + ph - cy_ * pw / iW - b / 2), b, b, fc="none", ec=NAT,
                               lw=0.9, zorder=7))
    ax.text(px, iy + ph + 0.03, "$p_L$ with candidates (top-$K$ peaks $>\\tau$) and support $M$", fontsize=5.3,
            va="bottom", color=LOW)
    # native track on a candidate
    zb = I["zoom_box"]
    zx0 = px + pw + 0.25
    zs = 0.60
    zi = I["zoom_img"]
    img(ax, zi, zx0, iy - 0.06, zs, zs, ec=NAT, lw=1.0)
    contour(ax, I["zoom_g"], zx0, iy - 0.06, zs, zs, c=GT, lw=0.5)
    ax.text(zx0 + zs / 2, iy - 0.06 + zs + 0.03, "native crop", ha="center", va="bottom", fontsize=5.3, color=NAT)
    cb_x, cb_y = px + zb[0] * pw / iW, iy + ph - zb[1] * pw / iW
    arrow(ax, (cb_x + b / 2, cb_y - b / 2), (zx0 - 0.02, iy + 0.12), c=NAT, rad=0.3)
    xe2 = net(ax, zx0 + zs + 0.14, iy + zs / 2 - 0.06, 0.5, NAT, label=False, stages=False)
    arrow(ax, (zx0 + zs + 0.01, iy + zs / 2 - 0.06), (zx0 + zs + 0.12, iy + zs / 2 - 0.06), c=NAT)
    ax.text((zx0 + zs + 0.14 + xe2) / 2, iy - 0.06 + zs / 2 - 0.30, "track S", fontsize=5.6, color=NAT, ha="center",
            va="top")
    cx1 = xe2 + 0.18
    img(ax, blend(zi, I["zoom_L"]), cx1, iy - 0.06, zs, zs, ec=LOW, lw=0.9)
    img(ax, blend(zi, I["zoom_P"]), cx1 + zs + 0.08, iy - 0.06, zs, zs, ec=NAT, lw=0.9)
    arrow(ax, (xe2, iy + zs / 2 - 0.06), (cx1 - 0.02, iy + zs / 2 - 0.06), c=NAT)
    ax.text(cx1 + zs / 2, iy - 0.06 + zs + 0.03, f"$p_L$ only: {I['zoom_L'].max():.2f}", ha="center", va="bottom",
            fontsize=5.3, color=LOW)
    ax.text(cx1 + 1.5 * zs + 0.08, iy - 0.06 + zs + 0.03, f"merged $p$: {I['zoom_P'].max():.2f}", ha="center",
            va="bottom", fontsize=5.3, color=NAT)
    # operating points
    oy = 0.33
    rbox(ax, 0.13, oy, 3.45, 0.60, fc="white", ec=NAT, lw=0.8)
    ax.text(0.20, oy + 0.55, "Precision mode", fontsize=6.0, weight="bold", color=NAT, va="top")
    ax.text(0.20, oy + 0.40, "L on the whole image; S only on the proposed crops; merge\n"
            "$p=\\max(p_L,\\ p_S\\,(1_M+0.5\\cdot 1_{\\bar{M}}))$: native evidence without low-res support is damped.\n"
            "3.6 native crops per image $\\Rightarrow$ 139 GFLOPs (1.6$\\times$ a single pass)",
            fontsize=5.3, va="top", linespacing=1.25)
    rbox(ax, 3.68, oy, 3.35, 0.60, fc="white", ec=MUTED, lw=0.8)
    ax.text(3.75, oy + 0.55, "Recall mode", fontsize=6.0, weight="bold", color=INK, va="top")
    ax.text(3.75, oy + 0.38, "S on all native $384^2$ tiles; maximum with L.\n"
            "26 crops per image $\\Rightarrow$ 429 GFLOPs; highest small-defect AUPRO", fontsize=5.3, va="top",
            linespacing=1.2)
    gx, gw = 6.05, 0.90
    gh = gw * iH / iW
    gy = oy + (0.62 - gh) / 2
    img(ax, im_, gx, gy, gw, gh)
    t = 384 * 900 / I["full_hw"][1] * gw / iW
    k = 1
    while k * t < gw:
        ax.plot([gx + k * t] * 2, [gy, gy + gh], color="white", lw=0.35, zorder=6)
        k += 1
    k = 1
    while k * t < gh:
        ax.plot([gx, gx + gw], [gy + gh - k * t] * 2, color="white", lw=0.35, zorder=6)
        k += 1

    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_framework.{ext}", dpi=300)
    print("saved", out / "fig1_framework.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/paper/figures")
    draw(ap.parse_args().out)
