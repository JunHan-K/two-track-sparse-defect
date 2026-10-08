"""Generic binary-segmentation dataset driven by split CSV files.

Split CSV columns (written by tools/prepare_*.py):
    id, image, mask, label, fg_ratio, height, width
`image`/`mask` are paths relative to the project root; `mask` is empty for
normal images. Masks are binarised with `mask > 0`.

Preprocessing = aspect-preserving resize to fit `input_size` (H, W) followed by
right/bottom zero padding. A `valid` map marks non-padded pixels so that the
loss and the evaluation can ignore the padding.
"""
import csv
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils import resolve

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_split(path) -> list[dict]:
    """Read one split CSV, or several (list) concatenated (e.g. Core A + Mining B)."""
    if isinstance(path, (list, tuple)):
        rows = [r for p in path for r in read_split(p)]
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), "duplicate ids across splits"
        return rows
    with open(resolve(path)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["fg_ratio"] = float(r["fg_ratio"])
        r["height"] = int(r["height"])
        r["width"] = int(r["width"])
    return rows


def letterbox_params(h: int, w: int, input_size) -> tuple[int, int, float]:
    """Return (new_h, new_w, scale) for aspect-preserving fit into input_size."""
    H, W = input_size
    s = min(H / h, W / w)
    return min(H, int(round(h * s))), min(W, int(round(w * s))), s


def load_image(path):
    return Image.open(resolve(path)).convert("RGB")


def load_mask(path, size_hw):
    """Original-resolution binary mask (uint8 0/1). size_hw used for normals."""
    if not path:
        return np.zeros(size_hw, dtype=np.uint8)
    m = np.array(Image.open(resolve(path)))
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


class SegDataset(Dataset):
    def __init__(self, split_csv, input_size, train=False, aug=None, mask_resize_threshold=0.5, rows=None,
                 size_weight=None, size_split=None):
        self.rows = rows if rows is not None else read_split(split_csv)
        self.size_weight = size_weight  # {a_ref, beta, cap}: per-component weight for small defects (train)
        self.size_split = size_split  # {area_px, margin}: small-only targets for size-specialised heads (train)
        self.input_size = tuple(input_size)
        self.train = train
        self.aug = aug or {}
        self.mask_thr = mask_resize_threshold

    def __len__(self):
        return len(self.rows)

    def load_resized(self, idx):
        r = self.rows[idx]
        img = load_image(r["image"])
        w, h = img.size
        nh, nw, _ = letterbox_params(h, w, self.input_size)
        img = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        mask = load_mask(r["mask"], (h, w))
        if mask.shape != (h, w):
            raise ValueError(f"mask/image size mismatch for {r['id']}: {mask.shape} vs {(h, w)}")
        # BOX = area average; threshold keeps thin/small defects reasonably when downscaling.
        m = Image.fromarray(mask * 255).resize((nw, nh), Image.BOX)
        mask = (np.asarray(m, dtype=np.float32) / 255.0 >= self.mask_thr).astype(np.float32)

        H, W = self.input_size
        img_p = np.zeros((H, W, 3), dtype=np.float32)
        img_p[:nh, :nw] = img
        mask_p = np.zeros((H, W), dtype=np.float32)
        mask_p[:nh, :nw] = mask
        valid = np.zeros((H, W), dtype=np.float32)
        valid[:nh, :nw] = 1.0
        return img_p, mask_p, valid, (h, w, nh, nw)

    def augment(self, img, mask, valid):
        hf = bool(self.aug.get("hflip", 0) and random.random() < self.aug["hflip"])
        if hf:
            img, mask, valid = img[:, ::-1], mask[:, ::-1], valid[:, ::-1]
        vf = bool(self.aug.get("vflip", 0) and random.random() < self.aug["vflip"])
        if vf:
            img, mask, valid = img[::-1], mask[::-1], valid[::-1]
        b = self.aug.get("brightness", 0)
        c = self.aug.get("contrast", 0)
        if b or c:
            alpha = 1.0 + random.uniform(-c, c)
            beta = random.uniform(-b, b)
            v = valid[..., None]
            img = np.clip((img - 0.5) * alpha + 0.5 + beta, 0, 1) * v
        return img, mask, valid

    @staticmethod
    def to_tensor(img, mask, valid):
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img * valid[..., None]  # keep padding at exactly 0 after normalisation
        return (
            torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(mask[None])),
            torch.from_numpy(np.ascontiguousarray(valid[None])),
        )

    def __getitem__(self, idx):
        img, mask, valid, (h, w, nh, nw) = self.load_resized(idx)
        if self.train:
            img, mask, valid = self.augment(img, mask, valid)
        pw = size_weight_map(mask, **self.size_weight) if (self.train and self.size_weight) else None
        spec = (small_only_target(mask, valid, self.size_split["area_px"], self.size_split.get("margin", 3))
                if (self.train and self.size_split) else None)
        img, mask, valid = self.to_tensor(img, mask, valid)
        item = {
            "image": img, "mask": mask, "valid": valid, "index": idx,
            "orig_hw": torch.tensor([h, w]), "resized_hw": torch.tensor([nh, nw]),
        }
        if pw is not None:
            item["pos_weight"] = torch.from_numpy(pw[None])
        if spec is not None:
            item["small_mask"] = torch.from_numpy(spec[0][None])
            item["small_valid"] = torch.from_numpy(spec[1][None])
        return item


def small_only_target(mask, valid, area_px, margin=3):
    """Targets for size-specialised heads: defect components smaller than `area_px` (training
    resolution) stay positive; larger components and a `margin`-px band around them are IGNORED
    (not negatives), so the head is neither rewarded nor punished on large defects."""
    from scipy import ndimage

    lab, n = ndimage.label(mask > 0.5, structure=np.ones((3, 3)))
    small = np.zeros_like(mask, dtype=np.float32)
    keep = valid.astype(np.float32).copy()
    if n:
        area = ndimage.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        is_small = np.concatenate([[False], area < area_px])
        small = is_small[lab].astype(np.float32) * (lab > 0)
        large = (lab > 0) & ~is_small[lab]
        if large.any():
            large = ndimage.binary_dilation(large, iterations=margin) if margin else large
            keep[large] = 0.0
    return small, keep


def size_weight_map(mask: np.ndarray, a_ref: float = 400.0, beta: float = 1.0, cap: float = 4.0) -> np.ndarray:
    """Per-pixel BCE weight: 1 on background, 1 + beta * clip(sqrt(a_ref / area) - 1, 0, cap) on a defect
    component of `area` pixels (training resolution, 8-connectivity). Small components weigh more."""
    from scipy import ndimage

    w = np.ones(mask.shape, np.float32)
    lab, n = ndimage.label(mask > 0.5, structure=np.ones((3, 3)))
    if n:
        area = ndimage.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        cw = 1.0 + beta * np.clip(np.sqrt(a_ref / np.maximum(area, 1)) - 1.0, 0.0, cap)
        w = np.where(lab > 0, np.concatenate([[1.0], cw])[lab], 1.0).astype(np.float32)
    return w


class NativePointCropDataset(Dataset):
    """Native-resolution crops centred (with jitter) on given points of given images:
    entries = [{"row": split row, "cy": y, "cx": x}] in ORIGINAL pixel coordinates.
    Used to replay the model's own native-resolution false positives and small defects (confusion replay)."""

    def __init__(self, entries, crop=384, aug=None, jitter=None):
        self.entries = entries
        self.crop = crop
        self.jitter = crop // 4 if jitter is None else jitter
        self._helper = SegDataset(None, (crop, crop), train=True, aug=aug, rows=[e["row"] for e in entries])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        return self.get_point(e["row"], e["cy"], e["cx"])

    def get_point(self, r, cy, cx):
        img = np.asarray(load_image(r["image"]), dtype=np.float32) / 255.0
        h, w = img.shape[:2]
        mask = load_mask(r["mask"], (h, w)).astype(np.float32)
        c, j = self.crop, self.jitter
        cy, cx = int(cy) + random.randint(-j, j), int(cx) + random.randint(-j, j)
        y0 = int(np.clip(cy - c // 2, 0, max(0, h - c)))
        x0 = int(np.clip(cx - c // 2, 0, max(0, w - c)))
        img_p = np.zeros((c, c, 3), np.float32)
        mask_p = np.zeros((c, c), np.float32)
        valid = np.zeros((c, c), np.float32)
        ch, cw = min(c, h - y0), min(c, w - x0)
        img_p[:ch, :cw] = img[y0:y0 + ch, x0:x0 + cw]
        mask_p[:ch, :cw] = mask[y0:y0 + ch, x0:x0 + cw]
        valid[:ch, :cw] = 1.0
        img_p, mask_p, valid = self._helper.augment(img_p, mask_p, valid)
        img_t, mask_t, valid_t = SegDataset.to_tensor(img_p, mask_p, valid)
        return {"image": img_t, "mask": mask_t, "valid": valid_t}


class NativeCropDataset(Dataset):
    """Second training view: fixed-size crops at the ORIGINAL image resolution (no downscaling), so small
    defects keep their native pixel size. A crop is centred on a random defect component with probability
    p_defect, otherwise placed uniformly at random (normal images: always random). Same normalisation as
    SegDataset."""

    def __init__(self, rows, crop=384, p_defect=0.5, aug=None):
        self.rows = rows
        self.crop = crop
        self.p_defect = p_defect
        self.aug = aug or {}
        self._helper = SegDataset(None, (crop, crop), train=True, aug=aug, rows=rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from scipy import ndimage

        r = self.rows[i]
        img = np.asarray(load_image(r["image"]), dtype=np.float32) / 255.0
        h, w = img.shape[:2]
        mask = load_mask(r["mask"], (h, w)).astype(np.float32)
        c = self.crop
        if r["label"] == 1 and random.random() < self.p_defect:
            lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3)))
            j = random.randint(1, n)
            ys, xs = np.nonzero(lab == j)
            k = random.randrange(len(ys))
            cy, cx = ys[k] + random.randint(-c // 4, c // 4), xs[k] + random.randint(-c // 4, c // 4)
        else:
            cy, cx = random.randrange(h), random.randrange(w)
        y0 = int(np.clip(cy - c // 2, 0, max(0, h - c)))
        x0 = int(np.clip(cx - c // 2, 0, max(0, w - c)))
        img_p = np.zeros((c, c, 3), np.float32)
        mask_p = np.zeros((c, c), np.float32)
        valid = np.zeros((c, c), np.float32)
        ch, cw = min(c, h - y0), min(c, w - x0)
        img_p[:ch, :cw] = img[y0:y0 + ch, x0:x0 + cw]
        mask_p[:ch, :cw] = mask[y0:y0 + ch, x0:x0 + cw]
        valid[:ch, :cw] = 1.0
        img_p, mask_p, valid = self._helper.augment(img_p, mask_p, valid)
        img_t, mask_t, valid_t = SegDataset.to_tensor(img_p, mask_p, valid)
        return {"image": img_t, "mask": mask_t, "valid": valid_t, "idx": torch.tensor(i)}
