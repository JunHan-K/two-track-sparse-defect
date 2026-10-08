"""External baseline: MSHNet (Liu et al., "Infrared Small Target Detection with Scale and Location
Sensitivity", CVPR 2024; https://github.com/ying-fu/MSHNet), a small-target segmentation SOTA.

The network is imported unchanged from third_party/MSHNet/model/MSHNet.py (not redistributed).
The SLS loss (scale- and location-sensitive IoU loss with multi-scale deep supervision and a warm-up
during which only the full-resolution output is trained) is re-implemented here line by line from
third_party/MSHNet/model/loss.py + main.py, because the original file imports scikit-image only for
unused code. Training recipe of the original: Adagrad, lr 0.05, warm-up 5 epochs.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_TP = Path(__file__).resolve().parents[2] / "third_party" / "MSHNet"


def _lloss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)
    n, _, h, w = pred.shape
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(n):
        pcx, pcy = (x_index * pred[i]).mean(), (y_index * pred[i]).mean()
        tcx, tcy = (x_index * target[i]).mean(), (y_index * target[i]).mean()
        angle = (4 / (torch.pi ** 2)) * torch.square(torch.arctan(pcy / (pcx + smooth)) - torch.arctan(tcy / (tcx + smooth)))
        pl = torch.sqrt(pcx * pcx + pcy * pcy + smooth)
        tl = torch.sqrt(tcx * tcx + tcy * tcy + smooth)
        length = torch.min(pl, tl) / (torch.max(pl, tl) + smooth)
        loss = loss + (1 - length + angle) / n
    return loss


def sls_iou_loss(pred_log, target, warm_epoch, epoch, with_shape=True):
    pred = torch.sigmoid(pred_log)
    smooth = 0.0
    inter = (pred * target).sum(dim=(1, 2, 3))
    ps, ts = pred.sum(dim=(1, 2, 3)), target.sum(dim=(1, 2, 3))
    dis = torch.pow((ps - ts) / 2, 2)
    alpha = (torch.min(ps, ts) + dis + smooth) / (torch.max(ps, ts) + dis + smooth)
    iou = (inter + smooth) / (ps + ts - inter + smooth)
    if epoch > warm_epoch:
        siou = alpha * iou
        return 1 - siou.mean() + (_lloss(pred, target) if with_shape else 0.0)
    return 1 - iou.mean()


class MSHNetSDS(nn.Module):
    def __init__(self, warm_epoch=5, loss="sls"):
        super().__init__()
        self.loss_kind = loss  # "sls" (authors) or "unified" (our BCE+Dice on the final output, sds.losses)
        if str(_TP) not in sys.path:
            sys.path.insert(0, str(_TP))
        from model.MSHNet import MSHNet  # noqa: E402  (third_party, unchanged)

        self.net = MSHNet(3)
        self.warm_epoch = warm_epoch
        self.epoch = 10 ** 9  # inference: after warm-up (multi-scale fused output), as in the original test
        self.down = nn.MaxPool2d(2, 2)
        self._masks = []

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, return_aux=None):
        masks, out = self.net(x, self.epoch > self.warm_epoch)
        self._masks = masks
        return {"logits": out, "aux": {}}

    @property
    def custom_loss(self):
        return self.loss_kind == "sls"

    def loss_fn(self, out, target):
        """Original training loss: SLS on the final output + each multi-scale mask (max-pooled labels)."""
        e, w = self.epoch, self.warm_epoch
        loss = sls_iou_loss(out["logits"].float(), target, w, e)
        lab = target
        for j, m in enumerate(self._masks):
            if j > 0:
                lab = self.down(lab)
            loss = loss + sls_iou_loss(m.float(), lab, w, e)
        return loss / (len(self._masks) + 1)

    def encoder_parameters(self):
        return self.net.parameters()

    def head_parameters(self):
        return []

    def inference_modules(self):
        return [self.net]
