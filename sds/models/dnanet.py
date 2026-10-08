"""External small-target baseline: DNANet (B. Li et al., "Dense Nested Attention Network for Infrared Small
Target Detection", IEEE TIP 2023; https://github.com/YeRen123455/Infrared-Small-Target-Detection, MIT).

Network imported unchanged from third_party/DNANet/model/model_DNANet.py with the authors' default setting
(channel_size "three" = [16, 32, 64, 128, 256], backbone "resnet_18" = Res_CBAM_block x [2, 2, 2, 2],
deep supervision on). Trained from scratch as in the original.
loss="softiou": the authors' SoftIoULoss (batch-level soft IoU, smooth 1) averaged over the four
deep-supervision outputs, Adagrad lr 0.05 (their recipe). loss="unified": BCE+Dice of our other baselines on the
same outputs. Inference uses the last output, as in the authors' test code.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..losses import bce_dice

_TP = Path(__file__).resolve().parents[2] / "third_party" / "DNANet"


def soft_iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1
    inter = pred * target
    return 1 - (inter.sum() + smooth) / (pred.sum() + target.sum() - inter.sum() + smooth)


class DNANetSDS(nn.Module):
    custom_loss = True

    def __init__(self, loss="softiou"):
        super().__init__()
        if str(_TP) not in sys.path:
            sys.path.insert(0, str(_TP))
        from model.model_DNANet import DNANet, Res_CBAM_block  # noqa: E402  (third_party, unchanged)

        self.net = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                          nb_filter=[16, 32, 64, 128, 256], deep_supervision=True)
        self.loss_kind = loss
        self._outs = []

    def forward(self, x, return_aux=None):
        outs = self.net(x)
        self._outs = outs if self.training else []
        return {"logits": outs[-1], "aux": {}}

    def loss_fn(self, out, target):
        if self.loss_kind == "softiou":
            return sum(soft_iou_loss(p.float(), target) for p in self._outs) / len(self._outs)
        valid = torch.ones_like(target)
        return sum(bce_dice(p.float(), target, valid) for p in self._outs) / len(self._outs)

    def encoder_parameters(self):
        return self.net.parameters()

    def head_parameters(self):
        return []

    def inference_modules(self):
        return [self.net]
