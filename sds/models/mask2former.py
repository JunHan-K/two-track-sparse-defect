"""Mask2Former baseline (Cheng et al., CVPR 2022; Defect Spectrum benchmark model),
Hugging Face implementation (transformers.Mask2FormerForUniversalSegmentation) with the Swin-T ADE20K semantic
checkpoint (facebook/mask2former-swin-tiny-ade-semantic, pretrained/mask2former-swin-tiny-ade), re-headed for two
classes (background, defect). Training: the model's own mask-classification loss (Hungarian matching, CE + mask BCE +
Dice, as in Mask2Former); per image the targets are the background mask and, if present, the defect mask.
Inference: semantic map = sum_q softmax(class_q)[c] * sigmoid(mask_q) (Mask2Former's semantic inference); defect
probability = defect / (background + defect), upsampled to the input size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import resolve


class Mask2FormerSDS(nn.Module):
    custom_loss = True

    def __init__(self, weights="pretrained/mask2former-swin-tiny-ade"):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation

        self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(resolve(weights)), num_labels=2, id2label={0: "background", 1: "defect"},
            label2id={"background": 0, "defect": 1}, ignore_mismatched_sizes=True)
        self._x = None

    def forward(self, x, return_aux=None):
        if self.training:  # the loss needs the targets: the forward pass is run in loss_fn
            self._x = x
            return {"logits": x.new_zeros(x.shape[0], 1, *x.shape[-2:]), "aux": {}}
        o = self.net(pixel_values=x)
        cls = o.class_queries_logits.double().softmax(-1)[..., :-1]
        seg = torch.einsum("bqc,bqhw->bchw", cls, o.masks_queries_logits.double().sigmoid())
        p = (seg[:, 1] / (seg.sum(1) + 1e-6))[:, None]
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # exact logit in float64 (no clamping: clamping p to [1e-6, 1 - 1e-6] created ties at both ends)
        return {"logits": (torch.log(p) - torch.log1p(-p)).float(), "aux": {}}

    def loss_fn(self, out, target):
        masks, classes = [], []
        for t in target:
            fg = (t[0] > 0.5).float()
            ms, cs = [1.0 - fg], [0]
            if fg.any():
                ms.append(fg)
                cs.append(1)
            masks.append(torch.stack(ms))
            classes.append(torch.tensor(cs, device=t.device))
        return self.net(pixel_values=self._x, mask_labels=masks, class_labels=classes).loss

    def encoder_parameters(self):
        return self.net.model.pixel_level_module.encoder.parameters()

    def head_parameters(self):
        enc = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.net.parameters() if id(p) not in enc]

    def inference_modules(self):
        return [self.net]
