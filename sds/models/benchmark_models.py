"""Two models of the Defect Spectrum segmentation benchmark (Yang et al., ECCV 2024), which evaluated
supervised defect segmentation on MVTec AD and VISION with UNet, PSPNet, DeepLabV3+, HRNet-W18-small,
BiSeNetV2, Segmenter, SegFormer-B0 and Mask2Former. Same training recipe as our other baselines.

- HRNet-W18-small (Wang et al., TPAMI 2021): timm backbone with ImageNet weights
  (pretrained/hrnet_w18_small_v2.safetensors, timm/hrnet_w18_small_v2.ms_in1k), HRNetV2 head: the four branch
  outputs (strides 4-32) are upsampled to stride 4, concatenated, 1x1 conv-BN-ReLU, 1x1 conv to a logit.
- BiSeNetV2 (Yu et al., IJCV 2021): reference implementation third_party/BiSeNet (CoinCheung, MIT), unchanged,
  with its ImageNet backbone (pretrained/bisenetv2_backbone.pth); its four auxiliary heads are trained with
  weight 1 as in that implementation.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import bce_dice
from ..utils import resolve


class HRNetW18SmallSeg(nn.Module):
    def __init__(self, weights="pretrained/hrnet_w18_small_v2.safetensors"):
        super().__init__()
        import timm

        kw = dict(features_only=True, feature_location="", out_indices=(1, 2, 3, 4))
        if weights:
            p = resolve(weights)
            if not p.exists():
                raise FileNotFoundError(p)
            self.backbone = timm.create_model("hrnet_w18_small_v2", pretrained=True,
                                              pretrained_cfg_overlay=dict(file=str(p)), **kw)
        else:
            self.backbone = timm.create_model("hrnet_w18_small_v2", pretrained=False, **kw)
        c = sum(self.backbone.feature_info.channels())
        self.head = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, 1, 1))

    def forward(self, x, return_aux=None):
        fs = self.backbone(x)
        s = fs[0].shape[-2:]
        f = torch.cat([fs[0]] + [F.interpolate(t, size=s, mode="bilinear", align_corners=False) for t in fs[1:]], 1)
        out = F.interpolate(self.head(f), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": out, "aux": {}}

    def encoder_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()

    def inference_modules(self):
        return [self.backbone, self.head]


class BiSeNetV2Seg(nn.Module):
    custom_loss = True

    def __init__(self, weights="pretrained/bisenetv2_backbone.pth"):
        super().__init__()
        root = Path(__file__).resolve().parents[2] / "third_party" / "BiSeNet"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import lib.models.bisenetv2 as bv2  # noqa: E402  (third_party, unchanged)

        local = resolve(weights) if weights else None
        orig = bv2.modelzoo.load_url
        if local is not None:
            if not local.exists():
                raise FileNotFoundError(local)
            bv2.modelzoo.load_url = lambda *a, **k: torch.load(local, map_location="cpu")
        else:
            bv2.BiSeNetV2.load_pretrain = lambda self: None
        try:
            self.net = bv2.BiSeNetV2(1, aux_mode="train")
        finally:
            bv2.modelzoo.load_url = orig
        self._aux = []

    def forward(self, x, return_aux=None):
        self.net.aux_mode = "train" if self.training else "eval"
        outs = self.net(x)
        self._aux = list(outs[1:]) if self.training else []
        return {"logits": outs[0], "aux": {}}

    def loss_fn(self, out, target):
        valid = torch.ones_like(target)
        loss = bce_dice(out["logits"].float(), target, valid)
        for a in self._aux:
            loss = loss + bce_dice(a.float(), target, valid)
        return loss

    def encoder_parameters(self):
        return self.net.parameters()

    def head_parameters(self):
        return []

    def inference_modules(self):
        return [self.net]
