"""U-Net / DeepLabV3+ (ResNet-34, ImageNet) baselines via segmentation_models_pytorch.

ImageNet weights are loaded from a local torchvision checkpoint so that training
works offline and does not depend on the smp weight-hosting scheme.
"""
import torch
import torch.nn as nn

from ..utils import resolve


class SMPBaseline(nn.Module):
    def __init__(self, arch="unet", encoder="resnet34", encoder_ckpt="pretrained/resnet34-b627a593.pth"):
        super().__init__()
        import segmentation_models_pytorch as smp

        cls = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[arch]
        self.net = cls(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
        if encoder_ckpt:  # None only when explicitly requested; a missing file is an error
            if not resolve(encoder_ckpt).exists():
                raise FileNotFoundError(f"ImageNet encoder weights not found: {resolve(encoder_ckpt)} (see README)")
            sd = torch.load(resolve(encoder_ckpt), map_location="cpu")
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            own = self.net.encoder.state_dict()
            matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            assert len(matched) == len(sd), f"unmatched encoder keys: {set(sd) - set(matched)}"
            own.update(matched)
            self.net.encoder.load_state_dict(own)
        self.head_type = arch
        self.dims = ()

    def encoder_parameters(self):
        return self.net.encoder.parameters()

    def head_parameters(self):
        enc = {id(p) for p in self.net.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]

    def forward(self, x, return_aux=None):
        return {"logits": self.net(x), "aux": {}}

    def inference_modules(self):
        return [self.net]
