"""SegFormer (MiT encoder + all-MLP decoder) with the heads used in the paper.

    x -> MiT encoder -> all-MLP decoder -> F (256-d, 1/4 resolution)
      original  : F -> 1x1 conv -> logit                                   (SegFormer baselines)
      stagewise : F -> Z = P(F) (128-d) -> H(Z) -> logit                   (ours)
                  + heads s1..s4 on the four encoder stages (strides 4..32). They are trained with auxiliary
                  losses (s1, s2 on small defects only, s3, s4 on all defects; tools/train.py) and are NOT
                  executed at inference, so the deployed network is the plain segmenter.
      fusion="small" (ablation only): the main logit is raised by shallow-stage (s1, s2) evidence through a
                  gate driven by the deep stages (s3, s4) -- the inference-time fusion that did not help.

Only the main head is executed when `return_aux=False` (inference, efficiency measurements).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    """SegFormer all-MLP decode head without the classifier (output = fused feature F)."""

    def __init__(self, in_channels, embed_dim=256, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)  # as in the original SegFormer decode head

    def forward(self, feats):
        size = feats[0].shape[-2:]
        outs = [F.interpolate(p(f), size=size, mode="bilinear", align_corners=False) for p, f in zip(self.proj, feats)]
        return self.dropout(self.fuse(torch.cat(outs[::-1], dim=1)))


class Head(nn.Module):
    """1x1 conv (in -> hidden) -> GELU -> 1x1 conv (hidden -> 1)."""

    def __init__(self, in_ch, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        return self.net(x)


class SegFormerSDS(nn.Module):
    def __init__(self, pretrained="pretrained/mit-b0", head="original", rep_dim=128, head_hidden=32,
                 decoder_dim=256, decoder_dropout=0.1, fusion="none"):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel

        from ..utils import resolve

        if pretrained is None:  # unit tests only: MiT-B0 architecture, random init
            self.encoder = SegformerModel(SegformerConfig())
        else:  # a missing path is an error: never silently train from scratch
            path = resolve(pretrained)
            if not (path / "pytorch_model.bin").exists() and not (path / "model.safetensors").exists():
                raise FileNotFoundError(f"pretrained MiT weights not found in {path} (scripts/download_weights.sh)")
            self.encoder = SegformerModel.from_pretrained(str(path))
        hs = list(self.encoder.config.hidden_sizes)
        self.decoder = MLPDecoder(hs, decoder_dim, decoder_dropout)
        assert head in ("original", "stagewise"), head
        assert fusion in ("none", "small"), fusion
        assert fusion == "none" or head == "stagewise", "fusion needs stage heads"
        self.head_type, self.fusion = head, fusion
        self.aux_keys = []
        if head == "original":
            self.classifier = nn.Conv2d(decoder_dim, 1, 1)
            self.dims = ()
            return
        self.dims = (rep_dim,)
        self.proj = nn.Sequential(nn.Conv2d(decoder_dim, rep_dim, 1))
        self.heads = nn.ModuleDict({str(rep_dim): Head(rep_dim, head_hidden)})
        self.aux_keys = [f"s{i + 1}" for i in range(len(hs))]
        for i, c in enumerate(hs):
            self.heads[f"s{i + 1}"] = Head(c, head_hidden)
        if fusion == "small":
            # shallow (s1, s2) evidence is ADDED to the main logit where a deep-context gate (s3, s4, main) opens
            self.fuse_add = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))
            nn.init.zeros_(self.fuse_add[2].weight)
            nn.init.constant_(self.fuse_add[2].bias, -6.0)  # softplus(-6) ~ 0: starts as the plain model
            self.fuse_gate = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 1, 3, padding=1))
            self.fuse_head = nn.ModuleList([self.fuse_add, self.fuse_gate])

    def encoder_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        enc = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]

    def forward(self, x, return_aux=None):
        """Returns {'logits': main logit, 'aux': {stage: logit}} at input resolution."""
        if return_aux is None:
            return_aux = self.training
        size = x.shape[-2:]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)  # noqa: E731
        stages = list(self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True).hidden_states)
        f = self.decoder(stages)
        if self.head_type == "original":
            return {"logits": up(self.classifier(f)), "aux": {}}
        z = self.proj(f)
        if self.fusion != "none":
            return self._forward_fused(z, stages, up, return_aux)
        aux = {f"s{i + 1}": up(self.heads[f"s{i + 1}"](s)) for i, s in enumerate(stages)} if return_aux else {}
        return {"logits": up(self.heads[str(self.dims[-1])](z)), "aux": aux}

    def _forward_fused(self, z, stages, up, return_aux):
        upz = lambda t: F.interpolate(t, size=z.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        main = self.heads[str(self.dims[-1])](z)
        st = [upz(self.heads[f"s{i + 1}"](s)) for i, s in enumerate(stages)]
        pm = torch.sigmoid(main)
        shallow = [st[0], st[1], (torch.sigmoid(st[0]) - pm).abs(), (torch.sigmoid(st[1]) - pm).abs()]
        add = F.softplus(self.fuse_add(torch.cat([main] + shallow, dim=1)))
        gate = torch.sigmoid(self.fuse_gate(torch.cat([main, st[2], st[3]], dim=1)))
        aux = {}
        if return_aux:
            aux = {f"s{i + 1}": up(l) for i, l in enumerate(st)}
            aux["pre"] = up(main)
        return {"logits": up(main + gate * add), "aux": aux}

    def inference_modules(self):
        """Modules used at inference (parameter counting)."""
        mods = [self.encoder, self.decoder]
        if self.head_type == "original":
            return mods + [self.classifier]
        mods += [self.proj, self.heads[str(self.dims[-1])]]
        if self.fusion != "none":
            mods += [self.heads[k] for k in self.aux_keys] + [self.fuse_head]
        return mods
