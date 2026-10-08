def build_model(cfg):
    m = cfg["model"]
    t = m.get("type", "segformer")
    if t == "segformer":
        from .segformer import SegFormerSDS

        return SegFormerSDS(
            pretrained=m["pretrained"] if "pretrained" in m else "pretrained/mit-b0",
            head=m.get("head", "original"),
            rep_dim=m.get("rep_dim", 128),
            head_hidden=m.get("head_hidden", 32),
            decoder_dim=m.get("decoder_dim", 256),
            decoder_dropout=m.get("decoder_dropout", 0.1),
            fusion=m.get("fusion", "none"),
        )
    if t in ("unet", "deeplabv3plus"):
        from .baselines import SMPBaseline

        return SMPBaseline(arch=t, encoder=m.get("encoder", "resnet34"),
                           encoder_ckpt=m["encoder_ckpt"] if "encoder_ckpt" in m else "pretrained/resnet34-b627a593.pth")
    if t == "mshnet":  # small-target model (CVPR 2024), see sds/models/mshnet.py
        from .mshnet import MSHNetSDS

        return MSHNetSDS(warm_epoch=m.get("warm_epoch", 5), loss=m.get("mshnet_loss", "sls"))
    if t == "dnanet":  # external small-target baseline (sds/models/dnanet.py)
        from .dnanet import DNANetSDS

        return DNANetSDS(loss=m.get("dnanet_loss", "softiou"))
    if t == "hrnet_w18_small":  # Defect Spectrum benchmark model (sds/models/benchmark_models.py)
        from .benchmark_models import HRNetW18SmallSeg

        return HRNetW18SmallSeg(weights=m["weights"] if "weights" in m else "pretrained/hrnet_w18_small_v2.safetensors")
    if t == "mask2former":  # Mask2Former (Swin-T), sds/models/mask2former.py
        from .mask2former import Mask2FormerSDS

        return Mask2FormerSDS(m.get("weights", "pretrained/mask2former-swin-tiny-ade"))
    if t == "bisenetv2":  # Defect Spectrum benchmark model
        from .benchmark_models import BiSeNetV2Seg

        return BiSeNetV2Seg(weights=m["weights"] if "weights" in m else "pretrained/bisenetv2_backbone.pth")
    raise ValueError(t)
