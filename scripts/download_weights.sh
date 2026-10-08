#!/usr/bin/env bash
# ImageNet / ADE20K weights used by the paper (not redistributed here; each keeps its own license).
#   bash scripts/download_weights.sh            # everything
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p pretrained
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
for repo, out in [("nvidia/mit-b0", "pretrained/mit-b0"),                     # ours, SegFormer-B0
                  ("nvidia/mit-b5", "pretrained/mit-b5"),                     # SegFormer-B5
                  ("facebook/mask2former-swin-tiny-ade-semantic", "pretrained/mask2former-swin-tiny-ade")]:
    snapshot_download(repo, local_dir=out, allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors"])
p = hf_hub_download("timm/hrnet_w18_small_v2.ms_in1k", "model.safetensors")   # HRNet-W18-small
shutil.copy(p, "pretrained/hrnet_w18_small_v2.safetensors")
PY
# U-Net / DeepLabV3+ encoder (torchvision ResNet-34) and the BiSeNetV2 backbone of its reference implementation
wget -nc -O pretrained/resnet34-b627a593.pth https://download.pytorch.org/models/resnet34-b627a593.pth || true
wget -nc -O pretrained/bisenetv2_backbone.pth https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth || true
ls -l pretrained
