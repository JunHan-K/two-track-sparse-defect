#!/usr/bin/env bash
# Run ONCE inside the GPU container.  Usage: bash scripts/setup_env.sh [cu121]
set -e
CU=${1:-cu121}
conda create -y -n sds python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sds
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/${CU}
pip install -r requirements.txt
python - <<'PY'
import torch, transformers, segmentation_models_pytorch as smp
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "smp", smp.__version__)
PY
mkdir -p results
pip freeze > results/requirements_frozen.txt
python -m pytest tests -q   # CPU unit tests, no data needed
