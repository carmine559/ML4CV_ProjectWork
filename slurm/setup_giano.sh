#!/usr/bin/env bash
# One-time environment + data setup on giano.cs.unibo.it (the DISI cluster login node).
# Run it from the project folder, e.g. /scratch.hpc/$USER/prw-seqnet:
#   bash slurm/setup_giano.sh
#
# Everything goes under /scratch.hpc/$USER: the home quota is only 400 MB (a CUDA torch
# install alone is several GB). /scratch.hpc deletes files not accessed for 40 days.
# Needs a Kaggle API token in ~/.kaggle/access_token (kaggle.com -> Settings -> API ->
# "Generate New Token"), unless PRW is already unzipped in $DATA.
set -euo pipefail

BASE="${BASE:-/scratch.hpc/$USER}"
VENV="${VENV:-$BASE/venv-prw}"
DATA="${DATA:-$BASE/data/PRW}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$BASE/weights}"   # SOLIDER checkpoints
export TORCH_HOME="${TORCH_HOME:-$BASE/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/.cache}"
export TMPDIR="${TMPDIR_SETUP:-$BASE/tmp}"    # pip unpacks multi-GB wheels here
export PIP_NO_CACHE_DIR=1                     # never let pip cache in the home quota
# Driver 590 (CUDA 13.1) runs any CUDA 12.x/13.x PyTorch wheel; cu128 is what the code was tested with.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

mkdir -p "$BASE" "$TORCH_HOME" "$TMPDIR" "$(dirname "$DATA")"
echo "python: $(python3 --version)   (torch 2.7.1 needs 3.9-3.13)"

[[ -x "$VENV/bin/python3" ]] || python3 -m venv "$VENV"
PY="$VENV/bin/python3"
"$PY" -m pip install --no-cache-dir --upgrade pip
"$PY" -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX"
"$PY" -m pip install --no-cache-dir -r requirements.txt     # torch pins are already satisfied

# Cache the ImageNet ResNet-50 weights now: compute nodes may have no internet access.
"$PY" -c "import torch, torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1'); \
print('torch', torch.__version__, '| weights cached in', torch.hub.get_dir())"

PATH="$VENV/bin:$PATH" bash tools/download_prw.sh "$DATA"
# SOLIDER Swin-Small weights for configs/prw_solider_swin_s.yaml (skipped if already present)
PATH="$VENV/bin:$PATH" bash tools/download_solider.sh "$WEIGHTS_DIR" small

rm -rf "$TMPDIR"
du -sh "$VENV" "$DATA" "$TORCH_HOME" "$WEIGHTS_DIR" 2>/dev/null || true
echo "quota check (home must stay < 400 MB):"; du -sh "$HOME" 2>/dev/null || true
echo "Setup done. Next: mkdir -p logs && SMOKE=1 sbatch --time=01:00:00 slurm/train_prw.sbatch"
