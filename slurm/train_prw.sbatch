#!/bin/bash
# SeqNet / PRW training + evaluation on the DISI GPU cluster, partition l40
# (1x L40 48 GB, 8-core CPU, 64 GB RAM; driver 590 / CUDA 13.1; SLURM 24.11).
# One-time setup on giano: bash slurm/setup_giano.sh
# Submit from the project folder on giano (e.g. /scratch.hpc/$USER/prw-seqnet):
#   mkdir -p logs
#   SMOKE=1 sbatch --time=01:00:00 slurm/train_prw.sbatch   # 20-iteration pipeline check + full eval
#   sbatch slurm/train_prw.sbatch                           # official recipe, fp32 (baseline)
#   AMP=bf16 sbatch slurm/train_prw.sbatch                  # faster autocast, not bit-exact official
# Re-submitting (or the automatic requeue) resumes from the newest epoch_*.pth in $OUT.
#SBATCH --job-name=prw
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00              # must not exceed the l40 limit: sinfo -p l40 -o "%P %l"
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append           # keep one log across requeues
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=carmine.santella@studio.unibo.it
#SBATCH --requeue
#SBATCH --signal=B:USR1@300          # 5 min before the time limit -> requeue (resume)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE="${BASE:-/scratch.hpc/$USER}"
PY="${VENV:-$BASE/venv-prw}/bin/python3"
export PRW_ROOT="${PRW_ROOT:-$BASE/data/PRW}"
export TORCH_HOME="${TORCH_HOME:-$BASE/.cache/torch}"       # ResNet-50 weights cached by setup
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/.cache}"     # keep caches out of the 400 MB home
export PYTHONUNBUFFERED=1
AMP="${AMP:-off}"                                           # off = official fp32 | bf16 | fp16

if [[ "${SMOKE:-0}" == 1 ]]; then
  OUT="${OUT:-$PWD/exp_smoke}"; EXTRA=(--epochs 1 --max-iters 20)
else
  OUT="${OUT:-$PWD/exp_prw}"; EXTRA=()
fi
mkdir -p "$OUT" logs

echo "[$(date)] job $SLURM_JOB_ID on $(hostname) | restart count ${SLURM_RESTART_COUNT:-0} | AMP=$AMP | OUT=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY is missing or not runnable on $(hostname)." >&2
  if [[ -L "$PY" ]]; then
    echo "  It links to $(readlink -f "$PY"), which does not exist on this compute node." >&2
  else
    echo "  Create the venv first, on giano: bash slurm/setup_giano.sh" >&2
  fi
  exit 1
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

trap 'echo "[$(date)] time limit close: requeueing $SLURM_JOB_ID"; \
  scontrol requeue "$SLURM_JOB_ID" || echo "requeue refused: re-run sbatch, training resumes by itself"; exit 0' USR1

# ---- train (official recipe: bs 5, lr 0.003, 18 epochs, lr x0.1 at 16) ----
"$PY" Scripts/train.py --cfg configs/prw.yaml --data-root "$PRW_ROOT" --output-dir "$OUT" \
  --amp "$AMP" --workers 6 --resume auto ${EXTRA[@]+"${EXTRA[@]}"} &
wait $!

# ---- evaluate the last epoch with the unchanged course eval_search_prw ----
LAST="$(ls -v "$OUT"/epoch_*.pth | tail -1)"
NAME="$(basename "$LAST" .pth)"
"$PY" Scripts/eval_prw.py --ckpt "$LAST" --data-root "$PRW_ROOT" --workers 4 \
  --cache "$OUT/features_$NAME.pt" --out "$OUT/eval_$NAME.json" &
wait $!
echo "[$(date)] done: $OUT/eval_$NAME.json"
