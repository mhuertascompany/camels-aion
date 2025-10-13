#!/bin/bash
# Train regression head on CAMELS embeddings using AION features

#SBATCH --job-name=camels-head-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.2.0

if [[ -n "${AION_VENV:-}" && -f "${AION_VENV}/bin/activate" ]]; then
  source "${AION_VENV}/bin/activate"
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MANIFEST=${MANIFEST:?"Please export MANIFEST pointing to the Illustris manifest JSON"}
SHARD_DIR=${SHARD_DIR:?"Please export SHARD_DIR containing the embedding shards"}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/heads/illustris}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${HEAD_BATCH_SIZE:-512}
LR=${LR:-3e-4}
DEVICE=${HEAD_DEVICE:-cuda}
TRAIN_FRAC=${TRAIN_FRAC:-0.7}
VAL_FRAC=${VAL_FRAC:-0.15}

mkdir -p "$OUTPUT_DIR" logs/slurm

python scripts/train_parameter_head.py \
  --manifest "$MANIFEST" \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --hidden-dim "$HIDDEN_DIM" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --train-frac "$TRAIN_FRAC" \
  --val-frac "$VAL_FRAC"
