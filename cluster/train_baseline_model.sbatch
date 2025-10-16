#!/bin/bash
# Train a CNN/ViT baseline on CAMELS maps

#SBATCH --job-name=camels-baseline-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH -C v100-32g
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.2.0


pip install --user --quiet umap-learn seaborn

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_TYPE=${MODEL_TYPE:-cnn}
BASE_PATH=${CAMELS_BASE_PATH:-/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data}
SUITE=${SUITE:-IllustrisTNG}
SET_NAME=${SET_NAME:-LH}
REDSHIFT=${REDSHIFT:-0.0}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/data/camels_aion/baselines/${SUITE}_${SET_NAME}/${MODEL_TYPE}}
NORMALIZATION_STATS=${NORM_STATS:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/norm/camels_TNG_log.json}
NORMALIZATION_CLIP=${NORM_CLIP:-1.5}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-50}
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
DEVICE=${BASELINE_DEVICE:-cuda}
SEED=${BASELINE_SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-8}

mkdir -p "$OUTPUT_DIR" logs/slurm

python scripts/train_baseline_model.py \
  --model "$MODEL_TYPE" \
  --base-path "$BASE_PATH" \
  --suite "$SUITE" \
  --set "$SET_NAME" \
  --redshift "$REDSHIFT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  ${NORMALIZATION_STATS:+--normalization-stats "$NORMALIZATION_STATS"} \
  --normalization-clip "$NORMALIZATION_CLIP"
