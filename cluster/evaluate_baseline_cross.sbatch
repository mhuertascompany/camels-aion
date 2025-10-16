#!/bin/bash
# Evaluate a trained baseline model on a different CAMELS suite

#SBATCH --job-name=camels-baseline-cross
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.2.0


pip install --user --quiet umap-learn seaborn

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/baselines/IllustrisTNG_LH/cnn/best_model.pt}
BASE_PATH=${CAMELS_BASE_PATH:-/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data}
SUITE=${SUITE:-SIMBA}
SET_NAME=${SET_NAME:-LH}
REDSHIFT=${REDSHIFT:-0.0}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/data/camels_aion/baselines/${SUITE}_${SET_NAME}/cross_eval}
NORMALIZATION_STATS=${NORM_STATS:--/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/norm/camels_SIMBA_log.json}
NORMALIZATION_CLIP=${NORM_CLIP:-1.5}
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE=${BASELINE_DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-8}

mkdir -p "$OUTPUT_DIR" logs/slurm

python scripts/evaluate_baseline_cross.py \
  --model "$MODEL_PATH" \
  --base-path "$BASE_PATH" \
  --suite "$SUITE" \
  --set "$SET_NAME" \
  --redshift "$REDSHIFT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS" \
  ${NORMALIZATION_STATS:+--normalization-stats "$NORMALIZATION_STATS"} \
  --normalization-clip "$NORMALIZATION_CLIP"
