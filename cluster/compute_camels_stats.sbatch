#!/bin/bash
# Compute per-field statistics for CAMELS maps to support normalization

#SBATCH --job-name=camels-stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
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

BASE_PATH=${CAMELS_BASE_PATH:?"Please export CAMELS_BASE_PATH"}
SUITE=${SUITE:-IllustrisTNG}
SET_NAME=${SET_NAME:-LH}
REDSHIFT=${REDSHIFT:-0.0}
OUTPUT=${OUTPUT_STATS:?"Please export OUTPUT_STATS for the stats JSON"}
FIELDS=${FIELDS:-"Mstar Mgas T Z"}
SAMPLE_SIZE=${SAMPLE_SIZE:-5000}
CLIP=${CLIP:-1.5}
SEED=${STATS_SEED:-42}

python scripts/compute_camels_stats.py \
  --base-path "$BASE_PATH" \
  --suite "$SUITE" \
  --set "$SET_NAME" \
  --redshift "$REDSHIFT" \
  --output "$OUTPUT" \
  --fields $FIELDS \
  --sample-size "$SAMPLE_SIZE" \
  --clip "$CLIP" \
  --seed "$SEED"
