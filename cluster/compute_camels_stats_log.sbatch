#!/bin/bash
# Compute log-scale normalization statistics for CAMELS maps

#SBATCH --job-name=camels-stats-log
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

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE_PATH=${CAMELS_BASE_PATH:-/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data}
SUITE=${SUITE:-IllustrisTNG}
SET_NAME=${SET_NAME:-LH}
REDSHIFT=${REDSHIFT:-0.0}
OUTPUT=${OUTPUT_STATS:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/norm/camels_TNG_log.json}
FIELDS=${FIELDS:-"Mstar Mgas T Z"}
SAMPLE_SIZE=${SAMPLE_SIZE:-2000}
PIXELS_PER_MAP=${PIXELS_PER_MAP:-4096}
CLIP=${CLIP:-1.5}
SEED=${STATS_SEED:-42}

python scripts/compute_camels_stats_log.py \
  --base-path "$BASE_PATH" \
  --suite "$SUITE" \
  --set "$SET_NAME" \
  --redshift "$REDSHIFT" \
  --output "$OUTPUT" \
  --fields $FIELDS \
  --sample-size "$SAMPLE_SIZE" \
  --pixels-per-map "$PIXELS_PER_MAP" \
  --clip "$CLIP" \
  --seed "$SEED"
