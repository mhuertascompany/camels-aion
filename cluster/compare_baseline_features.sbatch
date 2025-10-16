#!/bin/bash
# Shared UMAP comparison for baseline features

#SBATCH --job-name=camels-baseline-umap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.2.0

if [[ -n "${AION_VENV:-}" && -f "${AION_VENV}/bin/activate" ]]; then
  source "${AION_VENV}/bin/activate"
fi

pip install --user --quiet umap-learn seaborn

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

REF_FEATURES=${REF_FEATURES:?\"Please export REF_FEATURES\"}
TARGET_FEATURES=${TARGET_FEATURES:?\"Please export TARGET_FEATURES\"}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/camels_aion/baselines/umap_compare}
REF_LABEL=${REF_LABEL:-IllustrisTNG}
TARGET_LABEL=${TARGET_LABEL:-SIMBA}
COLOR_PARAM=${COLOR_PARAM:-Omega_m}
MAX_POINTS=${MAX_POINTS:-5000}
SEED=${UMAP_SEED:-42}

mkdir -p "$OUTPUT_DIR" logs/slurm

python scripts/compare_baseline_features.py \
  --ref-features "$REF_FEATURES" \
  --target-features "$TARGET_FEATURES" \
  --output-dir "$OUTPUT_DIR" \
  --ref-label "$REF_LABEL" \
  --target-label "$TARGET_LABEL" \
  --color-parameter "$COLOR_PARAM" \
  --max-points "$MAX_POINTS" \
  --seed "$SEED"
