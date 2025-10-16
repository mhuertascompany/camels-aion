#!/bin/bash
# Compare UMAP embeddings between two CAMELS suites

#SBATCH --job-name=camels-umap-compare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.2.0


pip install --user --quiet umap-learn seaborn

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

REF_MANIFEST=${REF_MANIFEST:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/IllustrisTNG_LH/IllustrisTNG_LH_z0p00_manifest.json}
REF_SHARD_DIR=${REF_SHARD_DIR:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/IllustrisTNG_LH}
TARGET_MANIFEST=${TARGET_MANIFEST:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/SIMBA_LH/SIMBA_LH_z0p00_manifest.json}
TARGET_SHARD_DIR=${TARGET_SHARD_DIR:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/SIMBA_LH}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/camels_aion/plots/umap_compare}
REF_LABEL=${REF_LABEL:-IllustrisTNG}
TARGET_LABEL=${TARGET_LABEL:-SIMBA}
MAX_POINTS=${MAX_POINTS:-5000}
SEED=${UMAP_SEED:-42}
COLOR_PARAM=${COLOR_PARAM:-Omega_m}

mkdir -p "$OUTPUT_DIR" logs/slurm

python scripts/compare_umap_embeddings.py \
  --ref-manifest "$REF_MANIFEST" \
  --ref-shard-dir "$REF_SHARD_DIR" \
  --target-manifest "$TARGET_MANIFEST" \
  --target-shard-dir "$TARGET_SHARD_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --ref-label "$REF_LABEL" \
  --target-label "$TARGET_LABEL" \
  --max-points "$MAX_POINTS" \
  --seed "$SEED" \
  --color-parameter "$COLOR_PARAM"
