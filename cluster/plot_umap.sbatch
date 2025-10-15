#!/bin/bash
# Generate UMAP scatter plots for CAMELS embeddings

#SBATCH --job-name=camels-umap
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

if [[ -n "${AION_VENV:-}" && -f "${AION_VENV}/bin/activate" ]]; then
  source "${AION_VENV}/bin/activate"
fi

pip install --user --quiet umap-learn seaborn

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SHARD_DIR=${SHARD_DIR:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/IllustrisTNG_LH}
MANIFEST=${MANIFEST:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/IllustrisTNG_LH/IllustrisTNG_LH_z0p00_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/plots}
MAX_POINTS=${MAX_POINTS:-5000}
SEED=${UMAP_SEED:-42}

mkdir -p "$OUTPUT_DIR" logs/slurm

python scripts/plot_umap_embeddings.py \
  --shard-dir "$SHARD_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --max-points "$MAX_POINTS" \
  --seed "$SEED"
