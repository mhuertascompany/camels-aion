#!/bin/bash
# Encode CAMELS maps with AION on a Jean Zay GPU node

#SBATCH --job-name=camels-encode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH -C v100-32g
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
module purge
module load pytorch-gpu/py3/2.2.0

# Activate virtualenv if provided (optional)
if [[ -n "${AION_VENV:-}" && -f "${AION_VENV}/bin/activate" ]]; then
  source "${AION_VENV}/bin/activate"
fi

# Ensure repository modules are importable
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Configuration (override via environment if needed)
# ------------------------------------------------------------------
BASE_PATH="${CAMELS_BASE_PATH:-/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data}"
MODEL_DIR="${AION_MODEL_DIR:-$WORK/models/aion}"
CODEC_REPO="${AION_CODEC_REPO:-$MODEL_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/embeddings/illustris_lh}"
SUITE="${SUITE:-IllustrisTNG}"
SET_NAME="${SET_NAME:-LH}"
REDSHIFT="${REDSHIFT:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ENC_TOKENS="${ENC_TOKENS:-600}"
DEVICE="${AION_DEVICE:-cuda}"
CODEC_DEVICE="${AION_CODEC_DEVICE:-cpu}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"
PREFIX="${PREFIX:-}"

mkdir -p "$OUTPUT_DIR" logs/slurm

# Optional start/end flags
START_FLAG=()
if [[ -n "$START_INDEX" ]]; then
  START_FLAG=(--start-index "$START_INDEX")
fi
END_FLAG=()
if [[ -n "$END_INDEX" ]]; then
  END_FLAG=(--end-index "$END_INDEX")
fi
PREFIX_FLAG=()
if [[ -n "$PREFIX" ]]; then
  PREFIX_FLAG=(--prefix "$PREFIX")
fi

# ------------------------------------------------------------------
# Run encoding
# ------------------------------------------------------------------
srun python scripts/encode_illustris_embeddings.py \
  --base-path "$BASE_PATH" \
  --suite "$SUITE" \
  --set "$SET_NAME" \
  --redshift "$REDSHIFT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --codec-device "$CODEC_DEVICE" \
  --num-encoder-tokens "$ENC_TOKENS" \
  --model-dir "$MODEL_DIR" \
  --codec-repo "$CODEC_REPO" \
  "${START_FLAG[@]}" \
  "${END_FLAG[@]}" \
  "${PREFIX_FLAG[@]}"
