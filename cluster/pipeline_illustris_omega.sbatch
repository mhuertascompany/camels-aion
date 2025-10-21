#!/bin/bash
# Full pipeline: encode IllustrisTNG LH embeddings and train Omega_m head

#SBATCH --job-name=camels-omega
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH -C v100-32g
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.2.0

if [[ -n "${AION_VENV:-}" && -f "${AION_VENV}/bin/activate" ]]; then
  source "${AION_VENV}/bin/activate"
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE_PATH=${CAMELS_BASE_PATH:-/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data}
MODEL_DIR=${AION_MODEL_DIR:-$WORK/models/aion}
CODEC_REPO=${AION_CODEC_REPO:-$MODEL_DIR}
SUITE=${SUITE:-IllustrisTNG}
SET_NAME=${SET_NAME:-LH}
REDSHIFT=${REDSHIFT:-0.0}
BATCH_SIZE=${BATCH_SIZE:-32}
ENC_TOKENS=${ENC_TOKENS:-600}
DEVICE=${AION_DEVICE:-cuda}
CODEC_DEVICE=${AION_CODEC_DEVICE:-cpu}
PREFIX=${PREFIX:-${SUITE}_${SET_NAME}_z0p00}
SKIP_ENCODING=${SKIP_ENCODING:-1}
NORM_STATS=${NORM_STATS:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/norm/camels_TNG_log.json}
NORM_CLIP=${NORM_CLIP:-1.5}

EMBED_DIR=${EMBED_DIR:-$WORK/data/camels_aion/embeddings/${SUITE}_${SET_NAME}}
HEAD_OUT=${HEAD_OUT:-$WORK/data/camels_aion/heads/${SUITE}_${SET_NAME}_omega}

mkdir -p "$EMBED_DIR" "$HEAD_OUT" logs/slurm

MANIFEST_PATH="$EMBED_DIR/${PREFIX}_manifest.json"

# ------------------------------------------------------------------
# Step 1: Encode embeddings (optional)
# ------------------------------------------------------------------
if [[ "$SKIP_ENCODING" != "1" ]]; then
  echo "[INFO] Running embedding step..."
  NORM_ARGS=()
  if [[ -n "$NORM_STATS" ]]; then
    NORM_ARGS+=(--normalization-stats "$NORM_STATS" --normalization-clip "$NORM_CLIP")
  fi
  srun python scripts/encode_illustris_embeddings.py \
    --base-path "$BASE_PATH" \
    --suite "$SUITE" \
    --set "$SET_NAME" \
    --redshift "$REDSHIFT" \
    --output-dir "$EMBED_DIR" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --codec-device "$CODEC_DEVICE" \
    --num-encoder-tokens "$ENC_TOKENS" \
    --model-dir "$MODEL_DIR" \
    --codec-repo "$CODEC_REPO" \
    --prefix "$PREFIX" \
    "${NORM_ARGS[@]}"
else
  echo "[INFO] SKIP_ENCODING=1 — reusing existing embeddings."
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Step 2: Train Omega_m regression head
# ------------------------------------------------------------------
python scripts/train_omega_head.py \
  --manifest "$MANIFEST_PATH" \
  --shard-dir "$EMBED_DIR" \
  --output-dir "$HEAD_OUT" \
  --hidden-dim ${HIDDEN_DIM:-256} \
  --epochs ${EPOCHS:-50} \
  --batch-size ${HEAD_BATCH_SIZE:-64} \
  --lr ${LR:-4e-3} \
  --device ${HEAD_DEVICE:-cuda} \
  --train-frac ${TRAIN_FRAC:-0.7} \
  --val-frac ${VAL_FRAC:-0.15} \
  --label-standardize
