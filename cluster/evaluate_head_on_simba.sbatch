#!/bin/bash
# Evaluate Illustris-trained head on SIMBA embeddings

#SBATCH --job-name=camels-eval-simba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.2.0


export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/heads/IllustrisTNG_LH/best_model.pt}
MANIFEST=${MANIFEST:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/SIMBA_LH/SIMBA_LH_z0p00_manifest.json}
SHARD_DIR=${SHARD_DIR:-/lustre/fswork/projects/rech/oxl/utl47bv/data/camels_aion/embeddings/SIMBA_LH}
OUTPUT=${OUTPUT:-$WORK/data/camels_aion/evals/simba_cross_eval.json}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${EVAL_DEVICE:-cuda}

mkdir -p "$(dirname "$OUTPUT")" logs/slurm

python scripts/evaluate_head_on_simba.py \
  --model "$MODEL_PATH" \
  --manifest "$MANIFEST" \
  --shard-dir "$SHARD_DIR" \
  --output "$OUTPUT" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE"
