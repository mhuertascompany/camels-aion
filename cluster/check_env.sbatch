#!/bin/bash
# Train multimodal LLaMA (multi spectral tokens) on Jean Zay 2x V100 32GB setup

#SBATCH --job-name=tl-llm-multitok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # 2 ranks = 2 GPUs
#SBATCH --gres=gpu:1                  # request 2 GPUs on the node
#SBATCH --cpus-per-task=30            # tune for your dataloader/tokenization
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00               # adjust as needed
# Ensure 32 GB V100s (Jean Zay)
#SBATCH -C v100-32g
# Logs
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail



# Load environment (adjust to your site)
module purge
module load pytorch-gpu/py3/2.2.0

# Run environment check
mkdir -p logs
srun python scripts/check_environment.py --device cuda | tee logs/check_environment.log
