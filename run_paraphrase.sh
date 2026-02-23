#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=search
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=/home/justin/cs224n-Default-Project/logs/%x_%j.out
#SBATCH --account=liquidai

set -euo pipefail

# ── Conda setup ───────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate cs224n_dfp

# ── Run training ──────────────────────────────────────────────
cd /home/justin/cs224n-Default-Project
CUDA_LAUNCH_BLOCKING=1
echo "Node: $(hostname)"
echo "Process ID: $$"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# export job id to environment variable
export JOB_ID=$SLURM_JOB_ID

python paraphrase_detection_justin.py \
  --use_gpu \
  --epochs 1 \
  --batch_size 256 \
  --lr 2e-5 \
  --model_size gpt2-medium
