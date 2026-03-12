#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=sonnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00


set -euo pipefail

# ── Conda setup ───────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate cs224n_dfp

# ── Run training ──────────────────────────────────────────────
CUDA_LAUNCH_BLOCKING=1
echo "Node: $(hostname)"
echo "Process ID: $$"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# export job id to environment variable
export JOB_ID=$SLURM_JOB_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python sonnet_generation_justin.py \
  --use_gpu \
  --epochs 15 \
  --batch_size 8 \
  --lr 2e-5 \
  --model_size gpt2-medium \
  --fine_tune_mode lora_attention \
  --lora_rank 8 \
  --lora_alpha 8 \
  --lora_dropout 0.1 \
  --use_flash_attention
