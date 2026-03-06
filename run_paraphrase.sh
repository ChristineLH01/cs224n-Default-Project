#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=paraphrase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=/home/justin/cs224n-Default-Project/logs/%x_%j.out
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-008


set -euo pipefail

# ── Conda setup ───────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate cs224n_dfp

# ── Run training ──────────────────────────────────────────────
cd /home/justin/cs224n-Default-Project
echo "Node: $(hostname)"
echo "Process ID: $$"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# export job id to environment variable
export JOB_ID=$SLURM_JOB_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 paraphrase_detection_justin.py \
  --use_gpu \
  --epochs 5 \
  --batch_size 64 \
  --lr 7e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --apply_lora_to_mlp \
  --model_size gpt2-medium \

  #--use_flash_attention 