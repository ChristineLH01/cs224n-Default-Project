#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=cs224n_para
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
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

python paraphrase_detection_justin.py \
  --use_gpu \
  --epochs 3 \
  --batch_size 64 \
  --lr 1e-5 \
  --model_size gpt2-medium
