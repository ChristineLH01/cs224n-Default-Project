#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=embedding_neg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=2:00:00
#SBATCH --output=/home/justin/cs224n-Default-Project/logs/%x_%j.out
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-008

set -euo pipefail

# ── Conda setup ───────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate cs224n_dfp

# ── Run ───────────────────────────────────────────────────────
cd /home/justin/cs224n-Default-Project
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0 2>/dev/null || pip install faiss-cpu --quiet

export JOB_ID=$SLURM_JOB_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

python data_processing/paraphase/embedding_negatives.py
