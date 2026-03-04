#!/bin/bash
# Colab Experiment Runner for LoRA Extensions

# Exit on error
set -e

# 1. Install required packages
echo "Installing dependencies..."
pip install torch transformers einops tqdm sacrebleu scikit-learn

echo "Dependencies installed."

# 2. Define hyperparameter constants (matching your LaTeX report)
PARA_EPOCHS=5
PARA_BATCH_SIZE=64
PARA_LR="7e-05"
PARA_MODEL="gpt2-medium"

SONNET_EPOCHS=25
SONNET_BATCH_SIZE=8
SONNET_LR="2e-05"
SONNET_MODEL="gpt2-medium"

echo "=================================================="
echo "Starting Paraphrase Detection Experiments..."
echo "=================================================="

# Experiment 1: rsLoRA (Rank Sweep)
for r in 8 16 32; do
    echo "Running Paraphrase: rsLoRA (rank=$r)"
    python paraphrase_detection_justin.py \
        --model_size $PARA_MODEL --epochs $PARA_EPOCHS --batch_size $PARA_BATCH_SIZE --lr $PARA_LR \
        --lora_rank $r --lora_alpha 32 --apply_lora_to_attention --use_rslora
done

# Experiment 2: All-Linear Targeting
echo "Running Paraphrase: All-Linear Targeting (rank=8)"
python paraphrase_detection_justin.py \
    --model_size $PARA_MODEL --epochs $PARA_EPOCHS --batch_size $PARA_BATCH_SIZE --lr $PARA_LR \
    --lora_rank 8 --lora_alpha 32 --apply_lora_to_all_linear

# Experiment 3: Layer-wise Rank Allocation (Top-heavy: 4,4,4,4,4,4,4,4, 8,8,8,8,8,8,8,8, 16,16,16,16,16,16,16,16 for 24 layers of Medium)
echo "Running Paraphrase: Layer-wise Rank Allocation"
python paraphrase_detection_justin.py \
    --model_size $PARA_MODEL --epochs $PARA_EPOCHS --batch_size $PARA_BATCH_SIZE --lr $PARA_LR \
    --lora_alpha 32 --apply_lora_to_attention \
    --rank_pattern "4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16"

echo "=================================================="
echo "Starting Sonnet Generation Experiments..."
echo "=================================================="

# Experiment 1: rsLoRA (Rank Sweep)
for r in 8 16 32; do
    echo "Running Sonnet: rsLoRA (rank=$r)"
    python sonnet_generation.py \
        --model_size $SONNET_MODEL --epochs $SONNET_EPOCHS --batch_size $SONNET_BATCH_SIZE --lr $SONNET_LR \
        --lora_rank $r --lora_alpha 16 --apply_lora_to_attention --use_rslora
done

# Experiment 2: All-Linear Targeting
echo "Running Sonnet: All-Linear Targeting (rank=8)"
python sonnet_generation.py \
    --model_size $SONNET_MODEL --epochs $SONNET_EPOCHS --batch_size $SONNET_BATCH_SIZE --lr $SONNET_LR \
    --lora_rank 8 --lora_alpha 16 --apply_lora_to_all_linear

# Experiment 3: Layer-wise Rank Allocation
echo "Running Sonnet: Layer-wise Rank Allocation"
python sonnet_generation.py \
    --model_size $SONNET_MODEL --epochs $SONNET_EPOCHS --batch_size $SONNET_BATCH_SIZE --lr $SONNET_LR \
    --lora_alpha 16 --apply_lora_to_attention \
    --rank_pattern "4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16"

echo "All experiments completed!"
