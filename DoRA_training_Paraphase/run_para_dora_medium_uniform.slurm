#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=dora_para_medium_uniform
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=14
#SBATCH --output=/home/justin/cs224n-Default-Project/logs/%x_%j.out
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-008

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate cs224n_dfp

cd /home/justin/cs224n-Default-Project
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

export JOB_ID=$SLURM_JOB_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ══════════════════════════════════════════════════════════════
#  Job 3 / 4 — gpt2-medium, DoRA attn+mlp, uniform rank  (24 runs)
#  fine_tune_mode=dora → paraphrase_exp_results_dora/
#
#  Anchored to LoRA medium best (grid_param.txt):
#    rank=32, alpha_mult=2.0, LR=(5e-4 7e-4 1e-3)
#  Sweep rank=16,32; alpha_mult fixed at 2.0.
#
#  Epochs = 16 (fixed, from grid_param.txt)
# ══════════════════════════════════════════════════════════════

MODEL="gpt2-medium"
LR_VALUES=(5e-4 7e-4 1e-3)
LORA_RANKS=(16 32)
LORA_ALPHA_MULT=2.0
AUG_VALUES=("on" "off")

EPOCHS=16
BATCH_SIZE=64
LORA_DROPOUT=0.1
FLASH="on"

# ══════════════════════════════════════════════════════════════

run_uniform() {
  local lr=$1 rank=$2 aug=$3

  local alpha
  alpha=$(python3 -c "print($rank * $LORA_ALPHA_MULT)")
  local lora_flags="--fine_tune_mode dora --lora_rank $rank --lora_alpha $alpha --lora_dropout $LORA_DROPOUT"

  local flash_flag=""
  [ "$FLASH" = "on" ] && flash_flag="--use_flash_attention"

  local para_train="data/quora-train.csv"
  local aug_label="aug=off"
  [ "$aug" = "on" ] && para_train="data/augmented_train.csv" && aug_label="aug=on"

  echo ""
  echo "======================================================"
  echo "  model=$MODEL | lr=$lr | epochs=$EPOCHS | rank=$rank | alpha=$alpha | mode=dora | $aug_label"
  echo "======================================================"

  torchrun --nproc_per_node=8 paraphrase_detection_justin.py \
    --use_gpu \
    --para_train "$para_train" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$lr" \
    --model_size "$MODEL" \
    $lora_flags \
    $flash_flag
}

# ── Pre-count ─────────────────────────────────────────────────
total=0
for lr in "${LR_VALUES[@]}"; do
  for rank in "${LORA_RANKS[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      ((total += 1))
    done
  done
done

echo "Total runs planned: $total"
echo "Starting: model=$MODEL | DoRA uniform | attn+mlp"

# ── Run grid ──────────────────────────────────────────────────
done_count=0
start_time=$(date +%s)

print_progress() {
  local elapsed=$(( $(date +%s) - start_time ))
  local elapsed_min=$(( elapsed / 60 ))
  local elapsed_sec=$(( elapsed % 60 ))
  local avg_sec=0
  local eta_min="N/A"
  if [ "$done_count" -gt 0 ]; then
    avg_sec=$(( elapsed / done_count ))
    local remaining=$(( (total - done_count) * avg_sec ))
    eta_min=$(( remaining / 60 ))
  fi
  echo ""
  echo "██ Progress: $done_count / $total done | elapsed: ${elapsed_min}m${elapsed_sec}s | avg: ${avg_sec}s/run | ETA: ~${eta_min}min remaining"
}

for lr in "${LR_VALUES[@]}"; do
  for rank in "${LORA_RANKS[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      run_uniform "$lr" "$rank" "$aug"
      ((done_count += 1))
      print_progress
    done
  done
done

echo ""
echo "Grid search complete. $done_count / $total runs finished."
