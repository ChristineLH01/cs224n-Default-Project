#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=adalora_para_medium_sched
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
#  Job 4 / 4 — gpt2-medium, AdaLoRA schedule sweep  (8 runs)
#  fine_tune_mode=adalora → paraphrase_exp_results_adalora/
#
#  Rank pair fixed at (16, 8) — sweep pruning schedule params.
#  tinit_ratio: warmup fraction before pruning begins.
#  delta_t: prune every N optimizer steps.
#
#  LoRA medium best: lr=1e-3, acc=0.9092
#  LR=(7e-4 1e-3), alpha=8.0 (fixed; 16.0 dropped — consistently worse)
#  tinit=0.2 (fixed; 0.1 dropped — 0.2 dominates top results)
#  delta_t=(10 50) — kept to verify for paraphrase task
#  Epochs = 16 (fixed, from grid_param.txt)
# ══════════════════════════════════════════════════════════════

MODEL="gpt2-medium"
LR_VALUES=(7e-4 1e-3)
ADALORA_TINIT=0.2
ADALORA_DELTA_T_VALUES=(10 50)
ADALORA_ALPHA=8.0
AUG_VALUES=("on" "off")

ADALORA_INIT_R=16
ADALORA_TARGET_R=8

EPOCHS=16
BATCH_SIZE=64
LORA_DROPOUT=0.1
FLASH="on"

# ══════════════════════════════════════════════════════════════

run_sched() {
  local lr=$1 delta_t=$2 aug=$3

  local lora_flags="--fine_tune_mode adalora"
  lora_flags="$lora_flags --adalora_init_r $ADALORA_INIT_R --adalora_target_r $ADALORA_TARGET_R"
  lora_flags="$lora_flags --adalora_alpha $ADALORA_ALPHA --lora_dropout $LORA_DROPOUT"
  lora_flags="$lora_flags --adalora_tinit_ratio $ADALORA_TINIT --adalora_delta_t $delta_t"

  local flash_flag=""
  [ "$FLASH" = "on" ] && flash_flag="--use_flash_attention"

  local para_train="data/quora-train.csv"
  local aug_label="aug=off"
  [ "$aug" = "on" ] && para_train="data/augmented_train.csv" && aug_label="aug=on"

  echo ""
  echo "======================================================"
  echo "  model=$MODEL | lr=$lr | epochs=$EPOCHS | init_r=$ADALORA_INIT_R | target_r=$ADALORA_TARGET_R | alpha=$ADALORA_ALPHA | tinit=$ADALORA_TINIT | delta_t=$delta_t | $aug_label"
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
  for delta_t in "${ADALORA_DELTA_T_VALUES[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      ((total += 1))
    done
  done
done

echo "Total runs planned: $total"
echo "Starting: model=$MODEL | AdaLoRA schedule sweep | init_r=$ADALORA_INIT_R | target_r=$ADALORA_TARGET_R | alpha=$ADALORA_ALPHA | tinit=$ADALORA_TINIT"

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
  for delta_t in "${ADALORA_DELTA_T_VALUES[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      run_sched "$lr" "$delta_t" "$aug"
      ((done_count += 1))
      print_progress
    done
  done
done

echo ""
echo "Grid search complete. $done_count / $total runs finished."
