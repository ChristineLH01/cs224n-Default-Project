#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=adalora_sonnet_gpt2_sched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
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
#  Job 2 / 4 — gpt2, AdaLoRA schedule sweep  (32 runs)
#  fine_tune_mode=adalora → sonnet_exp_results_adalora/
#
#  Rank pair fixed at (16, 4) — sweep pruning schedule params.
#  Sonnet has small data → larger tinit_ratio (0.2, 0.3) needed
#  to let model stabilize before pruning.
#
#  LR=(1e-4 5e-4), alpha=(8.0 16.0)
#  aug=off → epochs=20, aug=on → epochs=40
# ══════════════════════════════════════════════════════════════

MODEL="gpt2"
LR_VALUES=(1e-4 5e-4)
ADALORA_TINIT_VALUES=(0.2 0.3)
ADALORA_DELTA_T_VALUES=(10 50)
ADALORA_ALPHA_VALUES=(8.0 16.0)
AUG_VALUES=("on" "off")

ADALORA_INIT_R=16
ADALORA_TARGET_R=4

BATCH_SIZE=4
TEMPERATURE=1.2
TOP_P=0.9
LORA_DROPOUT=0.1
FLASH="on"

# ══════════════════════════════════════════════════════════════

run_sched() {
  local lr=$1 epochs=$2 tinit=$3 delta_t=$4 alpha=$5 aug=$6

  local lora_flags="--fine_tune_mode adalora"
  lora_flags="$lora_flags --adalora_init_r $ADALORA_INIT_R --adalora_target_r $ADALORA_TARGET_R"
  lora_flags="$lora_flags --adalora_alpha $alpha --lora_dropout $LORA_DROPOUT"
  lora_flags="$lora_flags --adalora_tinit_ratio $tinit --adalora_delta_t $delta_t"

  local flash_flag=""
  [ "$FLASH" = "on" ] && flash_flag="--use_flash_attention"

  local sonnet_path="data/sonnets.txt"
  local aug_label="aug=off"
  [ "$aug" = "on" ] && sonnet_path="data/sonnets_merged.txt" && aug_label="aug=on"

  echo ""
  echo "======================================================"
  echo "  model=$MODEL | lr=$lr | epochs=$epochs | init_r=$ADALORA_INIT_R | target_r=$ADALORA_TARGET_R | alpha=$alpha | tinit=$tinit | delta_t=$delta_t | $aug_label"
  echo "======================================================"

  python sonnet_generation_justin.py \
    --use_gpu \
    --sonnet_path "$sonnet_path" \
    --epochs "$epochs" \
    --batch_size "$BATCH_SIZE" \
    --lr "$lr" \
    --model_size "$MODEL" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    $lora_flags \
    $flash_flag
}

# ── Pre-count ─────────────────────────────────────────────────
total=0
for lr in "${LR_VALUES[@]}"; do
  for tinit in "${ADALORA_TINIT_VALUES[@]}"; do
    for delta_t in "${ADALORA_DELTA_T_VALUES[@]}"; do
      for alpha in "${ADALORA_ALPHA_VALUES[@]}"; do
        for aug in "${AUG_VALUES[@]}"; do
          ((total += 1))
        done
      done
    done
  done
done

echo "Total runs planned: $total"
echo "Starting: model=$MODEL | AdaLoRA schedule sweep | init_r=$ADALORA_INIT_R | target_r=$ADALORA_TARGET_R"

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
  for tinit in "${ADALORA_TINIT_VALUES[@]}"; do
    for delta_t in "${ADALORA_DELTA_T_VALUES[@]}"; do
      for alpha in "${ADALORA_ALPHA_VALUES[@]}"; do
        for aug in "${AUG_VALUES[@]}"; do
          epochs=20
          [ "$aug" = "on" ] && epochs=40
          run_sched "$lr" "$epochs" "$tinit" "$delta_t" "$alpha" "$aug"
          ((done_count += 1))
          print_progress
        done
      done
    done
  done
done

echo ""
echo "Grid search complete. $done_count / $total runs finished."
