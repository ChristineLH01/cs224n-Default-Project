#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=adalora_sonnet_gpt2_uniform
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
#  Job 1 / 4 — gpt2, AdaLoRA rank-pair sweep  (48 runs)
#  fine_tune_mode=adalora → sonnet_exp_results_adalora/
#
#  Sweeps init_r/target_r pairs.
#  Schedule fixed: tinit_ratio=0.2, delta_t=10.
#
#  Anchored to LoRA best (grid_param.txt):
#    LR=(1e-4 5e-4 7e-4 1e-3), alpha=8.0
#  Rank pairs: (8,4) (16,4) (16,8)
#
#  aug=off → epochs=20, aug=on → epochs=40 (from grid_param.txt)
# ══════════════════════════════════════════════════════════════

MODEL="gpt2"
LR_VALUES=(1e-4 5e-4 7e-4 1e-3)
# (init_r target_r) pairs  — init_r must be > target_r
INIT_R_VALUES=(8  16 16)
TARGET_R_VALUES=(4  4  8)
ADALORA_ALPHA=8.0
AUG_VALUES=("on" "off")

ADALORA_TINIT_RATIO=0.2
ADALORA_DELTA_T=10

BATCH_SIZE=4
TEMPERATURE=1.2
TOP_P=0.9
LORA_DROPOUT=0.1
FLASH="on"

# ══════════════════════════════════════════════════════════════

run_uniform() {
  local lr=$1 epochs=$2 init_r=$3 target_r=$4 aug=$5

  local lora_flags="--fine_tune_mode adalora"
  lora_flags="$lora_flags --adalora_init_r $init_r --adalora_target_r $target_r"
  lora_flags="$lora_flags --adalora_alpha $ADALORA_ALPHA --lora_dropout $LORA_DROPOUT"
  lora_flags="$lora_flags --adalora_tinit_ratio $ADALORA_TINIT_RATIO --adalora_delta_t $ADALORA_DELTA_T"

  local flash_flag=""
  [ "$FLASH" = "on" ] && flash_flag="--use_flash_attention"

  local sonnet_path="data/sonnets.txt"
  local aug_label="aug=off"
  [ "$aug" = "on" ] && sonnet_path="data/sonnets_merged.txt" && aug_label="aug=on"

  echo ""
  echo "======================================================"
  echo "  model=$MODEL | lr=$lr | epochs=$epochs | init_r=$init_r | target_r=$target_r | alpha=$ADALORA_ALPHA | $aug_label"
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
num_pairs=${#INIT_R_VALUES[@]}
total=0
for lr in "${LR_VALUES[@]}"; do
  for (( i=0; i<num_pairs; i++ )); do
    for aug in "${AUG_VALUES[@]}"; do
      ((total += 1))
    done
  done
done

echo "Total runs planned: $total"
echo "Starting: model=$MODEL | AdaLoRA rank-pair sweep | schedule fixed (tinit=$ADALORA_TINIT_RATIO, delta_t=$ADALORA_DELTA_T)"

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
  for (( i=0; i<num_pairs; i++ )); do
    for aug in "${AUG_VALUES[@]}"; do
      epochs=20
      [ "$aug" = "on" ] && epochs=40
      run_uniform "$lr" "$epochs" "${INIT_R_VALUES[$i]}" "${TARGET_R_VALUES[$i]}" "$aug"
      ((done_count += 1))
      print_progress
    done
  done
done

echo ""
echo "Grid search complete. $done_count / $total runs finished."
