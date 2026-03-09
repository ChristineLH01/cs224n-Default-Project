#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=dora_sonnet_gpt2_mode
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
#  Job 2 / 4 — gpt2, DoRA target-mode sweep  (24 runs)
#  fine_tune_mode ∈ {dora, dora_attention, dora_mlp}
#  → sonnet_exp_results_dora/
#
#  LoRA sonnet best: rank=32, alpha_mult=2.0, lr=1e-4 → chrf=42.80
#  Sweep top-2 LRs (1e-4, 5e-4) × 3 modes × aug × rank=32.
#
#  aug=off → epochs=20, aug=on → epochs=40
# ══════════════════════════════════════════════════════════════

MODEL="gpt2"
LR_VALUES=(1e-4 5e-4)
DORA_MODES=("dora" "dora_attention" "dora_mlp")
LORA_RANK=32
LORA_ALPHA_MULT=2.0
AUG_VALUES=("on" "off")

BATCH_SIZE=4
TEMPERATURE=1.2
TOP_P=0.9
LORA_DROPOUT=0.1
FLASH="on"

# ══════════════════════════════════════════════════════════════

run_mode() {
  local lr=$1 epochs=$2 mode=$3 aug=$4

  local alpha
  alpha=$(python3 -c "print($LORA_RANK * $LORA_ALPHA_MULT)")
  local lora_flags="--fine_tune_mode $mode --lora_rank $LORA_RANK --lora_alpha $alpha --lora_dropout $LORA_DROPOUT"

  local flash_flag=""
  [ "$FLASH" = "on" ] && flash_flag="--use_flash_attention"

  local sonnet_path="data/sonnets.txt"
  local aug_label="aug=off"
  [ "$aug" = "on" ] && sonnet_path="data/sonnets_merged.txt" && aug_label="aug=on"

  echo ""
  echo "======================================================"
  echo "  model=$MODEL | lr=$lr | epochs=$epochs | rank=$LORA_RANK | alpha=$alpha | mode=$mode | $aug_label"
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
  for mode in "${DORA_MODES[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      ((total += 1))
    done
  done
done

echo "Total runs planned: $total"
echo "Starting: model=$MODEL | DoRA mode sweep | rank=$LORA_RANK"

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
  for mode in "${DORA_MODES[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      epochs=20
      [ "$aug" = "on" ] && epochs=40
      run_mode "$lr" "$epochs" "$mode" "$aug"
      ((done_count += 1))
      print_progress
    done
  done
done

echo ""
echo "Grid search complete. $done_count / $total runs finished."
