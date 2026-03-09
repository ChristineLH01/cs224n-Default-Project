#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=lora_medium_pattern
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
#  Job 4 / 4 — gpt2-medium, rank patterns  (24 runs)
#  Sweeps both regular LoRA and rsLoRA for each pattern.
#
#  gpt2-medium = 24 layers; patterns are ascending (early layers
#  get lower rank, later semantic layers get higher rank).
#
#  Alpha-mult per scaling mode:
#    regular LoRA: 2.0  (best from grid_param.txt)
#    rsLoRA:       1.0  (adjusted for alpha/sqrt(rank) scaling)
#  alpha = max_rank_in_pattern * alpha_mult
# ══════════════════════════════════════════════════════════════

MODEL="gpt2-medium"
RANK_PATTERNS=(
  "4,4,4,4,4,4,4,4,8,8,8,8,16,16,16,16,16,16,32,32,32,32,32,32"     # gentle ascent
  "8,8,8,8,8,8,8,8,8,8,8,8,16,16,16,16,16,16,32,32,32,32,32,32"     # two-step ascent
  "0,4,0,4,0,4,0,4,8,8,8,8,16,16,16,16,16,16,32,32,32,32,32,32"     # skip early layers
)
LR_VALUES=(5e-4 7e-4)
AUG_VALUES=("on" "off")
RSLORA_VALUES=("off" "on")   # regular LoRA first, then rsLoRA

# alpha_mult per scaling mode (regular LoRA best from grid_param.txt = 2.0)
ALPHA_MULT_REGULAR=2.0
ALPHA_MULT_RSLORA=1.0

BATCH_SIZE=4
TEMPERATURE=1.2
TOP_P=0.9
LORA_DROPOUT=0.1
FLASH="on"

# ══════════════════════════════════════════════════════════════

run_pattern() {
  local lr=$1 epochs=$2 pattern=$3 aug=$4 rslora=$5

  local alpha_mult=$ALPHA_MULT_REGULAR
  [ "$rslora" = "on" ] && alpha_mult=$ALPHA_MULT_RSLORA

  local alpha
  alpha=$(python3 -c "
ranks = [int(r) for r in '$pattern'.split(',') if int(r) > 0]
max_r = max(ranks) if ranks else 1
print(max_r * $alpha_mult)
")
  local lora_flags="--rank_pattern $pattern --lora_alpha $alpha --lora_dropout $LORA_DROPOUT"
  lora_flags="$lora_flags --apply_lora_to_all_linear"
  [ "$rslora" = "on" ] && lora_flags="$lora_flags --use_rslora"

  local flash_flag=""
  [ "$FLASH" = "on" ] && flash_flag="--use_flash_attention"

  local sonnet_path="data/sonnets.txt"
  local aug_label="aug=off"
  [ "$aug" = "on" ] && sonnet_path="data/sonnets_merged.txt" && aug_label="aug=on"

  local rslora_label="rslora=off"
  [ "$rslora" = "on" ] && rslora_label="rslora=on"

  echo ""
  echo "======================================================"
  echo "  model=$MODEL | lr=$lr | epochs=$epochs | pattern=$pattern | alpha=$alpha | $rslora_label | $aug_label"
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
for pattern in "${RANK_PATTERNS[@]}"; do
  for lr in "${LR_VALUES[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      for rslora in "${RSLORA_VALUES[@]}"; do
        ((total += 1))
      done
    done
  done
done

echo "Total runs planned: $total"
echo "Starting: model=$MODEL | rank patterns | regular LoRA + rsLoRA | all_linear=on"

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

for pattern in "${RANK_PATTERNS[@]}"; do
  for lr in "${LR_VALUES[@]}"; do
    for aug in "${AUG_VALUES[@]}"; do
      for rslora in "${RSLORA_VALUES[@]}"; do
        epochs=20
        [ "$aug" = "on" ] && epochs=40
        run_pattern "$lr" "$epochs" "$pattern" "$aug" "$rslora"
        ((done_count += 1))
        print_progress
      done
    done
  done
done

echo ""
echo "Grid search complete. $done_count / $total runs finished."
