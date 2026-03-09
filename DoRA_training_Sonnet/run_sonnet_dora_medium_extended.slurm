#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=search
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
#  Extended training — gpt2-medium, DoRA, aug=on, 55 epochs  (2 runs)
#
#  Both runs peaked late and still had upward chrF trend at ep 40:
#    Run 1 (line 55): dora,     lr=1e-3,  rank=32 → best chrF=47.51 @ep34, last5Δ=+1.38
#    Run 2 (line 48): dora_mlp, lr=5e-4,  rank=32 → best chrF=44.47 @ep34, last5Δ=+0.98
# ══════════════════════════════════════════════════════════════

MODEL="gpt2-medium"
LORA_RANK=32
LORA_ALPHA=64.0
LORA_DROPOUT=0.1
EPOCHS=55
BATCH_SIZE=4
TEMPERATURE=1.2
TOP_P=0.9
FLASH="on"
SONNET_PATH="data/sonnets_merged.txt"

# ══════════════════════════════════════════════════════════════

run_dora() {
  local mode=$1 lr=$2

  local lora_flags="--fine_tune_mode $mode --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"

  local flash_flag=""
  [ "$FLASH" = "on" ] && flash_flag="--use_flash_attention"

  echo ""
  echo "======================================================"
  echo "  model=$MODEL | mode=$mode | lr=$lr | epochs=$EPOCHS | rank=$LORA_RANK | alpha=$LORA_ALPHA | aug=on"
  echo "======================================================"

  python sonnet_generation_justin.py \
    --use_gpu \
    --sonnet_path "$SONNET_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$lr" \
    --model_size "$MODEL" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    $lora_flags \
    $flash_flag
}

# ── Run grid ──────────────────────────────────────────────────
total=2
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

echo "Total runs planned: $total"
echo "Starting: model=$MODEL | DoRA extended training | epochs=$EPOCHS | aug=on"

# Run 1 (line 55): dora, lr=1e-3, rank=32
run_dora "dora" "1e-3"
((done_count += 1))
print_progress

# Run 2 (line 48): dora_mlp, lr=5e-4, rank=32
run_dora "dora_mlp" "5e-4"
((done_count += 1))
print_progress

echo ""
echo "Extended training complete. $done_count / $total runs finished."
