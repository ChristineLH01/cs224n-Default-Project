#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=paraphrase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=14
#SBATCH --output=/home/justin/cs224n-Default-Project/logs/%x_%j.out
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-008

set -euo pipefail

# ── Conda setup ───────────────────────────────────────────────
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
#  GRID CONFIG — edit these arrays to define the search space
# ══════════════════════════════════════════════════════════════

# Model sizes: "gpt2"  "gpt2-medium"  "gpt2-large"
MODEL_SIZE_VALUES=("gpt2" "gpt2-medium" "gpt2-large")

# Basic hyperparams
LR_VALUES=(1e-4 5e-4 7e-4 1e-3)
EPOCH_VALUES=(16)
BATCH_SIZE_VALUES=(64)

# LoRA: set LORA_RANKS=(0) to skip LoRA entirely,
#       include 0 to also run full fine-tuning alongside LoRA runs
LORA_RANKS=(8 16 32)

# Which layers to apply LoRA to (ignored when rank=0)
# Options: "attn"  "mlp"  "attn_mlp"
LORA_TARGETS=("attn" "attn_mlp" "mlp")

# LoRA alpha = rank * LORA_ALPHA_MULT  (1.0 means alpha == rank)
LORA_ALPHA_MULT=(1.0 2.0)
LORA_DROPOUT=(0.1)

# rsLoRA rank scaling: "on" or "off"
RSLORA_VALUES=("off")

# Apply LoRA to all linear layers at once (overrides attn/mlp target flags): "on" or "off"
LORA_ALL_LINEAR_VALUES=("off")

# Flash attention: "on" "off" or both ("on" "off")
FLASH_VALUES=("on")

# Augmented training data: "on" uses data/augmented_train.csv, "off" uses data/quora-train.csv
AUG_VALUES=("off")

# ══════════════════════════════════════════════════════════════

run() {
  local model=$1
  local lr=$2
  local epochs=$3
  local bs=$4
  local rank=$5
  local target=$6
  local alpha_mult=$7
  local dropout=$8
  local flash=$9
  local aug=${10}
  local rslora=${11}
  local all_linear=${12}

  local lora_flags=""
  if [ "$rank" -gt 0 ]; then
    local alpha
    alpha=$(python3 -c "print($rank * $alpha_mult)")
    lora_flags="--lora_rank $rank --lora_alpha $alpha --lora_dropout $dropout"
    if [ "$all_linear" = "on" ]; then
      lora_flags="$lora_flags --apply_lora_to_all_linear"
    else
      if [ "$target" = "attn" ] || [ "$target" = "attn_mlp" ]; then
        lora_flags="$lora_flags --apply_lora_to_attention"
      fi
      if [ "$target" = "mlp" ] || [ "$target" = "attn_mlp" ]; then
        lora_flags="$lora_flags --apply_lora_to_mlp"
      fi
    fi
    [ "$rslora" = "on" ] && lora_flags="$lora_flags --use_rslora"
  fi

  local flash_flag=""
  [ "$flash" = "on" ] && flash_flag="--use_flash_attention"

  local label="model=$model | lr=$lr | epochs=$epochs | bs=$bs | lora_rank=$rank"
  if [ "$rank" -gt 0 ]; then
    label="$label | target=$target | alpha_mult=$alpha_mult | dropout=$dropout | rslora=$rslora | all_linear=$all_linear"
  fi
  label="$label | flash=$flash | aug=$aug"

  echo ""
  echo "======================================================"
  echo "  $label"
  echo "======================================================"

  local para_train_file="data/quora-train.csv"
  [ "$aug" = "on" ] && para_train_file="data/augmented_train.csv"

  torchrun --nproc_per_node=8 paraphrase_detection_justin.py \
    --use_gpu \
    --para_train "$para_train_file" \
    --epochs "$epochs" \
    --batch_size "$bs" \
    --lr "$lr" \
    --model_size "$model" \
    $lora_flags \
    $flash_flag
}

# ── Pre-count total runs ──────────────────────────────────────
total=0
for model in "${MODEL_SIZE_VALUES[@]}"; do
  for lr in "${LR_VALUES[@]}"; do
    for epochs in "${EPOCH_VALUES[@]}"; do
      for bs in "${BATCH_SIZE_VALUES[@]}"; do
        for rank in "${LORA_RANKS[@]}"; do
          for flash in "${FLASH_VALUES[@]}"; do
            for aug in "${AUG_VALUES[@]}"; do
              if [ "$rank" -eq 0 ]; then
                ((total += 1))
              else
                for target in "${LORA_TARGETS[@]}"; do
                  for alpha_mult in "${LORA_ALPHA_MULT[@]}"; do
                    for dropout in "${LORA_DROPOUT[@]}"; do
                      for rslora in "${RSLORA_VALUES[@]}"; do
                        for all_linear in "${LORA_ALL_LINEAR_VALUES[@]}"; do
                          ((total += 1))
                        done
                      done
                    done
                  done
                done
              fi
            done
          done
        done
      done
    done
  done
done
echo "Total runs planned: $total"
echo "Starting grid search..."

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

for model in "${MODEL_SIZE_VALUES[@]}"; do
  for lr in "${LR_VALUES[@]}"; do
    for epochs in "${EPOCH_VALUES[@]}"; do
      for bs in "${BATCH_SIZE_VALUES[@]}"; do
        for rank in "${LORA_RANKS[@]}"; do
          for flash in "${FLASH_VALUES[@]}"; do
            for aug in "${AUG_VALUES[@]}"; do
              if [ "$rank" -eq 0 ]; then
                run "$model" "$lr" "$epochs" "$bs" 0 "none" "1.0" "0.0" "$flash" "$aug" "off" "off"
                ((done_count += 1))
                print_progress
              else
                for target in "${LORA_TARGETS[@]}"; do
                  for alpha_mult in "${LORA_ALPHA_MULT[@]}"; do
                    for dropout in "${LORA_DROPOUT[@]}"; do
                      for rslora in "${RSLORA_VALUES[@]}"; do
                        for all_linear in "${LORA_ALL_LINEAR_VALUES[@]}"; do
                          run "$model" "$lr" "$epochs" "$bs" "$rank" "$target" "$alpha_mult" "$dropout" "$flash" "$aug" "$rslora" "$all_linear"
                          ((done_count += 1))
                          print_progress
                        done
                      done
                    done
                  done
                done
              fi
            done
          done
        done
      done
    done
  done
done

echo ""
echo "Grid search complete. $done_count / $total runs finished."
