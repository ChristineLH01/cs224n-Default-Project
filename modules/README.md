# Modules

Core building blocks for the GPT-2 model and parameter-efficient fine-tuning adapters.

## Files

| File | Description |
|---|---|
| `attention.py` | `CausalSelfAttention` — multi-head causal self-attention with optional Flash Attention via `F.scaled_dot_product_attention` |
| `gpt2_layer.py` | `GPT2Layer` — single transformer block (pre-norm architecture): LayerNorm → attention → residual, LayerNorm → FFN → residual |
| `lora.py` | `LoRALinear` — standard LoRA adapter (`W + scaling * A @ B`). Supports rsLoRA scaling (`alpha / sqrt(rank)`) and per-layer rank patterns via `apply_lora_to_model` |
| `dora.py` | `DoRALinear` — weight-decomposed LoRA that separates magnitude and direction: `W' = m * normalize(W + scaling * B @ A)`. Applied via `apply_dora_to_model` |
| `adalora.py` | `AdaLoRALinear` — adaptive LoRA with SVD-style parameterization (`P @ diag(Λ) @ Q`). Includes `AdaLoRARankScheduler` for sensitivity-based importance scoring and cubic budget pruning |

## LoRA (`lora.py`)

`LoRALinear` wraps a frozen `nn.Linear` layer and adds a trainable low-rank update:

```
output = W(x) + scaling * x @ lora_A @ lora_B
```

- `lora_A` — down-projection matrix of shape `(in_features, rank)`
- `lora_B` — up-projection matrix of shape `(rank, out_features)`
- `scaling` — `alpha / rank` (standard) or `alpha / sqrt(rank)` (rsLoRA)

`lora_A` is initialized with Kaiming uniform and `lora_B` is initialized to zeros, so the adapter starts as a no-op. Only `lora_A` and `lora_B` are trainable; the original weight `W` stays frozen.

`apply_lora_to_model` injects `LoRALinear` wrappers into the model based on `fine_tune_mode`:
- `lora_attention` — wraps Q, K, V projections and the attention output dense layer
- `lora_mlp` — wraps the intermediate and output dense layers in the FFN
- `lora` — wraps both attention and MLP

Supports per-layer rank schedules via `--rank_pattern` (e.g. `'4,4,8,8,16,16'`).

## DoRA (`dora.py`)

`DoRALinear` extends LoRA by decomposing the adapted weight into **magnitude** and **direction**:

```
W_adapted = W + scaling * (lora_A @ lora_B)^T
direction = W_adapted / ||W_adapted||_column
output    = (m * direction) @ x + bias
```

- `lora_A`, `lora_B` — same low-rank matrices as LoRA
- `magnitude` (`m`) — learnable per-output-neuron scaling vector of shape `(out_features,)`, initialized to the column-wise norm of the original weight

This decouples *how much* to change (magnitude) from *which direction* to change, which can improve training stability. Trainable parameters: `lora_A`, `lora_B`, and `magnitude`.

`apply_dora_to_model` works the same as `apply_lora_to_model`, controlled by `fine_tune_mode` (`dora`, `dora_attention`, `dora_mlp`).

## AdaLoRA (`adalora.py`)

`AdaLoRALinear` uses an SVD-style parameterization instead of the standard A/B factorization:

```
output = W(x) + (alpha / ranknum) * x @ (Q * Lambda)^T @ P^T
```

- `P` — left singular vectors, shape `(out_features, rank)`
- `Q` — right singular vectors, shape `(rank, in_features)`
- `Lambda` — singular values, shape `(rank, 1)`
- `ranknum` — tracks the number of active (non-pruned) singular values

Trainable parameters: `P`, `Q`, and `Lambda`. The key advantage is that individual singular values can be zeroed out (pruned) during training to adaptively allocate rank where it matters most.

`apply_adalora_to_model` always wraps all attention and MLP linears (no attention-only/MLP-only modes).

### Rank Scheduling

`AdaLoRARankScheduler` dynamically prunes low-importance singular values during training using sensitivity-based importance scoring:

- **Importance score**: `I(w) = |w * grad(L)|`, smoothed via EMA with uncertainty quantification
- **Triplet score**: combines the importance of `Lambda_i`, the corresponding row of `P`, and column of `Q`

Pruning proceeds in three phases:
1. **Initial warmup** — no pruning, all ranks active
2. **Cubic budget decrease** — globally rank all triplets and prune the least important to meet a decreasing budget
3. **Final fine-tuning** — rank pattern is fixed at the target budget

Orthogonal regularization (`compute_orth_regu`) encourages `P` and `Q` to stay near-orthogonal: `R = ||P^T P - I||_F + ||Q Q^T - I||_F`.

## Adapter Injection Summary

| Function | Wraps | Trainable params |
|---|---|---|
| `apply_lora_to_model` | Attention and/or MLP linears (controlled by `fine_tune_mode`) | `lora_A`, `lora_B` |
| `apply_dora_to_model` | Attention and/or MLP linears (controlled by `fine_tune_mode`) | `lora_A`, `lora_B`, `magnitude` |
| `apply_adalora_to_model` | All attention + MLP linears | `P`, `Q`, `Lambda` |
