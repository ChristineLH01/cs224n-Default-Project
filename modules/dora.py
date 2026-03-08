import torch
import torch.nn as nn
import math


class DoRALinear(nn.Module):
  """
  DoRA: Decomposes the weight update into magnitude and direction:
    W' = m * (W + scaling * B @ A) / || W + scaling * B @ A ||_c

    - m is a learnable magnitude vector of shape (out_features,)
    - A is the low-rank down-projection (in_features, rank)
    - B is the low-rank up-projection (rank, out_features)
    - ||...||_c is the column-wise L2 norm (norm over input dimension per output neuron)
    - scaling = alpha / rank (or alpha / sqrt(rank) for rsLoRA)

  This decouples "how much" to change (magnitude) from "which direction" to change
  """

  def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float,
               dropout: float = 0.0, use_rslora: bool = False):
    super().__init__()
    self.linear = linear_layer
    self.rank = rank
    self.alpha = alpha
    self.use_rslora = use_rslora

    if use_rslora:
      self.scaling = alpha / math.sqrt(rank)
    else:
      self.scaling = alpha / rank

    in_features = linear_layer.weight.shape[1]
    out_features = linear_layer.weight.shape[0]

    self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
    self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    # Learnable magnitude vector
    # m_init = || W ||_c  (column-wise norm, i.e. norm over in_features for each out_feature)
    with torch.no_grad():
      weight_norm = linear_layer.weight.norm(dim=1)  # (out_features,)
    self.magnitude = nn.Parameter(weight_norm.clone())

    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # LoRA delta: delta_W = scaling * B^T @ A^T  ->  (out, in)

    # Step 1: Compute the adapted weight direction
    # W_adapted = W + scaling * (A @ B)^T = W + scaling * B^T @ A^T
    # For column-wise norm, we need the full weight matrix.
    lora_weight = self.scaling * (self.lora_A @ self.lora_B).T  # (out_features, in_features)
    adapted_weight = self.linear.weight + lora_weight

    # Step 2: Column-wise normalization (norm over in_features dim for each output neuron)
    weight_norm = adapted_weight.norm(dim=1, keepdim=True)  # (out_features, 1)
    weight_norm = weight_norm.clamp(min=1e-8)  # Avoid division by zero
    direction = adapted_weight / weight_norm  # Normalized direction

    # Step 3: Apply learnable magnitude
    # W' = diag(m) @ direction
    final_weight = self.magnitude.unsqueeze(1) * direction  # (out_features, in_features)

    # Step 4: Apply dropout to input (consistent with LoRA)
    x_drop = self.dropout(x)

    # Step 5: Compute output using the decomposed weight
    out = x_drop @ final_weight.T
    if self.linear.bias is not None:
      out = out + self.linear.bias

    return out


def apply_dora_to_model(gpt_model, args):
  mode = args.fine_tune_mode
  dora_attention = mode in ('dora', 'dora_attention')
  dora_mlp = mode in ('dora', 'dora_mlp')

  rank = args.lora_rank
  alpha = args.lora_alpha
  dropout = args.lora_dropout
  use_rslora = getattr(args, 'use_rslora', False)

  for layer in gpt_model.gpt_layers:
    if dora_attention:
      layer.self_attention.query = DoRALinear(
        layer.self_attention.query, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.self_attention.key = DoRALinear(
        layer.self_attention.key, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.self_attention.value = DoRALinear(
        layer.self_attention.value, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.attention_dense = DoRALinear(
        layer.attention_dense, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
    if dora_mlp:
      layer.interm_dense = DoRALinear(
        layer.interm_dense, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.out_dense = DoRALinear(
        layer.out_dense, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)

  # Freeze all base parameters, unfreeze only DoRA parameters (lora_A, lora_B, magnitude)
  for name, param in gpt_model.named_parameters():
    param.requires_grad = any(k in name for k in ('lora_A', 'lora_B', 'magnitude'))
