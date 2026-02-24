import torch
import torch.nn as nn
import math
from modules.attention import CausalSelfAttention

class LoRALinear(nn.Module):
  def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
    super().__init__()
    self.linear = linear_layer
    self.rank = rank
    self.alpha = alpha
    self.scaling = alpha / rank

    in_features = linear_layer.weight.shape[1]
    out_features = linear_layer.weight.shape[0]

    self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
    self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))
    self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
    
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_out = self.linear(x)
    lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
    lora_out = lora_out * self.scaling
    return base_out + lora_out

def apply_lora_to_model(gpt_model, args):
  """
  Inject LoRA adapters when using a LoRA fine-tune mode.

  Args:
    gpt_model.
    args: Must have fine_tune_mode ('lora', 'lora_attention', 'lora_mlp'),
          lora_rank, lora_alpha, lora_dropout.
  """
  mode = args.fine_tune_mode
  lora_attention = mode in ('lora', 'lora_attention')
  lora_mlp = mode in ('lora', 'lora_mlp')
  rank, alpha, dropout = args.lora_rank, args.lora_alpha, args.lora_dropout

  for layer in gpt_model.gpt_layers:
    if lora_attention:
      layer.self_attention.query = LoRALinear(layer.self_attention.query, rank=rank, alpha=alpha, dropout=dropout)
      layer.self_attention.key = LoRALinear(layer.self_attention.key, rank=rank, alpha=alpha, dropout=dropout)
      layer.self_attention.value = LoRALinear(layer.self_attention.value, rank=rank, alpha=alpha, dropout=dropout)
    if lora_mlp:
      layer.attention_dense = LoRALinear(layer.attention_dense, rank=rank, alpha=alpha, dropout=dropout)
      layer.interm_dense = LoRALinear(layer.interm_dense, rank=rank, alpha=alpha, dropout=dropout)
      layer.out_dense = LoRALinear(layer.out_dense, rank=rank, alpha=alpha, dropout=dropout)

  # Freeze all base parameters, unfreeze only LoRA parameters
  for param in gpt_model.parameters():
    param.requires_grad = False
  for name, param in gpt_model.named_parameters():
    if 'lora' in name:
      param.requires_grad = True
