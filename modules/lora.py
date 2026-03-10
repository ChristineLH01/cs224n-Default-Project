import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
  def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, use_rslora: bool = False):
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
    args: Must have fine_tune_mode ('lora', 'lora_attention', 'lora_mlp'), lora_rank, lora_alpha
          with optional lora_dropout, use_rslora, and rank_pattern (comma-separated per-layer
          ranks, e.g. '4,4,8,8,16,16,32,32,...'). rank_pattern takes precedence over lora_rank.
  """
  mode = args.fine_tune_mode
  lora_attention = mode in ('lora', 'lora_attention')
  lora_mlp = mode in ('lora', 'lora_mlp')
  alpha, dropout = args.lora_alpha, args.lora_dropout
  use_rslora = args.use_rslora

  rank_pattern = getattr(args, 'rank_pattern', None)
  if rank_pattern:
    ranks = [int(r) for r in rank_pattern.split(',')]
    assert len(ranks) == len(gpt_model.gpt_layers), (
      f"rank_pattern has {len(ranks)} entries but model has {len(gpt_model.gpt_layers)} layers"
    )
  else:
    ranks = None

  for i, layer in enumerate(gpt_model.gpt_layers):
    rank = ranks[i] if ranks is not None else args.lora_rank
    if lora_attention:
      layer.self_attention.query = LoRALinear(layer.self_attention.query, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.self_attention.key = LoRALinear(layer.self_attention.key, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.self_attention.value = LoRALinear(layer.self_attention.value, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.attention_dense = LoRALinear(layer.attention_dense, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
    if lora_mlp:
      layer.interm_dense = LoRALinear(layer.interm_dense, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
      layer.out_dense = LoRALinear(layer.out_dense, rank=rank, alpha=alpha, dropout=dropout, use_rslora=use_rslora)

  # Freeze all base parameters, unfreeze only LoRA parameters
  for name, param in gpt_model.named_parameters():
    param.requires_grad = 'lora' in name
