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
    
    if rank > 0:
        if use_rslora:
            self.scaling = alpha / math.sqrt(rank)
        else:
            self.scaling = alpha / rank
    else:
        self.scaling = 0.0

    in_features = linear_layer.weight.shape[1]
    out_features = linear_layer.weight.shape[0]

    if rank > 0:
      self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
      self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))
      self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
      self.reset_parameters()
    
    # Freeze the base linear layer
    self.linear.weight.requires_grad = False
    if self.linear.bias is not None:
      self.linear.bias.requires_grad = False

  def reset_parameters(self):
    if self.rank > 0:
      bound = math.sqrt(1 / self.linear.weight.shape[1])
      nn.init.uniform_(self.lora_A, -bound, bound)
      nn.init.zeros_(self.lora_B)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_out = self.linear(x)
    
    if self.rank == 0:
      return base_out
    
    lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
    lora_out = lora_out * self.scaling
    
    return base_out + lora_out

def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float = 0.0, target_modules=["query", "key", "value"], use_rslora: bool = False):
  if rank == 0:
    return

  for name, module in model.named_children():
    if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
      lora_layer = LoRALinear(module, rank, alpha, dropout, use_rslora=use_rslora)
      setattr(model, name, lora_layer)
    else:
      inject_lora(module, rank, alpha, dropout, target_modules, use_rslora=use_rslora)
