import torch
import torch.nn as nn
import math
from typing import List

class AdaLoRALinear(nn.Module):
  """
  Three learnable components:
    lora_P  : (out_features, rank) 
    lora_Lambda : (rank,)        
    lora_Q  : (rank, in_features) 

  Forward:  y = Wx + scaling * x @ Q^T @ diag(Λ) @ P^T (SVD form)

  An importance score S of shape (rank,) is maintained to decide which
  triplets to prune.  Pruning is done externally by the rank scheduler.
  """

  def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
    super().__init__()
    self.linear = linear_layer
    self.rank = rank           # initial (maximum) rank
    self.alpha = alpha

    self.scaling = alpha / rank

    in_features = linear_layer.weight.shape[1]
    out_features = linear_layer.weight.shape[0]

    # SVD triplet
    self.P = nn.Parameter(torch.zeros(out_features, rank))       
    self.Lambda = nn.Parameter(torch.ones(rank))            
    self.Q = nn.Parameter(torch.zeros(rank, in_features))    

    # Importance score per triplet (updated externally)
    self.register_buffer('importance_score', torch.zeros(rank))
    # Mask: 1 = active, 0 = pruned
    self.register_buffer('rank_mask', torch.ones(rank))

    self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))
    nn.init.zeros_(self.Lambda)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_out = self.linear(x)

    # Apply mask to singular values (pruned triplets contribute nothing)
    masked_lambda = self.Lambda * self.rank_mask           # (rank,)

    x_drop = self.dropout(x)
    out = x_drop @ self.Q.T
    out = out * masked_lambda.unsqueeze(0) if out.dim() == 2 else out * masked_lambda
    out = out @ self.P.T
    return base_out + out * self.scaling

  @torch.no_grad()
  def compute_importance(self):
    """
    Compute sensitivity-based importance for each triplet.
    Importance(i) = |Λ_i| * || P[:,i] ||_F * || Q[i,:] ||_F
    This is a proxy for the contribution of each rank-1 component.
    """
    p_norm = self.P.norm(dim=0)         # (rank,)
    q_norm = self.Q.norm(dim=1)         # (rank,)
    self.importance_score = (self.Lambda.abs() * p_norm * q_norm)

  @property
  def current_rank(self):
    return int(self.rank_mask.sum().item())


def apply_adalora_to_model(gpt_model, args):
  """
  Inject AdaLoRA adapters into ALL linear layers of the GPT model.

  Args:
    gpt_model: The GPT2Model instance.
    args: Must have adalora_rank, adalora_alpha with optional adalora_dropout, use_rslora.
  """
  rank = args.adalora_init_r
  alpha = args.adalora_alpha
  dropout = args.lora_dropout 
 
  for layer in gpt_model.gpt_layers:
    # Attention 
    layer.self_attention.query = AdaLoRALinear(layer.self_attention.query, rank=rank, alpha=alpha, dropout=dropout)
    layer.self_attention.key = AdaLoRALinear(layer.self_attention.key, rank=rank, alpha=alpha, dropout=dropout)
    layer.self_attention.value = AdaLoRALinear(layer.self_attention.value, rank=rank, alpha=alpha, dropout=dropout)
    layer.attention_dense = AdaLoRALinear(layer.attention_dense, rank=rank, alpha=alpha, dropout=dropout)
    # MLP 
    layer.interm_dense = AdaLoRALinear(layer.interm_dense, rank=rank, alpha=alpha, dropout=dropout)
    layer.out_dense = AdaLoRALinear(layer.out_dense, rank=rank, alpha=alpha, dropout=dropout)

  # Freeze all base parameters, unfreeze only AdaLoRA parameters
  for name, param in gpt_model.named_parameters():
    param.requires_grad = any(k in name for k in ('.P', '.Q', '.Lambda'))


def get_adalora_modules(model) -> List[AdaLoRALinear]:
  """Collect all AdaLoRALinear modules from a model."""
  return [m for m in model.modules() if isinstance(m, AdaLoRALinear)]


class AdaLoRARankScheduler:
  """
  Manages the global rank budget and prunes the least important triplets
  over the course of training.

  Algorithm:
    1. Start with total_budget = initial_rank * num_adalora_modules.
    2. Linearly decrease the budget from total_budget to target_budget
       over [warmup_steps, total_steps].
    3. At each pruning step (every prune_every steps), compute importance
       scores for every AdaLoRA module, globally rank all triplets, and
       mask out the ones with the lowest importance.
  """

  def __init__(
    self,
    model,
    init_r: int,
    target_r: int,
    total_steps: int,
    tinit_ratio: float = 0.1,
    delta_t: int = 50,
  ):
    self.adalora_modules: List[AdaLoRALinear] = get_adalora_modules(model)
    self.num_modules = len(self.adalora_modules)

    self.initial_rank = init_r
    self.target_rank = target_r
    self.total_steps = total_steps
    self.tnit = int(total_steps * tinit_ratio)
    self.delta_t = delta_t
    self.initial_budget = init_r * self.num_modules
    self.target_budget = target_r * self.num_modules

    self.global_step = 0

  def _current_budget(self) -> int:
    """Linearly interpolate the rank budget between warmup and end of training."""
    if self.global_step <= self.tnit:
      return self.initial_budget
    progress = min(
      (self.global_step - self.tnit) / max(self.total_steps - self.tnit, 1),
      1.0
    )
    budget = self.initial_budget - progress * (self.initial_budget - self.target_budget)
    return max(int(budget), self.target_budget)

  @torch.no_grad()
  def step(self):
    """Call after each optimizer step."""
    self.global_step += 1

    if self.global_step % self.delta_t != 0:
      return
    if self.global_step <= self.tnit:
      return

    budget = self._current_budget()

    # 1. Compute importance for every module
    all_scores = []
    module_indices = []
    for idx, module in enumerate(self.adalora_modules):
      module.compute_importance()
      scores = module.importance_score  # (rank,)
      all_scores.append(scores)
      module_indices.extend([idx] * scores.numel())

    all_scores = torch.cat(all_scores)   # (total_triplets,)

    # 2. Global ranking: keep top-`budget` triplets
    if budget >= all_scores.numel():
      return  # nothing to prune

    _, topk_indices = torch.topk(all_scores, k=budget)
    keep_set = set(topk_indices.cpu().tolist())

    # 3. Update masks
    offset = 0
    for module in self.adalora_modules:
      r = module.rank
      mask = torch.zeros(r, device=module.rank_mask.device)
      for j in range(r):
        if (offset + j) in keep_set:
          mask[j] = 1.0
      module.rank_mask = mask
      offset += r

  def state_dict(self):
    return {'global_step': self.global_step}

  def load_state_dict(self, state_dict):
    self.global_step = state_dict['global_step']
