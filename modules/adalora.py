import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict

beta1 = 0.85      # EMA coefficiens
beta2 = 0.85      
reg_orth_coef = 0.1  # Orthogonal regularization weight 

class AdaLoRALinear(nn.Module):
  def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
    super().__init__()
    self.linear = linear_layer
    self.rank = rank
    self.alpha = alpha

    in_features = linear_layer.weight.shape[1]
    out_features = linear_layer.weight.shape[0]
              
    self.P = nn.Parameter(torch.zeros(out_features, rank)) 
    self.Lambda = nn.Parameter(torch.zeros(rank, 1))       
    self.Q = nn.Parameter(torch.zeros(rank, in_features))
  
    # Track number of active singular values (non-pruned)
    self.ranknum = nn.Parameter(torch.zeros(1), requires_grad=False)
    self.ranknum.data.fill_(float(rank))

    self.scaling = self.alpha 

    self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    nn.init.zeros_(self.Lambda)
    nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """y = Wx + scaling * x @ Q^T @ diag(Λ) @ P^T / (ranknum + eps)"""
    base_out = self.linear(x)
    lora_out = self.dropout(x) @ (self.Q * self.Lambda).T @ self.P.T
    # lora_out = lora_out * self.scaling / (self.ranknum + 1e-5)
    # Use float32 for the scaling division to avoid FP16 overflow when ranknum → 0
    lora_out = lora_out * self.scaling / (self.ranknum.float() + 1e-5)
    return base_out + lora_out

  @property
  def current_rank(self):
    """Number of active (non-zero) singular values."""
    return int((self.Lambda.data.abs().squeeze() > 0).sum().item())


def apply_adalora_to_model(gpt_model, args):
  """
  Inject AdaLoRA adapters into all linear layers 
  Args: Must have adalora_init_r, adalora_alpha, lora_dropout.
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

  # Freeze all base parameters, only train P, Q, Lambda
  for name, param in gpt_model.named_parameters():
    param.requires_grad = any(name.endswith(k) for k in ('.P', '.Q', '.Lambda'))


def get_adalora_modules(model) -> List[AdaLoRALinear]:
  return [m for m in model.modules() if isinstance(m, AdaLoRALinear)]


def compute_orth_regu(model, regu_weight=reg_orth_coef):
  """
  Orthogonal regularization :
    R(P, Q) = ||P^T P - I||_F + ||Q Q^T - I||_F
  """
  regu_loss = 0.0
  num_param = 0
  for n, p in model.named_parameters():
    if n.endswith('.Q') or n.endswith('.P'):
      if n.endswith('.Q'):
        para_cov = p @ p.T   # Q Q^T
      else:
        para_cov = p.T @ p   # P^T P
      I = torch.eye(*para_cov.size(), device=para_cov.device)
      regu_loss += torch.norm(para_cov - I, p='fro')
      num_param += 1

  if num_param == 0:
    return torch.tensor(0.0, device=next(model.parameters()).device)
  return regu_weight * regu_loss / num_param


class AdaLoRARankScheduler:
  """
  Sensitivity-based importance: I(w) = |w * grad_w(L)|

  Sensitivity smoothing and uncertainty quantification::
    I_bar(t) = beta1 * I_bar(t-1) + (1 - beta1) * I(t)
    U_bar(t) = beta2 * U_bar(t-1) + (1 - beta2) * |I(t) - I_bar(t)|
  
  Triplet importance:
    S_{k,i} = s(lambda_{k,i}) + (1/d1)*sum_j s(P_{k,ji}) + (1/d2)*sum_j s(Q_{k,ij})
    s(w) = I_bar(t) * U_bar(t)

  Args:
    lora_r: Initial rank 
    target_rank: Average target rank per matrix after pruning.
    init_warmup: Steps of initial warmup (no pruning).
    final_warmup: Steps of final fine-tuning (fixed rank pattern).
    mask_interval: Steps between two budget allocations.
    beta1: EMA coefficient for sensitivity smoothing.
    beta2: EMA coefficient for uncertainty quantification.
    total_step: Total training steps.
  """

  def __init__(
    self,
    model,
    init_r: int,
    target_rank: int,
    init_warmup: int,
    final_warmup: int,
    mask_interval: int,
    total_step: int,
  ):
    self.model = model
    self.lora_init_rank = init_r
    self.target_rank = target_rank
    self.initial_warmup = init_warmup
    self.final_warmup = final_warmup
    self.mask_interval = mask_interval
    self.beta1 = beta1
    self.beta2 = beta2
    self.total_step = total_step

    # Importance tracking
    self.ipt: Dict[str, torch.Tensor] = {}          
    self.exp_avg_ipt: Dict[str, torch.Tensor] = {}   # EMA-smoothed sensitivity
    self.exp_avg_unc: Dict[str, torch.Tensor] = {}   # EMA-smoothed uncertainty

    self.global_step = 0

    # Compute budget
    self.total_rank = 0
    self.shape_dict: Dict[str, torch.Size] = {}

    for n, p in self.model.named_parameters():
      if n.endswith('.Q'):
        self.total_rank += p.size(0)
        self.shape_dict[n] = p.shape
      if n.endswith('.P'):
        self.shape_dict[n] = p.shape

    self.adalora_modules = get_adalora_modules(self.model)

    # Set target total rank
    self.target_budget = self.target_rank * len(self.adalora_modules)

  def schedule_threshold(self, step: int):
    """
    Claculate current budget and whether to apply masking
    """
    mask_ind = False
    target_budget = self.target_budget
    initial_warmup = self.initial_warmup
    final_warmup = self.final_warmup
    total_step = self.total_step
    self.global_step = step

    if step <= initial_warmup:
      # Phase 1: Initial warmup - no pruning
      curr_rank = self.total_rank
      mask_ind = False
    elif step > total_step - final_warmup:
      # Phase 3: Final fine-tuning - fix the rank pattern
      curr_rank = self.target_budget
      mask_ind = True
    else:
      # Phase 2: Cubic budget decrease
      mul_coeff = 1 - (step - initial_warmup) / (total_step - final_warmup - initial_warmup)
      curr_rank = target_budget + (self.total_rank - target_budget) * (mul_coeff ** 3)
      curr_rank = int(curr_rank)
      mask_ind = True

    return curr_rank, mask_ind

  @torch.no_grad()
  def update_ipt(self, model):
    """
    Update sensitivity-based importance scores 
    """
    for n, p in model.named_parameters():
      if (n.endswith('.P') or n.endswith('.Q') or n.endswith('.Lambda')) and p.requires_grad and p.grad is not None:
        if n not in self.ipt:
          self.ipt[n] = torch.zeros_like(p)
          self.exp_avg_ipt[n] = torch.zeros_like(p)
          self.exp_avg_unc[n] = torch.zeros_like(p)

        # Instantaneous sensitivity I(t)(w) = |w * grad_w(L)|
        self.ipt[n] = (p * p.grad).abs().detach()

        # Smoothed sensitivity via EMA
        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                              (1 - self.beta1) * self.ipt[n]

        # Uncertainty via EMA of absolute deviation
        self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                              (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

  def calculate_score(self, n):
    """Calculate importance score: s(w) = I_bar(w) * U_bar(w)."""
    return self.exp_avg_ipt[n] * self.exp_avg_unc[n]


  def _combine_ipt(self, ipt_Lambda, ipt_PQ):
    """
    S_{k,i} = s(lambda_{k,i}) + mean_j(s(P_{k,ji})) + mean_j(s(Q_{k,ij}))

    ipt_Lambda:  (r, 1) importance of singular values
    ipt_PQ: (r, 2) mean importance of P rows and Q columns
    """
    ipt_PQ = ipt_PQ.sum(dim=1, keepdim=False)
    sum_ipt = ipt_Lambda.view(-1) + ipt_PQ.view(-1)
    return sum_ipt

  @torch.no_grad()
  def mask_to_target_rank(self, model, curr_rank):
    """
    Globally rank all triplets by importance and prune the least important
    singular values (by zeroing lora_Lambda) to meet the current budget.
    """
    is_dict = {}
    combine_dict = {}
    singular_dict = {}

    # Calculate importance scores for each sub-matrix
    for n, p in model.named_parameters():
      if n.endswith('.Q') and n in self.exp_avg_ipt:
        # Q: mean importance over columns (dim=1) -> (r, 1)
        ipt_score = self.calculate_score(n)
        comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
        name_mat = n[:-1] + '%s'
        if name_mat not in combine_dict:
          combine_dict[name_mat] = [comb_ipt]
        else:
          combine_dict[name_mat].append(comb_ipt)

      if n.endswith('.P') and n in self.exp_avg_ipt:
        # P: mean importance over rows (dim=0) -> (r, 1)
        ipt_score = self.calculate_score(n)
        comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
        name_mat = n[:-1] + '%s'
        if name_mat not in combine_dict:
          combine_dict[name_mat] = [comb_ipt]
        else:
          combine_dict[name_mat].append(comb_ipt)

      if n.endswith('.Lambda') and n in self.exp_avg_ipt:
        ipt_score = self.calculate_score(n)
        name_mat = n[:-len('Lambda')] + '%s'
        singular_dict[name_mat] = ipt_score

    # Combine importance scores per triplet
    all_is = []
    for name_mat in combine_dict:
      if name_mat in singular_dict:
        ipt_Lambda = singular_dict[name_mat]
        ipt_PQ = torch.cat(combine_dict[name_mat], dim=1)  # (r, 2)
        sum_ipt = self._combine_ipt(ipt_Lambda, ipt_PQ)
        name_Lambda = name_mat % 'Lambda'
        is_dict[name_Lambda] = sum_ipt.view(-1, 1)
        all_is.append(sum_ipt.view(-1))

    if not all_is:
      return None

    # Global ranking: find threshold to keep top curr_rank triplets
    all_is_cat = torch.cat(all_is)
    num_to_prune = self.total_rank - curr_rank
    if num_to_prune <= 0 or num_to_prune >= all_is_cat.numel():
      return None

    mask_threshold = torch.kthvalue(all_is_cat, num_to_prune)[0].item()

    # Mask out unimportant singular values by zeroing lora_Lambda directly
    with torch.no_grad():
      for n, p in model.named_parameters():
        if n.endswith('.Lambda') and n in is_dict:
          p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)
          ranknum = (is_dict[n] > mask_threshold).sum().item()
          # Update the corresponding ranknum parameter
          ranknum_name = n[:-len('Lambda')] + 'ranknum'
          for rn, rp in model.named_parameters():
            if rn == ranknum_name:
              rp.data.fill_(ranknum)
              break

    return mask_threshold

  def update_and_mask(self, model, global_step):
    """
    Called after each optimizer step.

    Steps:
      1. Update importance scores (except during final warmup)
      2. Compute current budget 
      3. If mask_ind, prune to target rank
    """
    if global_step < self.total_step - self.final_warmup:
      self.update_ipt(model)

    curr_rank, mask_ind = self.schedule_threshold(global_step)

    mask_threshold = None
    if mask_ind:
      mask_threshold = self.mask_to_target_rank(model, curr_rank)

    return curr_rank, mask_threshold

  def step(self):
    self.global_step += 1
    return self.update_and_mask(self.model, self.global_step)

  def state_dict(self):
    return {
      'global_step': self.global_step,
      'ipt': {k: v.cpu() for k, v in self.ipt.items()},
      'exp_avg_ipt': {k: v.cpu() for k, v in self.exp_avg_ipt.items()},
      'exp_avg_unc': {k: v.cpu() for k, v in self.exp_avg_unc.items()},
    }

  def load_state_dict(self, state_dict):
    self.global_step = state_dict['global_step']
    if 'ipt' in state_dict:
      self.ipt = state_dict['ipt']
      self.exp_avg_ipt = state_dict['exp_avg_ipt']
      self.exp_avg_unc = state_dict['exp_avg_unc']
