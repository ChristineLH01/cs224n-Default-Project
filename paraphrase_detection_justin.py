'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import json
import os
import random
import shutil
import time
import uuid

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.amp import GradScaler, autocast

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW
from transformers import get_cosine_schedule_with_warmup

from modules.lora import apply_lora_to_model
from modules.dora import apply_dora_to_model
from modules.adalora import apply_adalora_to_model, AdaLoRARankScheduler
from modules.attention import CausalSelfAttention

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919
GPI_VOCAB_SIZE = 50257

class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
    
    self.fine_tune_mode = args.fine_tune_mode
    self.use_flash_attention = getattr(args, 'use_flash_attention', False)

    # Set flash attention if specified
    if self.use_flash_attention:
      print("Using Flash Attention.")
      for module in self.gpt.modules():
        if isinstance(module, CausalSelfAttention):
          module.use_flash_attention = self.use_flash_attention

    # Configure fine-tuning mode
    if self.fine_tune_mode == 'full-finetune':
      print("Full Fine-tuning")
      for param in self.gpt.parameters():
        param.requires_grad = True
    elif self.fine_tune_mode == 'adalora':
      print(f"Applying AdaLoRA"
            f"initial rank {args.adalora_init_r}, target rank {args.adalora_target_r}, tinit ratio {args.adalora_tinit_ratio}, deltaT {args.adalora_delta_t}, "
            f"alpha {args.adalora_alpha}")
      apply_adalora_to_model(self.gpt, args)
    elif self.fine_tune_mode in ('dora', 'dora_attention', 'dora_mlp'):
      self.use_rslora = getattr(args, 'use_rslora', False)
      print(f"Applying {self.fine_tune_mode}, rank {args.lora_rank}, alpha {args.lora_alpha}, dropout {args.lora_dropout}, use_rslora {self.use_rslora}")
      apply_dora_to_model(self.gpt, args)
    else:
      self.use_rslora = getattr(args, 'use_rslora', False)
      print(f"Applying {self.fine_tune_mode}, rank {args.lora_rank}, alpha {args.lora_alpha}, dropout {args.lora_dropout}, use_rslora {self.use_rslora}")
      apply_lora_to_model(self.gpt, args)

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    # Get the hidden states from GPT-2; use the last token's hidden state
    # since GPT-2 is autoregressive and the last token aggregates the full context.
    outputs = self.gpt(input_ids, attention_mask)
    
    last_token_hidden = outputs['last_token']  # [batch_size, hidden_size]
    # Project to 2 classes: index 0 = "no", index 1 = "yes".
    logits = self.paraphrase_detection_head(last_token_hidden)  # [batch_size, 2]
    
    # not modifying other code, map output to 8505 or 3919
    batch_size = logits.size(0)
    logit_vector_size = max(YES_TOKEN_ID, NO_TOKEN_ID) + 1 #GPI_VOCAB_SIZE
    mapped_logits = torch.full((batch_size, logit_vector_size), float('-inf'), dtype=logits.dtype, device=logits.device)
    mapped_logits[:, YES_TOKEN_ID] = logits[:, 1]
    mapped_logits[:, NO_TOKEN_ID] = logits[:, 0]
  
    return mapped_logits #logits



def save_model(model, optimizer, args, filepath, adalora_scheduler=None):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  if adalora_scheduler is not None:
    save_info['adalora_scheduler'] = adalora_scheduler.state_dict()

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args, run_dir):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  # ── DDP setup ──────────────────────────────────────────────────
  ddp = dist.is_available() and dist.is_initialized()
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  device = torch.device(f'cuda:{local_rank}') if args.use_gpu else torch.device('cpu')
  is_main = (local_rank == 0)

  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  num_workers = min(12, os.cpu_count())
  train_sampler = DistributedSampler(para_train_data) if ddp else None
  para_train_dataloader = DataLoader(para_train_data,
                                     batch_size=args.batch_size,
                                     shuffle=(train_sampler is None),
                                     sampler=train_sampler,
                                     collate_fn=para_train_data.collate_fn,
                                     num_workers=num_workers, pin_memory=True,
                                     persistent_workers=True, prefetch_factor=2)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn,
                                   num_workers=num_workers, pin_memory=True,
                                   persistent_workers=True, prefetch_factor=2)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)
  if ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)

  # Set up AdaLoRA rank scheduler if using AdaLoRA
  adalora_scheduler = None
  if args.fine_tune_mode == 'adalora':
    total_steps = args.epochs * len(para_train_dataloader)
    adalora_scheduler = AdaLoRARankScheduler(
      model=model,
      init_r=args.adalora_init_r,
      target_r=args.adalora_target_r,
      total_steps=total_steps,
      tinit_ratio=args.adalora_tinit_ratio,
      delta_t=args.adalora_delta_t,
    )

  scaler = GradScaler('cuda')

  num_training_steps = len(para_train_dataloader) * args.epochs
  num_warmup_steps = int(0.05 * num_training_steps)
  scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
  )

  best_dev_acc = 0
  best_dev_out_path = None
  dev_acc_scores = []
  dev_f1_scores = []
  train_losses = []
  total_train_time = 0.0
  epoch_train_times = []
  epoch_dev_inference_times = []

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    if train_sampler is not None:
      train_sampler.set_epoch(epoch)
    train_loss = 0
    num_batches = 0
    epoch_train_start = time.time()
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=not is_main):
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device, non_blocking=True)
      b_mask = b_mask.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)

      optimizer.zero_grad(set_to_none=True)
      with autocast('cuda'):
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, labels, reduction='mean')
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()

      # AdaLoRA rank pruning step
      if adalora_scheduler is not None:
        adalora_scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    epoch_train_time = time.time() - epoch_train_start
    total_train_time += epoch_train_time
    epoch_train_times.append(round(epoch_train_time, 2))
    train_loss = train_loss / num_batches if num_batches > 0 else 0.0
    train_losses.append(round(train_loss, 6))

    if is_main:
      eval_model = model.module if hasattr(model, 'module') else model
      dev_inf_start = time.time()
      dev_acc, dev_f1, dev_y_pred, dev_y_true, dev_sent_ids = model_eval_paraphrase(para_dev_dataloader, eval_model, device)
      epoch_dev_inference_times.append(round(time.time() - dev_inf_start, 2))
      dev_acc_scores.append(dev_acc)
      dev_f1_scores.append(dev_f1)
      print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
      print(f"dev acc history: {dev_acc_scores}")
      print(f"dev f1 history: {dev_f1_scores}")

      epoch_out_path = os.path.join(run_dir, f'para-dev-output_epoch{epoch}.csv')
      with open(epoch_out_path, 'w+') as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_sent_ids, dev_y_pred):
          f.write(f"{p}, {s} \n")

      if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_dev_out_path = epoch_out_path
        save_model(model.module if hasattr(model, 'module') else model, optimizer, args, args.filepath)

    if ddp:
      dist.barrier()

  # Copy best epoch's predictions and clean up per-epoch files.
  if is_main:
    final_dev_path = os.path.join(run_dir, 'para-dev-output.csv')
    shutil.copy2(best_dev_out_path, final_dev_path)
    print(f"Saved best dev predictions (acc {best_dev_acc:.3f}) to {final_dev_path}")
    for i in range(len(dev_acc_scores)):
      epoch_path = os.path.join(run_dir, f'para-dev-output_epoch{i}.csv')
      if epoch_path != best_dev_out_path and os.path.exists(epoch_path):
        os.remove(epoch_path)

  return best_dev_acc, dev_acc_scores, dev_f1_scores, train_losses, total_train_time, epoch_train_times, epoch_dev_inference_times


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_inf_start = time.time()
  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  dev_inference_time = round(time.time() - dev_inf_start, 2)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_inf_start = time.time()
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)
  test_inference_time = round(time.time() - test_inf_start, 2)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")

  return dev_inference_time, test_inference_time


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  # Fine-tuning mode 
  parser.add_argument("--fine_tune_mode", type=str, default="full-finetune",
                      choices=['full-finetune', 'lora', 'lora_attention', 'lora_mlp', 'adalora',
                               'dora', 'dora_attention', 'dora_mlp'],
                      help="Fine-tuning mode: full-finetune, lora (attn+mlp), lora_attention, lora_mlp, "
                           "adalora, dora (attn+mlp), dora_attention, dora_mlp")

  # LoRA arguments (same with DoRA, dropout shared with AdaLoRA)
  parser.add_argument("--lora_rank", type=int, default=0, help="Rank for LoRA adapters (must be > 0)")
  parser.add_argument("--lora_alpha", type=float, default=0, help="Alpha multiplier for LoRA")
  parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA/AdaLoRA")
  parser.add_argument("--use_rslora", action="store_true", help="Use RSLORA scaling (alpha / sqrt(rank))")

  # AdaLoRA arguments (Large dataset, larger alpha/init_r/target_r, smaller tinit_ratio may work better)
  parser.add_argument("--adalora_alpha", type=float, default=16.0, help="Alpha multiplier for AdaLoRA")
  parser.add_argument("--adalora_init_r", type=int, default=16, help="Initial rank for AdaLoRA (must be > 0)")
  parser.add_argument("--adalora_target_r", type=int, default=8, 
                      help="Target (final) rank for AdaLoRA after pruning. Must be <= adalora_init_r.")
  parser.add_argument("--adalora_tinit_ratio", type=float, default=0.05,
                      help="Fraction of total training steps before rank pruning begins.")
  parser.add_argument("--adalora_delta_t", type=int, default=50,
                      help="Prune ranks every N optimizer steps.")

  # Flash Attention arguments
  parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention")
            
  args = parser.parse_args()
  if args.fine_tune_mode in ('lora', 'lora_attention', 'lora_mlp',
                              'dora', 'dora_attention', 'dora_mlp'):
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
      raise ValueError(f"--lora_rank and --lora_alpha must be > 0")
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  # ── DDP init ───────────────────────────────────────────────────
  if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
    dist.init_process_group(backend="nccl")

  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  is_main = (local_rank == 0)

  args = get_args()
  run_name = f'{args.fine_tune_mode}_epochs{args.epochs}_lr{args.lr}_bs{args.batch_size}'
  if args.fine_tune_mode in ('lora', 'lora_attention', 'lora_mlp', 'dora', 'dora_attention', 'dora_mlp'):
    run_name += f'_rank{args.lora_rank}_alpha{args.lora_alpha}'
    run_name += '_rslora' if args.use_rslora else ''
  
  if args.fine_tune_mode == 'adalora':
    run_name += f'_adalora_initr{args.adalora_init_r}_targetr{args.adalora_target_r}_tinit{int(args.adalora_tinit_ratio*100)}%_deltat{args.adalora_delta_t}_alpha{args.adalora_alpha}'

  if args.use_flash_attention:
    run_name += '_flash'
  exp_base = 'paraphrase_exp_results_lora' if args.lora_rank > 0 else 'paraphrase_exp_results_full'
  run_dir = os.path.join(exp_base, args.model_size, run_name)
  if is_main:
    os.makedirs(run_dir, exist_ok=True)
  if dist.is_available() and dist.is_initialized():
    dist.barrier()  # Wait for rank 0 to create the directory.

  args.para_test_out = os.path.join(run_dir, 'para-test-output.csv')
  args.filepath = os.path.join(run_dir, 'model.pt')
  seed_everything(args.seed)
  best_dev_acc, dev_acc_scores, dev_f1_scores, train_losses, total_train_time, epoch_train_times, epoch_dev_inference_times = train(args, run_dir)

  if is_main:
    dev_inference_time, test_inference_time = test(args)
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    results_file = os.path.join(exp_base, 'experiment_results.jsonl')
    result = {
      "uuid": str(uuid.uuid4()),
      "task": "paraphrase_detection",
      "additional_data_used": True if args.para_train != "data/quora-train.csv" else False,
      "best_dev_acc": best_dev_acc,
      "dev_acc_scores": dev_acc_scores,
      "dev_f1_scores": dev_f1_scores,
      "train_losses": train_losses,
      "job_id": os.environ.get("JOB_ID"),
      "method": "LoRA" if args.lora_rank > 0 else "full_parameter_tuning",
      "method_metadata": {},
      "args": vars(args),
      "training_meta_data": {
        "num_gpus": num_gpus,
        "gpu": "H100",
        "total_train_time_sec": round(total_train_time, 2),
        "avg_training_time_per_epoch_sec": round(sum(epoch_train_times) / len(epoch_train_times), 2),
        "avg_dev_inference_time_per_epoch_sec": round(sum(epoch_dev_inference_times) / len(epoch_dev_inference_times), 2),
        "total_dev_inference_time_sec": round(sum(epoch_dev_inference_times), 2),
        "total_test_inference_time_sec": test_inference_time,
      },
    }
    with open(results_file, "a") as f:
      f.write(json.dumps(result) + "\n")
    print(f"Saved experiment result to {results_file}")
    if os.path.exists(args.filepath):
      os.remove(args.filepath)
      print(f"Deleted checkpoint {args.filepath}")

  if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()
