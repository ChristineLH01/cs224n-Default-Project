'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import os
import random
import shutil
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

from modules.lora import apply_lora_to_model
from modules.dora import apply_dora_to_model
from modules.adalora import apply_adalora_to_model, AdaLoRARankScheduler, compute_orth_regu
from modules.attention import CausalSelfAttention
from evaluation import test_sonnet

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


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    
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
            f"init_r={args.adalora_init_r}, target_r={args.adalora_target_r}, tinit ratio {args.adalora_tinit_ratio}, tfinal ratio {args.adalora_tfinal_ratio}, "
            f"deltaT {args.adalora_delta_t}, alpha={args.adalora_alpha}")
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
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    outputs = self.gpt(input_ids, attention_mask)
    sequence_hidden_state = outputs['last_hidden_state']
    sequence_logits = self.gpt.hidden_state_to_token(sequence_hidden_state)
    return sequence_logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


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


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Test held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  # Dev held-out dataset
  held_out_sonnet_dev_dataset = SonnetsDataset(args.held_out_sonnet_dev_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  best_chrf = -1.0

  # Set up AdaLoRA rank scheduler if using AdaLoRA
  adalora_scheduler = None
  if args.fine_tune_mode == 'adalora':
    total_steps = args.epochs * len(sonnet_dataloader)
    init_warmup = int(total_steps * args.adalora_tinit_ratio)
    final_warmup = int(total_steps * args.adalora_tfinal_ratio)
    adalora_scheduler = AdaLoRARankScheduler(
      model=model,
      init_r=args.adalora_init_r,
      target_rank=args.adalora_target_r,
      init_warmup=init_warmup,
      final_warmup=final_warmup,
      mask_interval=args.adalora_delta_t,
      total_step=total_steps,
    )

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, ignore_index=model.tokenizer.pad_token_id, reduction='mean')

      # AdaLoRA: add orthogonal regularization to loss 
      if adalora_scheduler is not None:
        orth_regu = compute_orth_regu(model)
        loss = loss + orth_regu

      loss.backward()
      
      # Clip gradients to prevent exploding gradients
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      
      optimizer.step()

      # AdaLoRA rank pruning step
      if adalora_scheduler is not None:
        adalora_scheduler.step()

      train_loss += loss.item()
      num_batches += 1
      
      if num_batches % 5 == 0:
        print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()

    dev_generated_sonnets = []
    for batch in held_out_sonnet_dev_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      dev_generated_sonnets.append((batch[0], output[1]))
      print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets

    # Store the generated sonnets in temp file
    with open(args.temp_sonnet_out, 'w+') as f:
      f.write('--Generated Sonnets-- \n\n')
      for sonnet_id, sonnet_text in dev_generated_sonnets:
        f.write(f'\n{sonnet_id}\n')
        f.write(sonnet_text)

    chrf = test_sonnet(test_path=args.temp_sonnet_out, gold_path=args.true_held_out_sonnet_dev_path)
    print(f'Epoch {epoch}: ChrF :: {chrf:.4f}, best :: {best_chrf:.4f}')
    if chrf > best_chrf:
      best_chrf = chrf
      # Copy the current best generated sonnets to dev submission file 
      shutil.copy2(args.temp_sonnet_out, args.dev_sonnet_out)
      # Save the model checkpoint that achieved the best performance on dev set
      save_model(model, optimizer, args, args.filepath, adalora_scheduler=adalora_scheduler)

  # Clean up temporary file
  print(f"Best ChrF on dev set: {best_chrf:.4f}")
  os.remove(args.temp_sonnet_out)  


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--held_out_sonnet_dev_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--true_held_out_sonnet_dev_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--temp_sonnet_out", type=str, default="predictions/temp_generated_sonnets.txt")
  parser.add_argument("--dev_sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # Fine-tuning mode 
  parser.add_argument("--fine_tune_mode", type=str, default="full-finetune",
                      choices=['full-finetune', 'lora', 'lora_attention', 'lora_mlp', 'adalora',
                               'dora', 'dora_attention', 'dora_mlp'],
                      help="Fine-tuning mode: full-finetune, lora (attn+mlp), lora_attention, lora_mlp, "
                           "adalora, dora (attn+mlp), dora_attention, dora_mlp")

  # LoRA arguments (same with DoRA, dropout shared with AdaLoRA)
  parser.add_argument("--lora_rank", type=int, default=0, help="Rank for LoRA/DoRA (must be > 0)")
  parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha multiplier for LoRA/DoRA")
  parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA/AdaLoRA/DoRA")
  parser.add_argument("--use_rslora", action="store_true", help="Use RSLORA scaling (alpha / sqrt(rank))")

  # AdaLoRA arguments (Sonnets has small data, smaller alpha/init_r/target_r, larger tinit_ratio may work better)
  # Smaller dataset can set tfinal_ratio to 0 to skip the final fixed-rank fine-tuning phase,
  parser.add_argument("--adalora_alpha", type=float, default=8.0, help="Alpha scaling for AdaLoRA")
  parser.add_argument("--adalora_init_r", type=int, default=12, help="Initial rank for each incremental matrix")
  parser.add_argument("--adalora_target_r", type=int, default=4, help="Average target rank per matrix after pruning")
  parser.add_argument("--adalora_tinit_ratio", type=float, default=0.2, help="Fraction of total steps for initial warmup (no pruning)")
  parser.add_argument("--adalora_tfinal_ratio", type=float, default=0.0, help="Fraction of total steps for final fine-tuning (fixed rank)")
  parser.add_argument("--adalora_delta_t", type=int, default=10, help="Mask/prune ranks every N optimizer steps")

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
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)
