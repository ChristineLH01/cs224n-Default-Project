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
import random
import torch

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

from modules.lora import apply_lora_to_model
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
      print(f"Applying AdaLoRA fine-tuning to all linear layers, "
            f"initial rank {args.adalora_init_r}, target rank {args.adalora_target_r}, tinit ratio {args.adalora_tinit_ratio}, deltaT {args.adalora_delta_t}, "
            f"alpha {args.adalora_alpha}, dropout {args.lora_dropout}")
      apply_adalora_to_model(self.gpt, args)
    else:
      self.use_rslora = getattr(args, 'use_rslora', False)
      print(f"Applying LoRA fine-tuning with mode {self.fine_tune_mode}, rank {args.lora_rank}, alpha {args.lora_alpha}, dropout {args.lora_dropout}, use_rslora {self.use_rslora}")
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
    outputs = self.gpt(input_ids, attention_mask)
    last_hidden_state = outputs["last_token"]
    logits = self.paraphrase_detection_head(last_hidden_state)

    batch_size = logits.size(0)
    mapped_logits = torch.full((batch_size, YES_TOKEN_ID  + 1), float('-inf'), dtype=logits.dtype, device=logits.device)
    mapped_logits[:, YES_TOKEN_ID] = logits[:, 1]
    mapped_logits[:, NO_TOKEN_ID] = logits[:, 0]
  
    return mapped_logits



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
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

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

  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      # AdaLoRA rank pruning step
      if adalora_scheduler is not None:
        adalora_scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


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

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


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
                      choices=['full-finetune', 'lora', 'lora_attention', 'lora_mlp', 'adalora'],
                      help="Fine-tuning mode: full-finetune, lora (attn+mlp), lora_attention, lora_mlp, adalora")

  # LoRA arguments
  parser.add_argument("--lora_rank", type=int, default=0, help="Rank for LoRA(must be > 0)")
  parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha multiplier for LoRA")
  parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA/AdaLoRA")
  parser.add_argument("--use_rslora", action="store_true", help="Use RSLORA scaling (alpha / sqrt(rank))")

  # AdaLoRA arguments (Large dataset, larger alpha/init_r/target_r, smaller tinit_ratio may work better)
  parser.add_argument("--adalora_alpha", type=float, default=16.0, help="Alpha multiplier for AdaLoRA")
  parser.add_argument("--adalora_init_r", type=int, default=16, help="Initial rank for AdaLoRA (must be > 0)")
  parser.add_argument("--adalora_target_r", type=int, default=8, 
                      help="Target (final) rank for AdaLoRA after pruning. Must be <= lora_rank.")
  parser.add_argument("--adalora_tinit_ratio", type=float, default=0.05,
                      help="Fraction of total training steps before rank pruning begins.")
  parser.add_argument("--adalora_delta_t", type=int, default=50,
                      help="Prune ranks every N optimizer steps.")

  # Flash Attention arguments
  parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention")

  args = parser.parse_args()
  if args.fine_tune_mode in ('lora', 'lora_attention', 'lora_mlp'):
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
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)