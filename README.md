# CS 224N Default Final Project: Build GPT-2

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course
website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the GPT-2 model to
better understand its architecture.
In the second part, you will use the token embeddings produced by your GPT-2 model on two downstream tasks: paraphrase
detection and sonnet generation. You will implement extensions to improve your model's performance on these tasks.

In broad strokes, Part 1 of this project targets:

* modules/attention.py: Missing code blocks.
* modules/gpt2_layer.py: Missing code blocks.
* models/gpt2.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

To test Part 1, you will run:

* `optimizer_test.py`: To test your implementation of `optimizer.py`.
* `sanity_check.py`: To test your implementation of GPT models.
* `classifier.py` : To perform sentiment classification using your models.

In Part 2 of this project, you will use GPT2 (via cloze-style classification) detect if one sentence is a paraphrase of 
another as well as generate sonnets via autoregressive language modeling.  

To test Part 2, you will run:

* `paraphrase_detection.py`: To perform paraphrase detection. 
* `sonnet_generation.py`: To perform sonnet generation.

Important: Adjust training hyperparameters, particularly batch size, according to your GPU's specifications to optimize performance and prevent out-of-memory errors.

## Pre-testing instructions

While there are missing code blocks that you need to implement in both of these files, the main focus of this second 
part are the extensions: how you modify your GPT2 model to improve its ability to determine if one sentence is a 
paraphrase of another as well as its ability to generate sonnets. 

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Extensions

### Parameter-Efficient Fine-Tuning (PEFT)

We implement several LoRA variants for parameter-efficient fine-tuning, all available via `--fine_tune_mode`:

- **LoRA** — low-rank adaptation applied to attention, MLP, or both (`lora`, `lora_attention`, `lora_mlp`)
- **AdaLoRA** — adaptive rank allocation that prunes less important ranks during training (`adalora`)
- **rsLoRA** — rank-stabilized LoRA scaling (`alpha / sqrt(rank)`) via `--use_rslora`
- **DoRA** — weight-decomposed LoRA with magnitude/direction separation (`dora`, `dora_attention`, `dora_mlp`)

Additional LoRA controls: `--lora_rank`, `--lora_alpha`, `--lora_dropout`, and `--rank_pattern` for per-layer rank schedules.

### Flash Attention

Pass `--use_flash_attention` to enable Flash Attention for faster training and inference with reduced memory usage.

### Data Augmentation

We construct augmented training data for paraphrase detection using several negative-mining strategies. The pipeline lives in `data_processing/paraphase/` and produces the following additional datasets:

| Strategy | Description |
|---|---|
| Random positives / negatives | Sampled from the Quora split (150k positives, 20k negatives) |
| NLI negatives | Near-miss negatives via adjective/adverb antonym flipping + negation |
| Embedding negatives | Hard negatives via FAISS nearest-neighbor search on sentence embeddings (GPU required) |
| Clustering negatives | Same-topic negatives via KMeans clustering + within-cluster nearest neighbor (GPU required) |
| Balancing positives | Extra positives to balance the augmented set |

The combined output files (`data/augmented_additional.csv` ~350 MB, `data/augmented_train.csv` ~400 MB) are too large for git. To regenerate:

```bash
python data_processing/paraphase/paraphasing_processing.py
```

This loads the pre-built splits from HuggingFace (`zetianli/*`), combines them, and merges with `data/quora-train.csv`. See `data_processing/paraphase/README.md` for full details.

For sonnet generation, we also include `data/diffused_sonnets.txt` (AI-generated sonnet variations) and `data/sonnets_merged.txt` (original + diffused sonnets merged).

#### External Datasets Used

| Dataset | Used In | Purpose |
|---|---|---|
| [`ltg/en-wiki-paraphrased`](https://huggingface.co/datasets/ltg/en-wiki-paraphrased) | Paraphrase augmentation | Source corpus for constructing paraphrase pairs; sampled and split into `zetianli/en-wiki-paraphrased_split_*` |
| [`redis/llm-paraphrases`](https://huggingface.co/datasets/redis/llm-paraphrases) | Paraphrase augmentation | LLM-generated paraphrase pairs (label=1 subset) used for balancing positives |
| [`Lambent/shakespeare_sonnets_diffused`](https://huggingface.co/datasets/Lambent/shakespeare_sonnets_diffused) | Sonnet generation | AI-generated sonnet variations used to augment the original Shakespeare sonnets |

### Distributed Training

The `*_justin.py` scripts are GPU-cluster-ready variants that support distributed data-parallel training:

- `paraphrase_detection_justin.py` — supports multi-GPU training via `torchrun` (NCCL backend, `DistributedDataParallel`)
- `sonnet_generation_justin.py` — single-GPU training script

### Running

The `.sh` scripts are SLURM job examples for launching training on a cluster:

```bash
# Paraphrase detection — multi-GPU with torchrun
sbatch run_paraphrase.sh

# Sonnet generation — single-GPU
sbatch run_sonnet.sh
```

## Test Submission Results

**Paraphrase Detection** (accuracy)

| Sub | Test | Dev | Model | Method | Key Params |
|---|---|---|---|---|---|
| #1 | 0.877 | 0.9077 | gpt2-medium | Full fine-tune | lr=1e-4, ep=13, aug=True |
| #2 | 0.868 | 0.9092 | gpt2-medium | Full fine-tune | lr=8e-5, ep=14, aug=False |
| #3 | **0.897** | 0.9080 | gpt2-medium | rsLoRA r=32 a=11.2 | lr=5e-4, ep=25, aug=True |

**Sonnet Generation** (chrF)

| Sub | Test | Dev | Model | Method | Key Params |
|---|---|---|---|---|---|
| #1 | 43.362 | 45.17 | gpt2-medium | Full fine-tune | lr=1e-3, ep=30, bs=4, aug=True |
| #2 | **44.99** | 46.45 | gpt2-medium | rsLoRA r=32 a=11.2 | lr=1e-3, ep=40, bs=4, aug=True |
| #3 | 43.364 | 46.38 | gpt2-medium | DoRA r=16 a=32 | lr=1e-3, ep=40, bs=4, aug=True |

See `test_sub_record.txt` for full details and file paths.

## Acknowledgement

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).