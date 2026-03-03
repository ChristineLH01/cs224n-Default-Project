# Paraphrase Data Processing

This folder contains scripts for constructing augmented paraphrase training data.

## Scripts

| Script | Description |
|---|---|
| `paraphasing_construct.py` | Samples from HuggingFace datasets and pushes split datasets |
| `paraphasing_processing.py` | **Main script** — combines all HuggingFace datasets and generates the augmented CSVs |
| `random_negatives_positive.py` | Constructs random positives (150k) and random negatives (20k) from `split_1` |
| `nli_negatives.py` | Constructs near-miss negatives via adjective/adverb antonym flipping + negation |
| `embedding_negatives.py` | Constructs hard negatives using FAISS nearest-neighbor search on sentence embeddings |
| `clustering_negatives.py` | Constructs same-topic negatives via KMeans clustering + within-cluster nearest neighbor |
| `run_embedding_negatives.sh` | SLURM script for `embedding_negatives.py` (requires GPU) |
| `run_clustering_negatives.sh` | SLURM script for `clustering_negatives.py` (requires GPU) |

## Generating the augmented data

The output CSV files (`augmented_additional.csv`, `augmented_train.csv`) are **not tracked in git** due to their large size (~350–400 MB each). To regenerate them, run from the project root:

```bash
python data_processing/paraphase/paraphasing_processing.py
```

This will:
1. Load the following HuggingFace datasets (must be accessible):
   - `zetianli/random_positives`
   - `zetianli/random_negatives`
   - `zetianli/nli_negatives`
   - `zetianli/embedding_negatives`
   - `zetianli/clustering_negatives`
   - `zetianli/balancing_pos_paraphasing`
2. Combine and shuffle them → `data/augmented_additional.csv`
3. Merge with `data/quora-train.csv` and shuffle → `data/augmented_train.csv`

Both output files are in the same TSV format as `quora-train.csv`:
```
id    sentence1    sentence2    is_duplicate
```
