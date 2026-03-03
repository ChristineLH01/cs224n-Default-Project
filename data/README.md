# Data

## Tracked files

| File | Description |
|---|---|
| `quora-train.csv` | Original Quora paraphrase training set |
| `quora-dev.csv` | Original Quora paraphrase dev set |
| `sonnets.txt` | Original Shakespeare sonnets |
| `sonnets_merged.txt` | Shakespeare sonnets + diffused sonnet variations merged |
| `diffused_sonnets.txt` | AI-generated sonnet variations from `ltg/en-wiki-paraphrased` |
| `sonnets_held_out*.txt` | Held-out sonnet splits for evaluation |

## Large files (not tracked in git)

The following files are **too large for GitHub (>100 MB)** and must be generated locally:

| File | Size | Description |
|---|---|---|
| `augmented_additional.csv` | ~350 MB | All augmented HuggingFace datasets combined |
| `augmented_train.csv` | ~400 MB | `augmented_additional.csv` + `quora-train.csv` merged |

### How to generate

Run from the project root:

```bash
python data_processing/paraphase/paraphasing_processing.py
```

See `data_processing/paraphase/README.md` for full details on the augmentation pipeline.
