"""
Combines augmented HuggingFace paraphrase datasets into quora-train format,
then merges with the original quora-train.csv.

All HuggingFace datasets must have columns: sentence1, sentence2, is_duplicate
Output format matches quora-train.csv: id  sentence1  sentence2  is_duplicate (TSV)

Usage:
    python data_processing/paraphase/paraphasing_processing.py
"""
import uuid
import pandas as pd
from datasets import load_dataset

# ── List of HuggingFace datasets to combine ────────────────────────────────
HF_DATASETS = [
    "zetianli/random_positives",
    "zetianli/random_negatives",
    "zetianli/nli_negatives",
    "zetianli/embedding_negatives",
    "zetianli/clustering_negatives",
    "zetianli/balancing_positive_paraphrases",
]

QUORA_TRAIN_PATH = "data/quora-train.csv"
AUGMENTED_ADDITIONAL_PATH = "data/augmented_additional.csv"
AUGMENTED_TRAIN_PATH = "data/augmented_train.csv"
SEED = 42


def make_id():
    return str(uuid.uuid4())


def load_hf_datasets(hf_datasets):
    """Load and concatenate all HuggingFace datasets into a unified DataFrame."""
    dfs = []
    for repo_id in hf_datasets:
        print(f"Loading {repo_id}...")
        ds = load_dataset(repo_id, split="train")
        df = pd.DataFrame({
            "sentence1": ds["sentence1"],
            "sentence2": ds["sentence2"],
            "is_duplicate": [float(x) for x in ds["is_duplicate"]],
        })
        print(f"  {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined.insert(0, "id", [make_id() for _ in range(len(combined))])
    print(f"\nTotal augmented rows: {len(combined)}")
    return combined


def save_augmented_additional(df, path):
    """Save augmented data in quora TSV format."""
    df.to_csv(path, sep="\t", index=False)
    print(f"Saved augmented_additional → {path}")


def combine_with_quora(augmented_path, quora_path, output_path):
    """Merge augmented_additional with quora-train, shuffle, and save."""
    print(f"\nLoading {quora_path}...")
    quora_df = pd.read_csv(quora_path, sep="\t")
    print(f"  Quora rows: {len(quora_df)}")

    augmented_df = pd.read_csv(augmented_path, sep="\t")
    print(f"  Augmented rows: {len(augmented_df)}")

    combined = pd.concat([quora_df, augmented_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined.to_csv(output_path, sep="\t", index=False)

    pos = (combined["is_duplicate"] == 1.0).sum()
    neg = (combined["is_duplicate"] == 0.0).sum()
    print(f"\nSaved augmented_train → {output_path}")
    print(f"  Total: {len(combined)} rows | pos: {pos} ({pos/len(combined):.1%}) | neg: {neg} ({neg/len(combined):.1%})")


if __name__ == "__main__":
    augmented_df = load_hf_datasets(HF_DATASETS)
    save_augmented_additional(augmented_df, AUGMENTED_ADDITIONAL_PATH)
    combine_with_quora(AUGMENTED_ADDITIONAL_PATH, QUORA_TRAIN_PATH, AUGMENTED_TRAIN_PATH)
