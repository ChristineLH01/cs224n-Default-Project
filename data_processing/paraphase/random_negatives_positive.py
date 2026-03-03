"""
Constructs from zetianli/en-wiki-paraphrased_split_1:
  - 150k positive pairs (original + paraphrase, is_duplicate=1.0)
  - 20k negative pairs from the remaining rows (random shift, is_duplicate=0.0)
Both with 50% probability of swapping sentence1/sentence2 positions.
Pushes combined dataset to zetianli/random_negatives.
"""
import random
import pandas as pd
from datasets import load_dataset, Dataset

N_POS = 160_000
N_NEG = 10_000
SEED = 42
REPO_POS = "zetianli/random_positives"
REPO_NEG = "zetianli/random_negatives"


def flip_pairs(pairs, labels, seed):
    random.seed(seed)
    s1, s2 = [], []
    for p in pairs:
        if random.random() < 0.5:
            s1.append(p[1])
            s2.append(p[0])
        else:
            s1.append(p[0])
            s2.append(p[1])
    return s1, s2


def build_random_negatives():
    print("Loading zetianli/en-wiki-paraphrased_split_1...")
    ds = load_dataset("zetianli/en-wiki-paraphrased_split_1", split="train")
    ds = ds.shuffle(seed=SEED)
    total = len(ds)
    assert total >= N_POS + N_NEG, f"Not enough data: {total} < {N_POS + N_NEG}"

    originals = ds["original"]
    paraphrases = ds["paraphrase"]

    # --- Positives: first N_POS rows ---
    pos_pairs = [(originals[i], paraphrases[i]) for i in range(N_POS)]
    print(f"Built {len(pos_pairs)} positive pairs.")

    # --- Negatives: from remaining rows, shift-paired ---
    neg_pool_orig = originals[N_POS: N_POS + N_NEG]
    neg_pool_para = paraphrases[N_POS: N_POS + N_NEG]
    shifted = list(range(N_NEG // 2, N_NEG)) + list(range(0, N_NEG // 2))
    neg_pairs = [(neg_pool_orig[i], neg_pool_para[shifted[i]]) for i in range(N_NEG)]
    print(f"Built {len(neg_pairs)} negative pairs.")

    # --- 50% position flip ---
    random.seed(SEED)
    pos_s1, pos_s2 = flip_pairs(pos_pairs, [1.0] * N_POS, SEED)
    neg_s1, neg_s2 = flip_pairs(neg_pairs, [0.0] * N_NEG, SEED + 1)

    pos_df = pd.DataFrame({
        "sentence1": pos_s1,
        "sentence2": pos_s2,
        "is_duplicate": [1.0] * N_POS,
    })

    neg_df = pd.DataFrame({
        "sentence1": neg_s1,
        "sentence2": neg_s2,
        "is_duplicate": [0.0] * N_NEG,
    })

    print(f"Pushing {len(pos_df)} positives to {REPO_POS}...")
    Dataset.from_pandas(pos_df).push_to_hub(REPO_POS)

    print(f"Pushing {len(neg_df)} negatives to {REPO_NEG}...")
    Dataset.from_pandas(neg_df).push_to_hub(REPO_NEG)

    print("Done.")


if __name__ == "__main__":
    build_random_negatives()
