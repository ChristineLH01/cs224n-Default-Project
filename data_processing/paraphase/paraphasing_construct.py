import pandas as pd
from datasets import load_dataset, Dataset


def analyze_dataset(csv_path):
    df = pd.read_csv(csv_path, sep='\t')
    total = len(df)
    count_1 = int((df['is_duplicate'] == 1.0).sum())
    count_0 = int((df['is_duplicate'] == 0.0).sum())
    ratio_1 = count_1 / total
    ratio_0 = count_0 / total
    to_add = abs(count_1 - count_0)
    minority = 'duplicate (1)' if count_1 < count_0 else 'non-duplicate (0)'

    print(f"Total datapoints:       {total}")
    print(f"Duplicate (1):          {count_1} ({ratio_1:.2%})")
    print(f"Non-duplicate (0):      {count_0} ({ratio_0:.2%})")
    print(f"To add to balance:      {to_add} more {minority} samples")

    return {
        'total': total,
        'count_1': count_1,
        'count_0': count_0,
        'ratio_1': ratio_1,
        'ratio_0': ratio_0,
        'to_add': to_add,
        'minority_class': minority,
    }




def sample_and_split_wiki_paraphrased(
    n_samples=3_000_000,
    n_splits=5,
    hf_repo_prefix="zetianli/en-wiki-paraphrased_split",
    seed=42,
):
    print("Loading dataset...")
    dataset = load_dataset("ltg/en-wiki-paraphrased", split="train")

    total = len(dataset)
    print(f"Total available: {total}")
    assert n_samples <= total, f"Requested {n_samples} but only {total} available"

    print(f"Sampling {n_samples} datapoints...")
    sampled = dataset.shuffle(seed=seed).select(range(n_samples))

    split_size = n_samples // n_splits
    for k in range(n_splits):
        start = k * split_size
        end = start + split_size if k < n_splits - 1 else n_samples
        split = sampled.select(range(start, end))
        repo_id = f"{hf_repo_prefix}_{k}"
        print(f"Pushing split {k} ({len(split)} samples) to {repo_id}...")
        split.push_to_hub(repo_id)
        print(f"Split {k} pushed.")

    print("Done.")


def build_balancing_pos_paraphasing(
    n_llm_paraphrases=35_000,
    n_wiki=38_845,
    seed=42,
    repo_id="zetianli/balancing_positive_paraphrases",
):
    # --- Part 1: redis/llm-paraphrases, label=1 only ---
    print("Loading redis/llm-paraphrases...")
    llm = load_dataset("redis/llm-paraphrases", split="train")
    llm_pos = llm.filter(lambda x: x["label"] == 1)
    print(f"  label=1 samples available: {len(llm_pos)}")
    llm_sampled = llm_pos.shuffle(seed=seed).select(range(n_llm_paraphrases))

    llm_df = pd.DataFrame({
        "sentence1": llm_sampled["sentence_a"],
        "sentence2": llm_sampled["sentence_b"],
        "is_duplicate": [1.0] * len(llm_sampled),
    })

    # --- Part 2: zetianli/en-wiki-paraphrased_split_0, all label=1 ---
    print("Loading zetianli/en-wiki-paraphrased_split_0...")
    wiki = load_dataset("zetianli/en-wiki-paraphrased_split_0", split="train")
    print(f"  total samples available: {len(wiki)}")
    wiki_sampled = wiki.shuffle(seed=seed).select(range(n_wiki))

    wiki_df = pd.DataFrame({
        "sentence1": wiki_sampled["original"],
        "sentence2": wiki_sampled["paraphrase"],
        "is_duplicate": [1.0] * len(wiki_sampled),
    })

    # --- Combine and push ---
    combined_df = pd.concat([llm_df, wiki_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Combined dataset size: {len(combined_df)}")

    combined_dataset = Dataset.from_pandas(combined_df)
    print(f"Pushing to {repo_id}...")
    combined_dataset.push_to_hub(repo_id)
    print("Done.")


if __name__ == "__main__":
    analyze_dataset("data/quora-train.csv")
    build_balancing_pos_paraphasing()

