"""
Negative sample construction: Embedding-similar negatives from zetianli/en-wiki-paraphrased_split_2.
Encodes all sentences with a sentence-transformer, builds a FAISS index,
then pairs each sentence with its nearest neighbor that is NOT its own paraphrase.
Produces 40k negative (is_duplicate=0) pairs pushed to zetianli/embedding_negatives.

Requirements:
    pip install sentence-transformers faiss-gpu
"""
import random
import numpy as np
import pandas as pd
import faiss
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer

N_SAMPLES = 50_000
SEED = 42
SIMILARITY_THRESHOLD = 0.95  # skip pairs that are suspiciously close (likely true paraphrases)
REPO_ID = "zetianli/embedding_negatives"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_embedding_negatives():
    print("Loading zetianli/en-wiki-paraphrased_split_2...")
    ds = load_dataset("zetianli/en-wiki-paraphrased_split_2", split="train")
    ds = ds.shuffle(seed=SEED)

    sentences = ds["original"]

    print(f"Encoding all {len(sentences)} sentences with {MODEL_NAME} on GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    embeddings = model.encode(sentences, batch_size=512, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    rng = np.random.default_rng(SEED)

    # try GPU FAISS first (faiss-gpu via conda), fall back to CPU IVF
    gpu_available = hasattr(faiss, "StandardGpuResources")
    if gpu_available:
        print("  Using GPU FAISS (flat exact search)...")
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print(f"  Adding {len(embeddings)} vectors to GPU index...")
        index.add(embeddings)
        print("  Index ready.")
    else:
        print("  Using CPU FAISS IVF (approximate)...")
        n_lists = 256
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_lists, faiss.METRIC_INNER_PRODUCT)
        train_size = 50_000
        train_idx = rng.choice(len(embeddings), size=train_size, replace=False)
        print(f"  Training IVF on {train_size} samples...")
        index.train(embeddings[train_idx])
        print(f"  Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)
        index.nprobe = 32
        print("  Index ready.")

    print(f"Searching for nearest neighbors (batch_size=50k)...")
    batch_size = 50_000
    n_batches = (len(embeddings) + batch_size - 1) // batch_size
    all_sims = []
    all_indices = []
    for b, start in enumerate(range(0, len(embeddings), batch_size)):
        batch = embeddings[start:start + batch_size]
        sims, idxs = index.search(batch, k=2)
        all_sims.append(sims)
        all_indices.append(idxs)
        end = min(start + batch_size, len(embeddings))
        print(f"  Batch {b+1}/{n_batches} — {end}/{len(embeddings)} searched")
    similarities = np.vstack(all_sims)
    indices = np.vstack(all_indices)

    pairs = []
    for i in range(len(sentences)):
        nn_idx = indices[i, 1]
        nn_sim = similarities[i, 1]
        if nn_sim >= SIMILARITY_THRESHOLD:
            continue  # too similar, likely a true paraphrase — skip
        pairs.append((sentences[i], sentences[nn_idx]))

    print(f"Built {len(pairs)} embedding-similar negative pairs.")

    random.seed(SEED)

    # trim to N_SAMPLES if we have more
    pairs = pairs[:N_SAMPLES]

    # 50% chance to swap sentence1 and sentence2
    s1 = [p[1] if random.random() < 0.5 else p[0] for p in pairs]
    s2 = [p[0] if s1[i] == p[1] else p[1] for i, p in enumerate(pairs)]

    neg_df = pd.DataFrame({
        "sentence1": s1,
        "sentence2": s2,
        "is_duplicate": [0.0] * len(pairs),
    })

    dataset = Dataset.from_pandas(neg_df)
    print(f"Pushing to {REPO_ID}...")
    dataset.push_to_hub(REPO_ID)
    print("Done.")


if __name__ == "__main__":
    build_embedding_negatives()
