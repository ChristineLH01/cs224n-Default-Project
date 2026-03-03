"""
Negative sample construction: Same-topic negatives from zetianli/en-wiki-paraphrased_split_3.
Encodes all 600k sentences, clusters with MiniBatchKMeans, then pairs each sentence
with its within-cluster nearest neighbor (not itself). Pairs with similarity outside
[SIM_LOW, SIM_HIGH] are discarded to avoid trivial and accidental-paraphrase negatives.
Produces 40k negative (is_duplicate=0) pairs pushed to zetianli/clustering_negatives.

Requirements:
    pip install sentence-transformers scikit-learn faiss-cpu
"""
import random
import numpy as np
import pandas as pd
import faiss
from collections import defaultdict
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

N_SAMPLES = 30_000
N_CLUSTERS = 5000
MIN_CLUSTER_SIZE = 3        # skip clusters too small for meaningful pairing
SIM_LOW = 0.3               # below this → too easy (different topic leaked in)
SIM_HIGH = 0.92             # above this → too similar (likely near-paraphrase)
SEED = 42
REPO_ID = "zetianli/clustering_negatives"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_clustering_negatives():
    print("Loading zetianli/en-wiki-paraphrased_split_3...")
    ds = load_dataset("zetianli/en-wiki-paraphrased_split_3", split="train")
    ds = ds.shuffle(seed=SEED)

    sentences = ds["original"]

    print(f"Encoding all {len(sentences)} sentences with {MODEL_NAME} on GPU...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    embeddings = model.encode(sentences, batch_size=512, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    print(f"Clustering into {N_CLUSTERS} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, batch_size=10_000, n_init=3)
    cluster_labels = kmeans.fit_predict(embeddings)

    # group indices by cluster, skip small clusters
    clusters = defaultdict(list)
    for idx, c in enumerate(cluster_labels):
        clusters[c].append(idx)
    clusters = {c: idxs for c, idxs in clusters.items() if len(idxs) >= MIN_CLUSTER_SIZE}
    print(f"Usable clusters (size >= {MIN_CLUSTER_SIZE}): {len(clusters)}")

    print("Finding within-cluster nearest neighbors via FAISS...")
    random.seed(SEED)
    pairs = []

    for c, idxs in clusters.items():
        if len(pairs) >= N_SAMPLES * 2:  # collect extra, filter later
            break

        cluster_embs = embeddings[idxs]  # (cluster_size, dim)
        local_index = faiss.IndexFlatIP(cluster_embs.shape[1])
        local_index.add(cluster_embs)

        # k=2: [0] is self, [1] is nearest neighbor
        sims, local_nns = local_index.search(cluster_embs, k=2)

        for local_i in range(len(idxs)):
            nn_local = local_nns[local_i, 1]
            nn_sim = sims[local_i, 1]

            # similarity band filter
            if nn_sim < SIM_LOW or nn_sim > SIM_HIGH:
                continue

            global_i = idxs[local_i]
            global_nn = idxs[nn_local]
            pairs.append((sentences[global_i], sentences[global_nn]))

    random.shuffle(pairs)
    pairs = pairs[:N_SAMPLES]
    print(f"Built {len(pairs)} within-cluster nearest-neighbor negative pairs.")

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
    build_clustering_negatives()
