"""
Negative sample construction: Near-miss semantic (1-3 word antonym flip) negatives
from zetianli/en-wiki-paraphrased_split_4.

For each paraphrase pair, take the paraphrase and replace 1-3 content words
(adjectives, verbs, nouns) with WordNet antonyms. The result looks topically
similar but has subtly flipped meaning — a hard negative for paraphrase detection.

Produces 40k negative (is_duplicate=0) pairs pushed to zetianli/nli_negatives.

Requirements:
    pip install nltk
    python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"
"""
import random
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
from datasets import load_dataset, Dataset

nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

_detokenizer = TreebankWordDetokenizer()

N_SAMPLES = 70_000
N_FLIPS_MIN = 2
N_FLIPS_MAX = 4
SEED = 42
REPO_ID = "zetianli/nli_negatives"

# Auxiliary verbs after which "not" can be safely inserted
AUXILIARIES = {"is", "are", "was", "were", "will", "can", "could", "should",
               "would", "has", "have", "had", "does", "do", "did", "must", "might", "shall"}

# Adjectives and adverbs — both are drop-in replaceable with no grammar side effects
ELIGIBLE_POS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}

# Map NLTK POS tags to WordNet POS
POS_MAP = {
    "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
    "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV,
}

# Hardcoded safe swap pairs: temporal/spatial/directional words
HARDCODED_SWAPS = {
    "before": "after", "after": "before",
    "early": "late", "late": "early",
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "inside": "outside", "outside": "inside",
    "above": "below", "below": "above",
    "over": "under", "under": "over",
    "past": "future", "future": "past",
    "start": "end", "end": "start",
    "open": "closed", "closed": "open",
    "first": "last", "last": "first",
    "left": "right", "right": "left",
    "up": "down", "down": "up",
    "increase": "decrease", "decrease": "increase",
    "rise": "fall", "fall": "rise",
    "gain": "loss", "loss": "gain",
    "success": "failure", "failure": "success",
    "victory": "defeat", "defeat": "victory",
    "majority": "minority", "minority": "majority",
    "presence": "absence", "absence": "presence",
    "strength": "weakness", "weakness": "strength",
    "truth": "lie", "lie": "truth",
    "peace": "war", "war": "peace",
    "day": "night", "night": "day",
    "life": "death", "death": "life",
    "beginning": "end", "all": "none", "none": "all",
}


def get_antonym(word, wn_pos):
    """Return a random antonym for word, or None if not found."""
    antonyms = set()
    for syn in wordnet.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace("_", " "))
    antonyms.discard(word)
    return random.choice(list(antonyms)) if antonyms else None


def negate_sentence(tokens):
    """Insert 'not' after the first auxiliary verb, or remove existing 'not'/'never'.
    Returns modified token list, or None if no suitable position found."""
    lower = [t.lower() for t in tokens]

    # if already negated, remove the negation instead
    for i, t in enumerate(lower):
        if t in {"not", "never", "n't"}:
            return tokens[:i] + tokens[i + 1:]

    # insert "not" after the first auxiliary
    for i, t in enumerate(lower):
        if t in AUXILIARIES:
            return tokens[:i + 1] + ["not"] + tokens[i + 1:]

    return None


def apply_swap(result, idx, antonym):
    """Apply an antonym swap at position idx and fix a/an agreement."""
    original = result[idx]
    if original[0].isupper():
        antonym = antonym.capitalize()
    result[idx] = antonym
    if idx > 0 and result[idx - 1].lower() in {"a", "an"}:
        if antonym[0].lower() in "aeiou":
            result[idx - 1] = "an" if result[idx - 1][0].islower() else "An"
        else:
            result[idx - 1] = "a" if result[idx - 1][0].islower() else "A"
    return result


def flip_sentence(sentence, n_flips_min=N_FLIPS_MIN, n_flips_max=N_FLIPS_MAX):
    """Flip sentence meaning using a cascade of strategies:
    1. WordNet antonyms for adjectives/adverbs
    2. Hardcoded safe swaps (temporal/spatial/abstract nouns)
    3. Negation insertion/removal (fallback)
    """
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    # --- Strategy 1: WordNet adj/adv antonyms ---
    wordnet_candidates = []
    for i, (word, tag) in enumerate(tagged):
        if tag not in ELIGIBLE_POS:
            continue
        ant = get_antonym(word.lower(), POS_MAP[tag])
        if ant:
            wordnet_candidates.append((i, word, ant))

    # --- Strategy 2: hardcoded safe swaps ---
    hardcoded_candidates = []
    for i, (word, _) in enumerate(tagged):
        swap = HARDCODED_SWAPS.get(word.lower())
        if swap:
            hardcoded_candidates.append((i, word, swap))

    all_candidates = wordnet_candidates + hardcoded_candidates

    if all_candidates:
        n_flips = min(random.randint(n_flips_min, n_flips_max), len(all_candidates))
        chosen = random.sample(all_candidates, n_flips)
        result = list(tokens)
        for idx, original, antonym in chosen:
            result = apply_swap(result, idx, antonym)
        return _detokenizer.detokenize(result)

    # --- Strategy 3: negation insertion/removal ---
    negated = negate_sentence(list(tokens))
    if negated:
        return _detokenizer.detokenize(negated)

    return None


def build_nli_negatives():
    print("Loading zetianli/en-wiki-paraphrased_split_4...")
    ds = load_dataset("zetianli/en-wiki-paraphrased_split_4", split="train")
    ds = ds.shuffle(seed=SEED)

    random.seed(SEED)

    pairs = []
    skipped = 0
    for example in ds:
        if len(pairs) >= N_SAMPLES:
            break
        # detokenize inputs first — dataset stores text in Treebank pre-tokenized format
        original = _detokenizer.detokenize(example["original"].split())
        paraphrase = _detokenizer.detokenize(example["paraphrase"].split())
        flipped = flip_sentence(paraphrase)
        if flipped is None:
            skipped += 1
            continue
        pairs.append((original, flipped))

    print(f"Built {len(pairs)} near-miss negative pairs. ({skipped} skipped — no antonyms found)")

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
    build_nli_negatives()
