"""Leakage-free interestingness features for the MOI.

Everything here is fit on the **≤Y corpus only** (``NoveltySpace``). The single
most important guarantee is that the TF-IDF vocabulary never sees post-Y text, so
a candidate idea's novelty is measured purely against what was known by year Y.

Features deliberately stay in the sklearn + numpy stack (no torch):
- ``nn_sim`` / ``mean_topk_sim`` — nearest-neighbour and local-density similarity
  to ≤Y work (high = derivative).
- ``novelty`` = ``1 - nn_sim``.
- ``recombination`` — how many distinct ≤Y theme clusters the idea's neighbours
  span (bridging distant areas is an OMNI-style interestingness signal).
- ``concept_count`` / ``concept_rarity`` — curated concepts the idea names, and
  how rare those were in ≤Y.
- ``specificity`` / ``length`` — guards against incoherent word-salad scoring as
  "novel".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..concepts import concept_matchers

# Feature names in a fixed order — the reward model relies on this ordering.
FEATURE_NAMES = [
    "novelty", "nn_sim", "mean_topk_sim", "recombination",
    "concept_count", "concept_rarity", "specificity", "length",
]


@dataclass
class NoveltySpace:
    """A TF-IDF view of the ≤Y corpus, plus derived structure for features."""

    vectorizer: Any
    matrix: Any  # sparse TF-IDF matrix of the ≤Y corpus
    cluster_labels: list[int]
    concept_share: dict[str, float]
    vocab: set[str]
    n_docs: int
    topk: int = 5


def fit_novelty_space(train_texts: list[str], *, topk: int = 5, seed: int = 0) -> NoveltySpace | None:
    """Fit the ≤Y novelty space. Returns ``None`` if sklearn is unavailable or n<2."""
    if len(train_texts) < 2:
        return None
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:  # pragma: no cover - sklearn is a hard dep in practice
        return None

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(train_texts)
    vocab = set(vec.vocabulary_)

    n = len(train_texts)
    k = max(2, min(6, n // 3))
    try:
        labels = list(KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(matrix))
    except Exception:  # pragma: no cover - degenerate corpora
        labels = [0] * n

    matchers = concept_matchers()
    share: dict[str, float] = {}
    for label, rx in matchers.items():
        hits = sum(1 for t in train_texts if rx.search(t))
        share[label] = hits / n
    return NoveltySpace(vec, matrix, labels, share, vocab, n, topk)


def idea_features(text: str, space: NoveltySpace) -> dict[str, float]:
    """Feature dict for one candidate idea, scored against the ≤Y space."""
    from sklearn.metrics.pairwise import cosine_similarity

    vquery = space.vectorizer.transform([text or ""])
    sims = cosine_similarity(vquery, space.matrix)[0]
    order = sims.argsort()[::-1]
    topk_idx = order[: space.topk]
    nn_sim = float(sims[order[0]]) if len(sims) else 0.0
    mean_topk = float(sims[topk_idx].mean()) if len(topk_idx) else 0.0
    recombination = len({space.cluster_labels[i] for i in topk_idx}) if len(topk_idx) else 0

    matchers = concept_matchers()
    named = [label for label, rx in matchers.items() if rx.search(text or "")]
    # Rarity: reward naming concepts that were uncommon in ≤Y (1 - share), summed.
    concept_rarity = sum(1.0 - space.concept_share.get(label, 0.0) for label in named)

    toks = [t for t in (text or "").lower().split() if t.isalpha()]
    in_vocab = sum(1 for t in toks if t in space.vocab)
    specificity = (in_vocab / len(toks)) if toks else 0.0
    length = min(1.0, len(toks) / 80.0)  # normalized, capped

    return {
        "novelty": 1.0 - nn_sim,
        "nn_sim": nn_sim,
        "mean_topk_sim": mean_topk,
        "recombination": float(recombination),
        "concept_count": float(len(named)),
        "concept_rarity": float(concept_rarity),
        "specificity": specificity,
        "length": length,
    }


def feature_vector(text: str, space: NoveltySpace) -> list[float]:
    f = idea_features(text, space)
    return [f[name] for name in FEATURE_NAMES]
