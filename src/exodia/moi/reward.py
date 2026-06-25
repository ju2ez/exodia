"""A Model of Interestingness: a Bradley-Terry reward model, sklearn-only.

The MOI learns, from ≤Y data alone, to rank ideas by "interestingness". The
training signal is an **age-normalized, within-window impact rank** of the ≤Y
papers (citations per year-since-publication, bucketed into terciles) — a proxy
for "was this notable by year Y", deliberately *not* raw current citation counts
(which are contaminated by future citations).

Training is a standard reward-model / Bradley-Terry formulation done with logistic
regression on feature **differences**: for a pair (i, j) with impact(i) >
impact(j), the row ``phi(i) - phi(j)`` gets label 1 (and the mirror gets 0). The
reward of a new idea is ``sigmoid(w · phi(idea))``. An ensemble over bootstrapped
pair samples yields a mean score plus a disagreement signal the steering loop uses
to resist reward-hacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..logging_setup import get_logger
from ..models import Entry
from .features import FEATURE_NAMES, NoveltySpace, feature_vector

log = get_logger(__name__)


def entry_impact(e: Entry, cutoff_year: int) -> float | None:
    """Age-normalized impact (citations per year since publication), ≤Y only."""
    if e.citation_count is None or not e.year:
        return None
    age = max(1, cutoff_year - e.year + 1)
    return e.citation_count / age


def _loo_corpus_features(space: NoveltySpace) -> list[list[float]]:
    """Leave-one-out feature vectors for the ≤Y corpus.

    A corpus paper's nearest neighbour is itself (cosine 1.0), which would make
    ``novelty`` a useless constant at train time. We mask the diagonal so each
    paper's novelty is measured against the *rest* of ≤Y — matching how a fresh
    idea is later scored.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(space.matrix)
    np.fill_diagonal(sims, -1.0)
    # Returns only the similarity-derived features [nn_sim, mean_topk, recombination];
    # the caller merges in the text-derived concept/specificity features.
    out: list[list[float]] = []
    for i in range(space.n_docs):
        row = sims[i]
        order = row.argsort()[::-1]
        topk_idx = order[: space.topk]
        nn_sim = float(row[order[0]]) if len(order) else 0.0
        nn_sim = max(0.0, nn_sim)
        mean_topk = float(np.clip(row[topk_idx], 0.0, 1.0).mean()) if len(topk_idx) else 0.0
        recombination = len({space.cluster_labels[j] for j in topk_idx}) if len(topk_idx) else 0
        out.append([nn_sim, mean_topk, float(recombination)])
    return out


@dataclass
class RewardModel:
    models: list[Any]  # fitted LogisticRegression estimators
    space: NoveltySpace

    def score(self, text: str) -> float:
        return self._mean_std(text)[0]

    def score_with_std(self, text: str) -> tuple[float, float]:
        return self._mean_std(text)

    def _mean_std(self, text: str) -> tuple[float, float]:
        import numpy as np

        phi = np.array(feature_vector(text, self.space)).reshape(1, -1)
        preds = [float(m.predict_proba(phi)[0, 1]) for m in self.models]
        arr = np.array(preds)
        return float(arr.mean()), float(arr.std())


def train_reward(
    train_entries: list[Entry], train_texts: list[str], space: NoveltySpace,
    *, n_pairs: int = 400, ensemble: int = 4, seed: int = 0,
) -> RewardModel | None:
    """Train the Bradley-Terry MOI ensemble on ≤Y data. ``None`` if too few labels."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    impacts = [entry_impact(e, max(e.year or 0 for e in train_entries)) for e in train_entries]
    labeled = [i for i, v in enumerate(impacts) if v is not None]
    if len(labeled) < 6:
        log.warning("MOI: only %d labeled ≤Y papers; skipping reward training", len(labeled))
        return None

    # Tercile buckets over the labeled impacts.
    vals = np.array([impacts[i] for i in labeled], dtype=float)
    lo, hi = np.percentile(vals, [33.3, 66.7])
    bucket = {i: (2 if impacts[i] >= hi else 1 if impacts[i] >= lo else 0) for i in labeled}

    # Full feature vectors for labeled docs: LOO similarity part + concept/specificity part.
    loo = _loo_corpus_features(space)  # [nn_sim, mean_topk, recombination] per doc
    feats: dict[int, list[float]] = {}
    for i in labeled:
        # concept/specificity/length recomputed from text; novelty from LOO nn_sim.
        ftext = feature_vector(train_texts[i], space)  # dict-order list
        nn_sim, mean_topk, recomb = loo[i]
        merged = dict(zip(FEATURE_NAMES, ftext, strict=True))
        merged["nn_sim"] = nn_sim
        merged["novelty"] = 1.0 - nn_sim
        merged["mean_topk_sim"] = mean_topk
        merged["recombination"] = recomb
        feats[i] = [merged[name] for name in FEATURE_NAMES]

    pairs = [(a, b) for a in labeled for b in labeled if bucket[a] > bucket[b]]
    if not pairs:
        return None
    rng = np.random.default_rng(seed)

    models = []
    for _ in range(ensemble):
        idx = rng.integers(0, len(pairs), size=min(n_pairs, len(pairs) * 2))
        X, y = [], []
        for p in idx:
            a, b = pairs[p % len(pairs)]
            da = np.array(feats[a]) - np.array(feats[b])
            X.append(da)
            y.append(1)
            X.append(-da)
            y.append(0)
        clf = LogisticRegression(fit_intercept=False, max_iter=1000, C=1.0)
        clf.fit(np.array(X), np.array(y))
        models.append(clf)
    log.info("MOI: trained %d-model reward ensemble on %d ≤Y pairs", ensemble, len(pairs))
    return RewardModel(models, space)
