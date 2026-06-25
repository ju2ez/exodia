"""Score predicted directions against the held-out future (years > Y).

Two complementary signals, both reusing existing machinery:
- **Semantic hit-rate** (reuses :func:`matching._similarity_matrix`): a predicted
  idea is a "hit" if it matches a real post-Y paper above a cosine threshold —
  precision/recall and precision@k. A conservative (lexical) lower bound.
- **Concept-level** (reuses :func:`concepts.concept_matchers`): did the agent name
  the curated concepts that actually *rose* after Y?

Comparisons: steered vs. baseline (steering lift) and honest vs. oracle (leakage
gap). Bootstrap CIs are attached because per-cutoff samples are tiny.
"""

from __future__ import annotations

from ..analysis import _entry_link, _light_text
from ..concepts import concept_matchers
from ..matching import _idea_text, _paper_text, _similarity_matrix
from ..models import Entry
from .schema import GenIdea


def _idea_dict(i: GenIdea) -> dict:
    return {"title": i.title, "name": i.name, "short_hypothesis": i.short_hypothesis,
            "abstract": i.abstract}


def _bootstrap_ci(flags: list[float], rng, n: int = 1000) -> list[float]:
    import numpy as np

    if not flags:
        return [0.0, 0.0]
    arr = np.array(flags, dtype=float)
    means = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def semantic_metrics(ideas: list[GenIdea], test_entries: list[Entry], settings, rng,
                     ks: tuple[int, ...] = (5, 10)) -> tuple[dict, list[dict]]:
    if not ideas or not test_entries:
        return {"n_ideas": len(ideas), "precision": 0.0, "recall": 0.0,
                "mean_best_sim": 0.0, "hit_ci": [0.0, 0.0]}, []
    thr = settings.moi_hit_threshold
    sims = _similarity_matrix([_idea_text(_idea_dict(i)) for i in ideas],
                              [_paper_text(e) for e in test_entries])
    best_j, best_sim, hit_flags = [], [], []
    for row in sims:
        j = max(range(len(test_entries)), key=lambda k: row[k])
        best_j.append(j)
        best_sim.append(float(row[j]))
        hit_flags.append(1.0 if row[j] >= thr else 0.0)

    n = len(ideas)
    hits = int(sum(hit_flags))
    matched_papers = {best_j[i] for i in range(n) if hit_flags[i]}
    metrics = {
        "n_ideas": n,
        "n_test": len(test_entries),
        "precision": round(hits / n, 3),
        "recall": round(len(matched_papers) / len(test_entries), 3),
        "mean_best_sim": round(sum(best_sim) / n, 3),
        "hit_ci": _bootstrap_ci(hit_flags, rng),
    }
    for k in ks:
        kk = min(k, n)
        topk_hits = sum(hit_flags[:kk])
        topk_papers = {best_j[i] for i in range(kk) if hit_flags[i]}
        metrics[f"precision_at_{k}"] = round(topk_hits / kk, 3) if kk else 0.0
        metrics[f"recall_at_{k}"] = round(len(topk_papers) / len(test_entries), 3)

    examples = []
    for i in range(n):
        if hit_flags[i]:
            e = test_entries[best_j[i]]
            examples.append({"idea_title": ideas[i].title, "paper_title": e.title,
                             "paper_url": _entry_link(e), "score": round(best_sim[i], 3),
                             "year": e.year})
    examples.sort(key=lambda d: d["score"], reverse=True)
    return metrics, examples


def _concept_share(entries: list[Entry]) -> dict[str, float]:
    matchers = concept_matchers()
    n = len(entries) or 1
    texts = [_light_text(e) for e in entries]
    return {label: sum(1 for t in texts if rx.search(t)) / n for label, rx in matchers.items()}


def risen_concepts(train_entries: list[Entry], test_entries: list[Entry]) -> set[str]:
    """Curated concepts whose share grew from ≤Y to >Y (and is non-trivial after)."""
    tr, te = _concept_share(train_entries), _concept_share(test_entries)
    return {c for c in te if te[c] > tr.get(c, 0.0) and te[c] >= 0.1}


def concept_metrics(ideas: list[GenIdea], train_entries: list[Entry],
                    test_entries: list[Entry]) -> tuple[dict, set[str], set[str]]:
    risen = risen_concepts(train_entries, test_entries)
    named: set[str] = set()
    for i in ideas:
        named.update(i.concepts)
    hit = named & risen
    metrics = {
        "concept_recall": round(len(hit) / len(risen), 3) if risen else 0.0,
        "concept_precision": round(len(hit) / len(named), 3) if named else 0.0,
        "n_risen": len(risen),
    }
    return metrics, risen, named
