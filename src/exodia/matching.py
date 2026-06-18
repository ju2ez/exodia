"""Detect when a newly-added upstream paper essentially realizes a prior idea.

When the curator adds a paper that closely matches a previously generated idea,
we mark that idea with a star and link it to the realizing paper. Similarity is
TF-IDF cosine over the idea text (title + hypothesis + abstract + related work)
vs. the paper text (title + abstract), reusing the same dependency-light stack as
the theme analysis; without scikit-learn it degrades to token-overlap (Jaccard).
"""

from __future__ import annotations

from .analysis import _entry_link
from .logging_setup import get_logger
from .models import Entry

log = get_logger(__name__)

DEFAULT_THRESHOLD = 0.18


def _idea_text(idea: dict) -> str:
    return " ".join(
        filter(None, [
            idea.get("title", ""), idea.get("name", ""),
            idea.get("short_hypothesis", ""), idea.get("abstract", ""),
            idea.get("related_work", ""),
        ])
    )


def _paper_text(e: Entry) -> str:
    return " ".join(filter(None, [e.title, e.abstract or ""]))


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _similarity_matrix(a_texts: list[str], b_texts: list[str]):
    """Return an len(a) x len(b) similarity matrix (TF-IDF cosine, else Jaccard)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:  # pragma: no cover - exercised only without sklearn
        return [[_jaccard(a, b) for b in b_texts] for a in a_texts]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(a_texts + b_texts)
    A, B = X[: len(a_texts)], X[len(a_texts):]
    return cosine_similarity(A, B)


def filter_novel(new_ideas: list, existing: list[dict], threshold: float = 0.82) -> list:
    """Drop newly generated ideas that essentially duplicate an existing one.

    Compares each candidate against the existing corpus *and* against earlier
    candidates in the same batch, so a weekly run that re-proposes a known idea
    adds nothing. Content-based idea ids already dedupe exact repeats; this also
    catches near-duplicates phrased slightly differently.
    """
    if not new_ideas:
        return []
    kept: list = []
    kept_texts = [_idea_text(d) for d in existing]
    for idea in new_ideas:
        text = _idea_text(idea.to_dict())
        if kept_texts:
            row = _similarity_matrix([text], kept_texts)[0]
            if max((float(x) for x in row), default=0.0) >= threshold:
                log.info("Dropping non-novel idea '%s' (>= %.2f similar to a prior idea)",
                         getattr(idea, "name", "?"), threshold)
                continue
        kept.append(idea)
        kept_texts.append(text)
    return kept


def match_ideas_to_papers(
    ideas: list[dict], new_entries: list[Entry], threshold: float = DEFAULT_THRESHOLD
) -> int:
    """Mark ideas realized by a newly-added paper (mutates dicts). Returns count newly marked."""
    candidates = [e for e in new_entries if (e.abstract or e.title)]
    pending = [i for i in ideas if not i.get("realized")]
    if not pending or not candidates:
        return 0

    sims = _similarity_matrix([_idea_text(i) for i in pending], [_paper_text(e) for e in candidates])
    marked = 0
    for row, idea in zip(sims, pending, strict=True):
        best_j = max(range(len(candidates)), key=lambda j: row[j])
        score = float(row[best_j])
        if score >= threshold:
            e = candidates[best_j]
            idea.update(
                realized=True,
                realized_paper_title=e.title,
                realized_paper_url=_entry_link(e),
                realized_entry_id=e.entry_id,
                realized_score=round(score, 3),
            )
            marked += 1
            log.info("Idea '%s' realized by paper '%s' (sim=%.3f)", idea.get("name"), e.title, score)
    return marked
