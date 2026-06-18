"""Resolve the *real* publication venue for an entry.

arXiv is a preprint server, not a publication venue, yet the curated list often
records "arXiv" (or nothing) as the venue. The arXiv API exposes the author-
declared ``journal_ref`` and ``comment`` fields, which frequently name where the
work was actually published ("Accepted at NeurIPS 2023", "Published in Nature").

``resolve_venue`` prefers those signals, then falls back to a real-looking
curated venue, and finally labels genuinely-unpublished arXiv work as a
*preprint* rather than miscounting "arXiv" as a venue. ``venue_display`` is
stored on each entry and used by the plots and the knowledge-base page.
"""

from __future__ import annotations

import re

from .models import Entry

PREPRINT_LABEL = "Preprint (arXiv)"

# Curated/journal_ref/comment text that is really just "this is on arXiv".
_ARXIV_LIKE = re.compile(r"^\s*(ar\s?xiv|arxiv preprint|preprint|corr|abs/?)\b", re.I)

# Known venues we lift out of free-text comments, with canonical casing.
_VENUES = [
    "NeurIPS", "NIPS", "ICML", "ICLR", "AAAI", "IJCAI", "CVPR", "ICCV", "ECCV",
    "ACL", "EMNLP", "NAACL", "COLING", "COLM", "CoRL", "RSS", "ICRA", "IROS",
    "AAMAS", "GECCO", "ALIFE", "CoG", "SIGGRAPH", "KDD", "WWW", "UAI", "AISTATS",
    "TMLR", "JMLR", "PMLR", "Nature", "Science", "PNAS", "JAIR", "Cell",
]
_CANON = {v.lower(): v for v in _VENUES}
_VENUE_RE = re.compile(r"\b(" + "|".join(re.escape(v) for v in _VENUES) + r")\b", re.I)


def _is_arxiv_like(text: str | None) -> bool:
    return bool(text and _ARXIV_LIKE.match(text.strip()))


def _known_venue(text: str | None) -> str | None:
    """Pull a recognized venue acronym/name out of free text, if present."""
    if not text:
        return None
    m = _VENUE_RE.search(text)
    if not m:
        return None
    return _CANON.get(m.group(1).lower(), m.group(1))


def resolve_venue(entry: Entry) -> str:
    """Best estimate of where the work was *actually* published.

    Returns a human-readable venue string; ``PREPRINT_LABEL`` for arXiv work with
    no evidence of formal publication; or ``"Unpublished"`` for non-arXiv items
    with no venue.
    """
    # 1. journal_ref is the strongest signal of real publication.
    jr = (entry.journal_ref or "").strip()
    if jr and not _is_arxiv_like(jr):
        return (_known_venue(jr) or jr)[:80]

    # 2. The author comment often says where it was accepted/published.
    v = _known_venue(entry.comment)
    if v:
        return v

    # 3. A real-looking curated venue (anything that isn't just "arXiv").
    cur = (entry.venue or "").strip()
    if cur and not _is_arxiv_like(cur):
        return cur[:80]

    # 4. Otherwise: an unpublished preprint, or simply unknown.
    return PREPRINT_LABEL if entry.arxiv_id else "Unpublished"


def resolve_venues(entries: list[Entry]) -> None:
    """Populate ``venue_display`` on every entry in place."""
    for e in entries:
        e.venue_display = resolve_venue(e)
