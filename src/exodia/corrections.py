"""Curated fixes for known upstream metadata errors.

The pipeline trusts the upstream awesome-open-ended list, but a few entries link
their ``[Paper]`` to the wrong arXiv id. That matters because the arXiv id keys
both the abstract enrichment and the Semantic Scholar citation count, so a wrong
id silently attaches another paper's abstract and citations. We apply a small,
audited set of corrections at parse time (keyed by normalized title) rather than
forking upstream's data.

Known case: *The AI Scientist-v2* links its ``[Paper]`` to arXiv ``2408.06292``,
which is actually *v1* (its ``[Code]`` link correctly points to AI-Scientist-v2).
The real v2 preprint is arXiv ``2504.08066`` — without this fix v2 inherits v1's
(much larger) citation count.
"""

from __future__ import annotations

from .util import normalize_title

# Normalized title -> the correct arXiv id (version-less).
ARXIV_ID_CORRECTIONS: dict[str, str] = {
    normalize_title(
        "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery "
        "via Agentic Tree Search"
    ): "2504.08066",
}


def corrected_arxiv_id(title: str, arxiv_id: str | None) -> str | None:
    """Return the audited arXiv id for ``title`` if one is known, else ``arxiv_id``."""
    return ARXIV_ID_CORRECTIONS.get(normalize_title(title), arxiv_id)
