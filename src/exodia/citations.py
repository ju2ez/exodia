"""Fetch citation counts for arXiv entries via the Semantic Scholar Graph API.

Cache-first (only entries without a citation count are fetched), batched, and
gently rate-limited. Stores ``citationCount`` + ``influentialCitationCount`` on
each entry so the Trends page can build citation-weighted ("impact") trends.

An optional ``S2_API_KEY`` (env) raises the rate limit; without it the public
endpoint is used. Only the integer counts are stored — factual metadata — and the
site keeps linking back to each paper, consistent with the project's compliance.
"""

from __future__ import annotations

import os
import time

import requests

from .config import Settings
from .enrich import base_id
from .logging_setup import get_logger
from .models import Entry

log = get_logger(__name__)

S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
_FIELDS = "citationCount,influentialCitationCount"
USER_AGENT = "exodia/0.1 (+https://github.com/ju2ez/exodia)"


def _headers() -> dict:
    h = {"User-Agent": USER_AGENT}
    key = os.environ.get("S2_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


def fetch_citation_batch(base_ids: list[str], timeout: int = 30) -> dict[str, tuple[int, int]]:
    """Fetch {base_arxiv_id: (citationCount, influentialCitationCount)} for a batch.

    Semantic Scholar's batch endpoint returns a list aligned with the input ids,
    with ``null`` for ids it doesn't know.
    """
    resp = requests.post(
        S2_BATCH,
        params={"fields": _FIELDS},
        json={"ids": [f"ARXIV:{b}" for b in base_ids]},
        headers=_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    out: dict[str, tuple[int, int]] = {}
    for b, rec in zip(base_ids, resp.json(), strict=True):
        if not rec:
            continue
        out[b] = (int(rec.get("citationCount") or 0), int(rec.get("influentialCitationCount") or 0))
    return out


def fetch_citations(entries: list[Entry], settings: Settings, batch_size: int = 100) -> int:
    """Fill in citation counts for arXiv entries lacking them. Returns count fetched."""
    todo = [e for e in entries if e.arxiv_id and e.citation_count is None]
    if not todo:
        log.info("Citations: nothing to fetch (cache hit on all arXiv entries)")
        return 0
    todo = todo[: settings.citations_max_new_fetches]

    by_base: dict[str, list[Entry]] = {}
    for e in todo:
        by_base.setdefault(base_id(e.arxiv_id), []).append(e)  # type: ignore[arg-type]

    bases = list(by_base)
    fetched = 0
    for i in range(0, len(bases), batch_size):
        if i > 0:
            time.sleep(settings.citations_request_delay_seconds)
        chunk = bases[i : i + batch_size]
        try:
            results = fetch_citation_batch(chunk)
        except Exception as ex:  # network/rate-limit hiccup: skip batch, keep going
            log.warning("Semantic Scholar batch failed (%d ids): %s", len(chunk), ex)
            continue
        for b, (cites, infl) in results.items():
            for e in by_base.get(b, []):
                e.citation_count = cites
                e.influential_citation_count = infl
                fetched += 1
    log.info("Citations: fetched %d counts via Semantic Scholar", fetched)
    return fetched
