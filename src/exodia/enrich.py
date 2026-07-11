"""Enrich arXiv entries with abstracts via the arXiv API.

Implemented directly on top of ``requests`` + stdlib XML parsing (no extra
dependencies). It respects arXiv's politeness guidance: a descriptive
User-Agent, a single connection, and at least ``request_delay_seconds`` between
requests. It is cache-first — only entries that have an arXiv id but no abstract
yet are fetched — so steady-state runs make essentially zero API calls.

Only abstracts + lightweight metadata are stored; full PDFs are never fetched.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

import requests

from .config import Settings
from .logging_setup import get_logger
from .models import Entry
from .util import arxiv_id_from_url

log = get_logger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "exodia/0.1 (+https://github.com/ju2ez/exodia)"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"  # journal_ref / comment / doi live here

_VERSION_RE = re.compile(r"v\d+$")
_WS_RE = re.compile(r"\s+")


@dataclass
class ArxivMeta:
    abstract: str
    abstract_url: str
    published: str | None
    categories: list[str]
    journal_ref: str | None = None
    comment: str | None = None
    doi: str | None = None


def base_id(arxiv_id: str) -> str:
    """Strip a trailing version suffix (e.g. '2305.16291v3' -> '2305.16291')."""
    return _VERSION_RE.sub("", arxiv_id)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def parse_arxiv_atom(xml_text: str) -> dict[str, ArxivMeta]:
    """Parse an arXiv Atom response into {base_arxiv_id: ArxivMeta}.

    Error entries (e.g. malformed ids) carry an id that is not an abs/pdf URL,
    so they are skipped automatically.
    """
    out: dict[str, ArxivMeta] = {}
    root = ET.fromstring(xml_text)
    for entry in root.findall(f"{_ATOM}entry"):
        id_text = entry.findtext(f"{_ATOM}id") or ""
        aid = arxiv_id_from_url(id_text)
        if not aid:
            continue
        summary = entry.findtext(f"{_ATOM}summary") or ""
        summary = _clean(summary)
        if not summary:
            continue
        base = base_id(aid)
        published = entry.findtext(f"{_ATOM}published")
        cats = [c.get("term") for c in entry.findall(f"{_ATOM}category") if c.get("term")]
        journal_ref = entry.findtext(f"{_ARXIV}journal_ref")
        comment = entry.findtext(f"{_ARXIV}comment")
        doi = entry.findtext(f"{_ARXIV}doi")
        out[base] = ArxivMeta(
            abstract=summary,
            abstract_url=f"https://arxiv.org/abs/{base}",
            published=published,
            categories=cats,
            journal_ref=_clean(journal_ref) if journal_ref else None,
            comment=_clean(comment) if comment else None,
            doi=doi.strip() if doi else None,
        )
    return out


def _chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    it = iter(seq)
    while batch := list(islice(it, size)):
        yield batch


def fetch_batch(ids: list[str], timeout: int = 30) -> dict[str, ArxivMeta]:
    """Fetch metadata for a batch of arXiv ids in one API call."""
    params = {"id_list": ",".join(ids), "max_results": str(len(ids))}
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return parse_arxiv_atom(resp.text)


def enrich_entries(entries: list[Entry], settings: Settings, batch_size: int = 25) -> int:
    """Fill in abstracts for arXiv entries lacking them. Returns count enriched."""
    todo = [e for e in entries if e.arxiv_id and not e.abstract]
    if not todo:
        log.info("Enrichment: nothing to fetch (cache hit on all arXiv entries)")
        return 0
    todo = todo[: settings.enrich_max_new_fetches]

    by_base: dict[str, list[Entry]] = {}
    for e in todo:
        by_base.setdefault(base_id(e.arxiv_id), []).append(e)  # type: ignore[arg-type]

    bases = list(by_base)
    enriched = 0
    for i, chunk in enumerate(_chunks(bases, batch_size)):
        if i > 0:
            time.sleep(settings.enrich_request_delay_seconds)
        try:
            results = fetch_batch(chunk)
        except Exception as ex:  # network/parse hiccup: skip this batch, keep going
            log.warning("arXiv batch failed (%d ids): %s", len(chunk), ex)
            continue
        for b, meta in results.items():
            for e in by_base.get(b, []):
                e.abstract = meta.abstract
                e.abstract_source = "arXiv"
                e.abstract_url = meta.abstract_url
                e.arxiv_published = meta.published
                e.arxiv_categories = meta.categories
                e.journal_ref = meta.journal_ref
                e.comment = meta.comment[:300] if meta.comment else None
                e.doi = meta.doi
                enriched += 1
    log.info("Enrichment: fetched %d abstracts via arXiv", enriched)
    return enriched
