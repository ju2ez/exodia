"""Knowledge-base persistence and merge logic.

knowledge_base.json layout::

    {
      "schema_version": 1,
      "generated_utc": "...",
      "source": {"repo": "...", "url": "..."},
      "entries": [ {Entry}, ... ]
    }

``merge`` is the heart of the changelog: it diffs the freshly parsed entries
against the previously committed knowledge base, carries forward enrichment
(abstracts) and provenance (first_seen), and reports added/removed/changed.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .logging_setup import get_logger
from .models import CATEGORIES, KB_SCHEMA_VERSION, Entry
from .util import now_utc_iso, read_json, write_json

log = get_logger(__name__)

# Display/sort order for categories.
_CATEGORY_ORDER = {key: i for i, key in enumerate(CATEGORIES)}


def load_entries(path: str | Path) -> list[Entry]:
    data = read_json(path, default=None)
    if not data:
        return []
    return [Entry.from_dict(e) for e in data.get("entries", [])]


def save_kb(settings: Settings, entries: list[Entry]) -> None:
    payload = {
        "schema_version": KB_SCHEMA_VERSION,
        "generated_utc": now_utc_iso(),
        "source": {"repo": settings.upstream_repo, "url": settings.upstream_url},
        "entries": [e.to_dict() for e in _sorted(entries)],
    }
    write_json(settings.kb_path, payload)
    log.info("Wrote knowledge base: %d entries -> %s", len(entries), settings.kb_path)


def _sorted(entries: list[Entry]) -> list[Entry]:
    return sorted(
        entries,
        key=lambda e: (
            _CATEGORY_ORDER.get(e.category, 99),
            -(e.year or 0),
            e.title.lower(),
        ),
    )


def merge(
    existing: list[Entry],
    parsed: list[Entry],
    now: str | None = None,
) -> tuple[list[Entry], dict[str, list]]:
    """Merge parsed entries over the existing KB and return (merged, diff)."""
    now = now or now_utc_iso()
    old_by_id = {e.entry_id: e for e in existing}
    new_by_id = {e.entry_id: e for e in parsed}

    added: list[Entry] = []
    removed: list[Entry] = []
    changed: list[tuple[Entry, Entry]] = []
    merged: list[Entry] = []

    for eid, ne in new_by_id.items():
        oe = old_by_id.get(eid)
        if oe is None:
            ne.first_seen_utc = now
            ne.last_seen_utc = now
            added.append(ne)
            merged.append(ne)
            continue

        ne.first_seen_utc = oe.first_seen_utc or now
        ne.last_seen_utc = now
        # Carry forward enrichment if the freshly parsed entry lacks it.
        if not ne.abstract and oe.abstract:
            ne.abstract = oe.abstract
            ne.abstract_source = oe.abstract_source
            ne.abstract_url = oe.abstract_url
            ne.arxiv_published = oe.arxiv_published
            ne.arxiv_categories = oe.arxiv_categories
            ne.journal_ref = oe.journal_ref
            ne.comment = oe.comment
            ne.doi = oe.doi
        if not ne.arxiv_id and oe.arxiv_id:
            ne.arxiv_id = oe.arxiv_id
        # Carry forward the locally cached PDF path so we don't re-download.
        if not ne.pdf_path and oe.pdf_path:
            ne.pdf_path = oe.pdf_path
        # Carry forward fetched citation counts (refreshed by the citations step).
        if ne.citation_count is None and oe.citation_count is not None:
            ne.citation_count = oe.citation_count
            ne.influential_citation_count = oe.influential_citation_count
        if ne.content_hash != oe.content_hash:
            changed.append((oe, ne))
        merged.append(ne)

    for eid, oe in old_by_id.items():
        if eid not in new_by_id:
            removed.append(oe)

    diff = {"added": added, "removed": removed, "changed": changed}
    log.info(
        "Merge: %d total, +%d added, -%d removed, ~%d changed",
        len(merged),
        len(added),
        len(removed),
        len(changed),
    )
    return _sorted(merged), diff
