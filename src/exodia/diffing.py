"""Turn a merge diff into a changelog entry and persist the changelog.

The merge step (store.merge) already produced the authoritative content diff by
comparing the freshly parsed entries against the previously committed knowledge
base. This module formats that into a human-readable, append-only changelog and
records the upstream SHA range so each run links back to the upstream diff.
"""

from __future__ import annotations

from pathlib import Path

from .logging_setup import get_logger
from .models import ChangelogEntry, Entry
from .util import now_utc_iso, read_json, write_json

log = get_logger(__name__)

_TRACKED_FIELDS = ("title", "authors", "venue", "year", "links", "abstract")


def has_changes(diff: dict[str, list]) -> bool:
    return bool(diff["added"] or diff["removed"] or diff["changed"])


def _ref(e: Entry) -> dict[str, str]:
    return {"entry_id": e.entry_id, "title": e.title, "category": e.category}


def field_changes(old: Entry, new: Entry) -> list[str]:
    return [f for f in _TRACKED_FIELDS if getattr(old, f) != getattr(new, f)]


def build_changelog_entry(
    run_id: str,
    diff: dict[str, list],
    *,
    from_sha: str | None,
    to_sha: str | None,
    compare_url: str | None,
    total_entries: int,
    new_ideas: int = 0,
    themes_changed: bool = False,
    note: str = "",
) -> ChangelogEntry:
    added = [_ref(e) for e in diff["added"]]
    removed = [_ref(e) for e in diff["removed"]]
    changed = [
        {**_ref(new), "fields": field_changes(old, new)} for old, new in diff["changed"]
    ]
    counts = {
        "added": len(added),
        "removed": len(removed),
        "changed": len(changed),
        "total_entries": total_entries,
    }
    return ChangelogEntry(
        run_id=run_id,
        run_utc=now_utc_iso(),
        from_sha=from_sha,
        to_sha=to_sha,
        upstream_compare_url=compare_url,
        added=added,
        removed=removed,
        changed=changed,
        counts=counts,
        new_ideas=new_ideas,
        themes_changed=themes_changed,
        note=note,
    )


def load_changelog(path: str | Path) -> list[dict]:
    return read_json(path, default=[]) or []


def append_changelog(path: str | Path, entry: ChangelogEntry) -> list[dict]:
    entries = load_changelog(path)
    entries.append(entry.to_dict())
    write_json(path, entries)
    log.info(
        "Changelog: +%d / -%d / ~%d (run %s)",
        entry.counts.get("added", 0),
        entry.counts.get("removed", 0),
        entry.counts.get("changed", 0),
        entry.run_id,
    )
    return entries
