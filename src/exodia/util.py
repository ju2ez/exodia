"""Small shared helpers: time, hashing, JSON IO, arXiv-id extraction."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_SEP = "\x1f"


def now_utc_iso() -> str:
    """Current UTC time as a second-resolution ISO-8601 string."""
    return (
        datetime.datetime.now(datetime.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def stable_id(*parts: str, length: int = 12) -> str:
    """Deterministic short hash of the given string parts (lowercased)."""
    joined = _SEP.join(p.strip().lower() for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def content_hash(*parts: str) -> str:
    """Longer hash over the value-bearing fields, used to detect changes."""
    joined = _SEP.join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_title(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for stable ids."""
    s = s.lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    # Atomic: a crash mid-write must never leave truncated JSON behind — this
    # state is committed to git and every later stage read-depends on it.
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#)]+)", re.IGNORECASE)


def arxiv_id_from_url(url: str) -> str | None:
    """Extract an arXiv id from an abs/pdf URL (keeps any version suffix)."""
    m = _ARXIV_RE.search(url or "")
    if not m:
        return None
    aid = m.group(1)
    aid = re.sub(r"\.pdf$", "", aid, flags=re.IGNORECASE)
    return aid.strip("/") or None


def arxiv_id_from_links(links: dict[str, str]) -> str | None:
    """Find the first arXiv id across an entry's links."""
    for key in ("paper", "website", "code", "blog", "video"):
        if key in links:
            aid = arxiv_id_from_url(links[key])
            if aid:
                return aid
    for url in links.values():
        aid = arxiv_id_from_url(url)
        if aid:
            return aid
    return None
