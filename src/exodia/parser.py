"""Parse the upstream awesome-open-ended README markdown into Entry records.

The upstream list is hand-edited markdown. Entries look like::

    * **Title** <br>
    *Author A, Author B* <br>
    Venue, Year. [[Paper]](url) [[Code]](url) [[Website]](url)

and blogs/videos use a shorter one-segment form::

    * **Title** <br> *Author*. Year. [[Blog]](url)

We split the document into sections by header, group each bullet (which may span
several physical lines joined by ``<br>``) into a block, then extract fields by
pattern rather than by fixed segment positions — so both shapes parse, and a
malformed entry is skipped (and logged) rather than crashing the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corrections import corrected_arxiv_id
from .logging_setup import get_logger
from .models import Entry
from .util import arxiv_id_from_links, content_hash, normalize_title, stable_id

log = get_logger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_BULLET_RE = re.compile(r"^\s*[*\-+]\s+\S")
_LEADING_BULLET_RE = re.compile(r"^\s*[*\-+]\s+")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]\(([^)]+)\)")
_TAG_RE = re.compile(r"<[^>]+>")
_MDLINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_TITLE_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_IGNORE_HEADERS = {
    "table of contents",
    "contents",
    "contributing",
    "contribution",
    "contributions",
    "how to contribute",
    "acknowledgement",
    "acknowledgements",
    "acknowledgments",
    "citation",
    "license",
    "star history",
}


def categorize(header_text: str) -> tuple[str, str | None]:
    """Map a header to (action, category).

    action is one of: SET (start a category), IGNORE (non-entry section, e.g.
    Table of Contents), or KEEP (unrecognized header — treat as a subsection of
    the current section and leave the active category unchanged).
    """
    t = _TAG_RE.sub("", header_text).strip().lower()
    t = t.strip("# ").strip()
    if any(t == h or h in t for h in _IGNORE_HEADERS):
        return ("IGNORE", None)
    if "safety" in t:
        return ("SET", "safety")
    if "survey" in t or "perspective" in t:
        return ("SET", "surveys")
    if "blog" in t or "hack" in t:
        return ("SET", "blogs")
    if "video" in t or "talk" in t:
        return ("SET", "videos")
    if "paper" in t:
        return ("SET", "papers")
    return ("KEEP", None)


def clean_inline(s: str) -> str:
    """Strip inline markdown/HTML to plain text and collapse whitespace."""
    s = _TAG_RE.sub(" ", s)
    s = _MDLINK_RE.sub(r"\1", s)
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_authors(raw: str) -> list[str]:
    if not raw:
        return []
    raw = raw.replace(" and ", ", ").replace(" & ", ", ")
    parts = [p.strip(" .") for p in raw.split(",")]
    return [p for p in parts if p]


@dataclass
class ParseStats:
    total_bullets: int = 0
    parsed: int = 0
    skipped: int = 0
    duplicates: int = 0


def _split_blocks(lines: list[str]) -> list[list[str]]:
    """Group physical lines into per-bullet blocks within one section."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if _BULLET_RE.match(line):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            if line.strip():
                current.append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def parse_entry(block_lines: list[str], category: str, source_repo: str) -> Entry | None:
    """Parse one bullet block into an Entry, or None if it is not an entry."""
    text = " ".join(line.strip() for line in block_lines)
    text = _LEADING_BULLET_RE.sub("", text, count=1)
    text = _BR_RE.sub(" ", text)

    links: dict[str, str] = {}
    for m in _LINK_RE.finditer(text):
        url = m.group(2).strip()
        # Upstream link targets become clickable hrefs on the site — allow only
        # web schemes so a hostile upstream edit can't inject javascript: links.
        if not url.lower().startswith(("http://", "https://")):
            continue
        links[m.group(1).strip().lower()] = url
    text_nolinks = _LINK_RE.sub("", text)

    mt = _TITLE_RE.search(text_nolinks)
    if not mt:
        return None  # not an entry (e.g. a section note or TOC link)
    title = clean_inline(mt.group(1))
    if not title:
        return None
    rest = text_nolinks[mt.end():]

    ma = _ITALIC_RE.search(rest)
    authors_raw = clean_inline(ma.group(1)) if ma else ""
    tail = rest[ma.end():] if ma else rest

    my = _YEAR_RE.search(tail)
    year = int(my.group(1)) if my else None

    pre = tail[: my.start()] if my else tail
    venue = clean_inline(pre).strip(" .,–—-|")
    venue = venue.rstrip(",").strip() or None
    if venue and len(venue) > 80:  # guard against catching prose, not a venue
        venue = None

    authors = split_authors(authors_raw)
    arxiv_id = arxiv_id_from_links(links)
    # Fix known upstream link errors (wrong arXiv id -> wrong abstract/citations).
    fixed = corrected_arxiv_id(title, arxiv_id)
    if fixed != arxiv_id:
        arxiv_id = fixed
        if links.get("paper"):  # keep the displayed link pointing at the right paper
            links["paper"] = f"https://arxiv.org/abs/{fixed}"
    entry_id = stable_id(category, normalize_title(title))
    ch = content_hash(
        title,
        ", ".join(authors),
        venue or "",
        str(year or ""),
        "|".join(f"{k}={v}" for k, v in sorted(links.items())),
    )
    return Entry(
        entry_id=entry_id,
        title=title,
        authors=authors,
        authors_raw=authors_raw,
        venue=venue,
        year=year,
        category=category,
        links=links,
        arxiv_id=arxiv_id,
        content_hash=ch,
        source_repo=source_repo,
    )


def parse_readme(md: str, source_repo: str) -> tuple[list[Entry], ParseStats]:
    """Parse the full README into deduplicated Entry records."""
    stats = ParseStats()
    entries: dict[str, Entry] = {}
    category: str | None = None

    # Partition lines into (category, section-lines).
    sections: list[tuple[str | None, list[str]]] = []
    buf: list[str] = []
    for line in md.splitlines():
        hm = _HEADER_RE.match(line)
        if hm:
            sections.append((category, buf))
            buf = []
            action, value = categorize(hm.group(2))
            if action == "SET":
                category = value
            elif action == "IGNORE":
                category = None
            # KEEP: leave category unchanged
        else:
            buf.append(line)
    sections.append((category, buf))

    for cat, lines in sections:
        if cat is None:
            continue
        for block in _split_blocks(lines):
            stats.total_bullets += 1
            entry = parse_entry(block, cat, source_repo)
            if entry is None:
                stats.skipped += 1
                log.debug("Skipped non-entry bullet in %s: %r", cat, " ".join(block)[:120])
                continue
            if entry.entry_id in entries:
                stats.duplicates += 1
                continue
            entries[entry.entry_id] = entry
            stats.parsed += 1

    log.info(
        "Parsed %d entries (%d bullets, %d skipped, %d duplicates)",
        stats.parsed,
        stats.total_bullets,
        stats.skipped,
        stats.duplicates,
    )
    return list(entries.values()), stats
