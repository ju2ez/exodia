"""Assemble the text analyzed for each entry.

The analysis layer (themes/clusters) and the trend plots both ask this module
for an entry's text instead of reading ``title``/``abstract`` directly, so the
*whole* paper (cached extracted PDF text) and video transcripts feed the results
— not just the abstract. All of the extra text lives in git-ignored local caches
(:mod:`fulltext`, :mod:`transcripts`); nothing extra is committed.

Falls back to ``title + abstract`` when no caches are present, so behaviour is
unchanged on a fresh checkout or when full-text/transcripts are disabled.
"""

from __future__ import annotations

from .config import Settings
from .fulltext import cached_pdf_text
from .models import Entry
from .transcripts import cached_transcript


def corpus_text(entry: Entry, settings: Settings) -> str:
    """Title + abstract + (cached transcript) + (cached PDF full text)."""
    parts = [entry.title]
    if entry.abstract:
        parts.append(entry.abstract)
    if settings.transcripts_fetch:
        t = cached_transcript(entry, settings)
        if t:
            parts.append(t)
    if settings.fulltext_analyze:
        p = cached_pdf_text(entry, settings)
        if p:
            parts.append(p)
    return " ".join(p for p in parts if p)
