"""Short summaries for entries that have no abstract (blogs, talks, position pieces).

The ideation brief ranks works by "has an abstract", so without this the curated
list's essays, blog posts and talks never reach idea generation, although they are
where the field argues for new directions. For each such entry we take the best
available source text (cached video transcript, cached PDF text, else the linked
page), ask the MOI oracle model for a 2-3 sentence summary, and persist it on the
entry (``summary`` + ``summary_source``). Summaries feed the brief only; the site
never presents them as the work's own text.

Cache-first (entries that already have a summary are skipped), capped per run, and
non-fatal: any failure leaves the entry as it was.
"""

from __future__ import annotations

import html
import re

import requests

from .config import Settings
from .fulltext import cached_pdf_text
from .logging_setup import get_logger
from .models import Entry
from .moi.generator import spec_for_model_id
from .moi.llm import LLMClient
from .transcripts import cached_transcript

log = get_logger(__name__)

MAX_SOURCE_CHARS = 8000

_SYSTEM = (
    "You summarize works from a curated reading list on open-endedness in AI. "
    "Given a work's title and source text, write 2-3 plain sentences on what it "
    "argues or contributes and why it matters for open-ended research. No preamble, "
    "no markdown, no claims that are not in the source. If the source text does not "
    "actually describe the work (boilerplate, a file listing, a login page), reply "
    "with exactly NONE."
)


def needs_summary(e: Entry) -> bool:
    # summary_source "none" = tried, the source did not describe the work: don't retry.
    return not e.abstract and not e.summary and e.summary_source != "none"


def _page_text(url: str, timeout: int = 20) -> str:
    """Visible text of a web page (crude tag strip; good enough for a summary)."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "exodia/1.0"})
    resp.raise_for_status()
    if "html" not in resp.headers.get("content-type", "html"):
        return ""
    text = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", resp.text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def source_text(e: Entry, settings: Settings) -> tuple[str, str]:
    """Best available text for an entry and where it came from."""
    t = cached_transcript(e, settings)
    if t:
        return t[:MAX_SOURCE_CHARS], "transcript"
    t = cached_pdf_text(e, settings)
    if t:
        return t[:MAX_SOURCE_CHARS], "pdf"
    for url in (e.links or {}).values():
        if not url or "youtu" in url:
            continue
        try:
            t = _page_text(url)
        except Exception as ex:
            log.info("Summary: page fetch failed for %s (%s)", url, ex)
            continue
        if len(t) > 200:
            return t[:MAX_SOURCE_CHARS], "page"
    return "", "title"


def summarize_entries(entries: list[Entry], settings: Settings, *, client: LLMClient | None = None,
                      limit: int = 15) -> int:
    """Fill ``summary`` for up to ``limit`` abstract-less entries. Returns how many."""
    todo = [e for e in entries if needs_summary(e)][:limit]
    if not todo:
        return 0
    client = client or LLMClient(spec_for_model_id(settings.summaries_model), settings.moi_cache_dir,
                                 temperature=0.2, tag="summaries",
                                 min_interval=settings.summaries_min_interval)
    done = 0
    for e in todo:
        text, src = source_text(e, settings)
        if not text:
            continue  # title alone is too thin to summarize honestly
        user = f"Title: {e.title}\nCategory: {e.category}\n\nSource text:\n{text}"
        try:
            out = client.complete(_SYSTEM, user, max_tokens=2000).strip()
        except Exception as ex:
            log.warning("Summary failed for %r: %s", e.title, ex)
            continue
        if out.strip(" .").upper() == "NONE":
            e.summary_source = "none"
        elif out:
            e.summary, e.summary_source = out[:900], src
            done += 1
    log.info("Summarized %d of %d abstract-less entries", done, len(todo))
    return done
