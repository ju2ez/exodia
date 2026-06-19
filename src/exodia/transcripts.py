"""Fetch video transcripts (YouTube captions) for the analysis corpus.

Talks and lectures in the curated list carry no abstract, so without this they
contribute only their title to the analysis. This module pulls the caption track
for each YouTube-hosted video entry and caches it under
``<data_dir>/transcripts/<video_id>.txt``.

Like PDFs, transcripts are kept locally for analysis only — git-ignored, never
redistributed; the site keeps linking back to the video. Fetching is cache-first
(entries already cached are skipped), capped per run, and politely rate-limited.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .config import Settings
from .logging_setup import get_logger
from .models import Entry

log = get_logger(__name__)

# youtu.be/<id>, youtube.com/watch?v=<id>, youtube.com/embed/<id>, /shorts/<id>
_YT_PATTERNS = [
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{6,})"),
    re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})"),
    re.compile(r"youtube\.com/(?:embed|shorts)/([A-Za-z0-9_-]{6,})"),
]
_PREFERRED_LANGS = ("en", "en-US", "en-GB")


def youtube_id(url: str) -> str | None:
    """Extract a YouTube video id from a URL, or None if it isn't a YouTube URL."""
    for rx in _YT_PATTERNS:
        m = rx.search(url or "")
        if m:
            return m.group(1)
    return None


def _video_id(entry: Entry) -> str | None:
    for url in (entry.links or {}).values():
        vid = youtube_id(url)
        if vid:
            return vid
    return None


def transcript_path(settings: Settings, video_id: str) -> Path:
    return settings.transcripts_dir / f"{video_id}.txt"


def _fetch_segments(video_id: str) -> str:
    """Fetch and join a YouTube transcript into one string.

    Isolated (and lazily importing youtube-transcript-api) so tests can
    monkeypatch it without the dependency or network.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    data = YouTubeTranscriptApi.get_transcript(video_id, languages=_PREFERRED_LANGS)
    return " ".join(seg["text"] for seg in data if seg.get("text"))


def fetch_transcripts(entries: list[Entry], settings: Settings) -> int:
    """Fetch+cache transcripts for YouTube video entries lacking one.

    Returns the number of transcripts newly fetched.
    """
    todo: list[tuple[Entry, str, Path]] = []
    for e in entries:
        if e.category != "videos":
            continue
        vid = _video_id(e)
        if not vid:
            continue
        dest = transcript_path(settings, vid)
        if dest.exists():  # cache-first
            continue
        todo.append((e, vid, dest))

    if not todo:
        log.info("Transcripts: nothing to fetch (cache hit on all videos)")
        return 0
    todo = todo[: settings.transcripts_max_new_fetches]

    fetched = 0
    for i, (_e, vid, dest) in enumerate(todo):
        if i > 0:
            time.sleep(settings.transcripts_request_delay_seconds)
        try:
            text = _fetch_segments(vid)
        except Exception as ex:  # no captions / unavailable / network: skip, keep going
            log.warning("Transcript fetch failed for %s: %s", vid, ex)
            continue
        if not text:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        fetched += 1
    log.info("Transcripts: fetched %d video transcript(s) into the analysis corpus", fetched)
    return fetched


def cached_transcript(entry: Entry, settings: Settings) -> str:
    """Return the cached transcript for a video entry (``""`` if none)."""
    vid = _video_id(entry)
    if not vid:
        return ""
    path = transcript_path(settings, vid)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
