"""Download full-text PDFs for arXiv entries.

Companion to :mod:`enrich`, which fetches only abstracts. This module fetches
the full PDF for each arXiv entry and stores it under ``<data_dir>/pdfs`` as
``<base_arxiv_id>.pdf``. Like enrichment it is cache-first (entries whose PDF is
already on disk are skipped), rate-limited, and capped per run, so steady-state
runs download nothing.

arXiv asks automated clients to be gentle: we send a descriptive User-Agent and
wait at least ``pdf_request_delay_seconds`` between downloads. PDFs are kept
locally for analysis only — they are git-ignored and never redistributed.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from .config import Settings
from .enrich import base_id
from .logging_setup import get_logger
from .models import Entry
from .paths import REPO_ROOT

log = get_logger(__name__)

ARXIV_PDF = "https://arxiv.org/pdf/{base}.pdf"
USER_AGENT = "exodia/0.1 (+https://github.com/ju2ez/exodia)"
_PDF_MAGIC = b"%PDF"


def pdf_path_for(settings: Settings, arxiv_id: str) -> Path:
    """Local destination path for a given arXiv id's PDF (version-stripped)."""
    return settings.pdfs_dir / f"{base_id(arxiv_id)}.pdf"


def _rel(path: Path) -> str:
    """Path relative to the repo root, for portable storage in the KB."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fetch_pdf(arxiv_id: str, dest: Path, timeout: int = 60) -> None:
    """Download one arXiv PDF to ``dest``. Raises on HTTP error or non-PDF body."""
    url = ARXIV_PDF.format(base=base_id(arxiv_id))
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    if not content.startswith(_PDF_MAGIC):
        raise ValueError(f"response for {arxiv_id} is not a PDF (got {content[:16]!r})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)


def download_pdfs(entries: list[Entry], settings: Settings) -> int:
    """Download full PDFs for arXiv entries lacking one. Returns count downloaded.

    Cache-first: entries whose PDF is already on disk are skipped (and have their
    ``pdf_path`` backfilled if missing). Bounded by ``pdf_max_new_downloads``.
    """
    todo: list[tuple[Entry, Path]] = []
    for e in entries:
        if not e.arxiv_id:
            continue
        dest = pdf_path_for(settings, e.arxiv_id)
        if dest.exists():
            if not e.pdf_path:  # backfill on a cache hit so the KB always points at it
                e.pdf_path = _rel(dest)
            continue
        todo.append((e, dest))

    if not todo:
        log.info("PDFs: nothing to download (cache hit on all arXiv entries)")
        return 0
    todo = todo[: settings.pdf_max_new_downloads]

    downloaded = 0
    for i, (e, dest) in enumerate(todo):
        if i > 0:
            time.sleep(settings.pdf_request_delay_seconds)
        try:
            fetch_pdf(e.arxiv_id, dest)  # type: ignore[arg-type]
        except Exception as ex:  # network/format hiccup: skip this one, keep going
            log.warning("PDF download failed for %s: %s", e.arxiv_id, ex)
            continue
        e.pdf_path = _rel(dest)
        downloaded += 1

    log.info("PDFs: downloaded %d full-text PDFs via arXiv", downloaded)
    return downloaded
