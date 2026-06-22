"""Extract the full text of locally-cached PDFs for the analysis corpus.

:mod:`pdfs` downloads the PDF; this module turns it into plain text and caches
it next to the PDF as ``<base_arxiv_id>.txt``. The text is mined into the
analysis corpus (themes/clusters/trends) so the *whole paper* — not just the
abstract — shapes the results.

Like the PDFs themselves, the extracted text is git-ignored and never
redistributed (the site keeps linking back to the source). Extraction is
cache-first (entries whose ``.txt`` sidecar already exists are skipped) and the
text is truncated to ``fulltext_max_chars`` to bound memory and reference noise.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .enrich import base_id
from .logging_setup import get_logger
from .models import Entry
from .pdfs import pdf_path_for

log = get_logger(__name__)


def text_path_for(settings: Settings, arxiv_id: str) -> Path:
    """Local path for a given arXiv id's extracted text (sidecar to the PDF)."""
    return settings.pdfs_dir / f"{base_id(arxiv_id)}.txt"


def _cap_marker(settings: Settings) -> Path:
    """File recording the ``fulltext_max_chars`` the cached sidecars were cut at."""
    return settings.pdfs_dir / ".fulltext_cap"


def _invalidate_stale_sidecars(settings: Settings) -> None:
    """Drop cached ``.txt`` sidecars whose cap differs from the current setting.

    Extraction is cache-first, so a sidecar written at an old cap (e.g. 40k,
    truncated before the conclusion) would otherwise live forever — including in
    the CI corpus cache, which has no marker. When the cap changes (or the marker
    is absent but sidecars exist, i.e. a legacy cache), re-extract from the still
    cached PDFs at the new cap; the PDFs themselves are never re-downloaded.
    """
    if not settings.pdfs_dir.exists():
        return
    marker = _cap_marker(settings)
    prior = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    if prior == str(settings.fulltext_max_chars):
        return
    stale = list(settings.pdfs_dir.glob("*.txt"))
    if not stale:
        return
    for p in stale:
        p.unlink(missing_ok=True)
    log.info("Full text: cap changed (%s -> %d); re-extracting %d sidecar(s)",
             prior or "unset", settings.fulltext_max_chars, len(stale))


def _pdf_to_text(path: Path, max_chars: int) -> str:
    """Extract text from a PDF, stopping once ``max_chars`` is reached.

    Isolated (and lazily importing pypdf) so tests can monkeypatch it without the
    dependency or a real PDF.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[str] = []
    total = 0
    for page in reader.pages:
        chunk = page.extract_text() or ""
        out.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return " ".join(out)[:max_chars]


def extract_pdf_texts(entries: list[Entry], settings: Settings) -> int:
    """Extract+cache text for arXiv PDFs on disk that lack a ``.txt`` sidecar.

    Returns the number of PDFs newly extracted.
    """
    _invalidate_stale_sidecars(settings)
    extracted = 0
    for e in entries:
        if not e.arxiv_id:
            continue
        pdf = pdf_path_for(settings, e.arxiv_id)
        if not pdf.exists():
            continue
        dest = text_path_for(settings, e.arxiv_id)
        if dest.exists():  # cache-first
            continue
        try:
            text = _pdf_to_text(pdf, settings.fulltext_max_chars)
        except Exception as ex:  # a malformed PDF shouldn't break the run
            log.warning("PDF text extraction failed for %s: %s", e.arxiv_id, ex)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        extracted += 1
    # Record the cap these sidecars were cut at, so a later cap change re-extracts.
    settings.pdfs_dir.mkdir(parents=True, exist_ok=True)
    _cap_marker(settings).write_text(str(settings.fulltext_max_chars), encoding="utf-8")
    log.info("Full text: extracted %d PDF(s) into the analysis corpus", extracted)
    return extracted


def cached_pdf_text(entry: Entry, settings: Settings) -> str:
    """Return the cached extracted text for an entry's PDF (``""`` if none)."""
    if not entry.arxiv_id:
        return ""
    path = text_path_for(settings, entry.arxiv_id)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
