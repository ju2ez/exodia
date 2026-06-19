from exodia.config import Settings
from exodia.fulltext import (
    _pdf_to_text,
    cached_pdf_text,
    extract_pdf_texts,
    text_path_for,
)
from exodia.models import Entry
from exodia.pdfs import pdf_path_for


def _settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    return s


def _paper(eid, arxiv_id):
    return Entry(eid, f"Paper {eid}", [], "", None, 2023, "papers", {}, arxiv_id=arxiv_id)


def _make_pdf(settings, arxiv_id):
    p = pdf_path_for(settings, arxiv_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def test_extract_writes_sidecar_and_is_cache_first(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    _make_pdf(s, "2305.16291v3")
    calls = []
    monkeypatch.setattr(
        "exodia.fulltext._pdf_to_text",
        lambda path, mx: calls.append(path) or "extracted body text",
    )
    e = _paper("a", "2305.16291v3")
    assert extract_pdf_texts([e], s) == 1
    # Sidecar is keyed by the version-stripped base id, next to the PDF.
    assert text_path_for(s, "2305.16291").read_text() == "extracted body text"

    calls.clear()
    assert extract_pdf_texts([e], s) == 0 and calls == []  # cache-first


def test_extract_skips_entries_without_a_downloaded_pdf(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    monkeypatch.setattr("exodia.fulltext._pdf_to_text", lambda path, mx: "x")
    # No PDF on disk -> nothing to extract.
    assert extract_pdf_texts([_paper("a", "1901.01753")], s) == 0


def test_extract_survives_bad_pdf(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    _make_pdf(s, "2305.16291")

    def boom(path, mx):
        raise ValueError("corrupt")

    monkeypatch.setattr("exodia.fulltext._pdf_to_text", boom)
    assert extract_pdf_texts([_paper("a", "2305.16291")], s) == 0


def test_cached_pdf_text_reader(tmp_path):
    s = _settings(tmp_path)
    e = _paper("a", "2305.16291")
    assert cached_pdf_text(e, s) == ""
    dest = text_path_for(s, "2305.16291")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("cached full text", encoding="utf-8")
    assert cached_pdf_text(e, s) == "cached full text"


def test_pdf_to_text_truncates_to_max_chars(tmp_path, monkeypatch):
    class _Page:
        def extract_text(self):
            return "x" * 5000

    class _Reader:
        def __init__(self, path):
            self.pages = [_Page(), _Page(), _Page()]

    monkeypatch.setattr("pypdf.PdfReader", _Reader)
    out = _pdf_to_text(tmp_path / "p.pdf", max_chars=8000)
    assert len(out) == 8000  # stops once the cap is reached
