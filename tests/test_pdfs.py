from pathlib import Path

import pytest

from exodia.config import Settings
from exodia.pdfs import download_pdfs, fetch_pdf, pdf_path_for


def _settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.pdf_fetch = True
    return s


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_pdf_path_for_strips_version(tmp_path):
    s = _settings(tmp_path)
    assert pdf_path_for(s, "2305.16291v3") == s.pdfs_dir / "2305.16291.pdf"


def test_fetch_pdf_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exodia.pdfs.requests.get", lambda *a, **k: _FakeResp(b"%PDF-1.5 body")
    )
    dest = tmp_path / "p.pdf"
    fetch_pdf("1901.01753", dest)
    assert dest.read_bytes().startswith(b"%PDF")


def test_fetch_pdf_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exodia.pdfs.requests.get", lambda *a, **k: _FakeResp(b"<html>nope</html>")
    )
    with pytest.raises(ValueError):
        fetch_pdf("1901.01753", tmp_path / "p.pdf")


def test_download_pdfs_fetches_and_sets_path(entries_v1, tmp_path, monkeypatch):
    s = _settings(tmp_path)
    monkeypatch.setattr(
        "exodia.pdfs.requests.get", lambda *a, **k: _FakeResp(b"%PDF-1.4 x")
    )
    n = download_pdfs(entries_v1, s)
    assert n == 3  # POET, Voyager, Rainbow Teaming are the arXiv entries
    arxiv_entries = [e for e in entries_v1 if e.arxiv_id]
    assert all(e.pdf_path for e in arxiv_entries)
    assert all((s.pdfs_dir / Path(e.pdf_path).name).exists() for e in arxiv_entries)


def test_download_pdfs_cache_first_makes_no_calls(entries_v1, tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.pdfs_dir.mkdir(parents=True)
    for e in entries_v1:
        if e.arxiv_id:
            pdf_path_for(s, e.arxiv_id).write_bytes(b"%PDF cached")

    def boom(*a, **k):
        raise AssertionError("fetch_pdf should not download when cached")

    monkeypatch.setattr("exodia.pdfs.fetch_pdf", boom)
    assert download_pdfs(entries_v1, s) == 0
    # cache hit still backfills pdf_path so the KB points at the local file
    assert all(e.pdf_path for e in entries_v1 if e.arxiv_id)


def test_download_pdfs_respects_cap(entries_v1, tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.pdf_max_new_downloads = 1
    monkeypatch.setattr(
        "exodia.pdfs.requests.get", lambda *a, **k: _FakeResp(b"%PDF y")
    )
    assert download_pdfs(entries_v1, s) == 1
