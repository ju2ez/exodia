from pathlib import Path

from exodia.config import Settings
from exodia.enrich import ArxivMeta, base_id, enrich_entries, parse_arxiv_atom

FIX = Path(__file__).parent / "fixtures"


def test_parse_arxiv_atom_extracts_and_skips_errors():
    xml = (FIX / "arxiv_response.xml").read_text(encoding="utf-8")
    meta = parse_arxiv_atom(xml)
    assert set(meta) == {"1901.01753"}  # error entry skipped
    m = meta["1901.01753"]
    assert "Paired Open-Ended Trailblazer" in m.abstract
    assert "\n" not in m.abstract  # whitespace collapsed
    assert m.abstract_url == "https://arxiv.org/abs/1901.01753"
    assert "cs.NE" in m.categories
    assert m.published.startswith("2019")


def test_base_id():
    assert base_id("2305.16291v3") == "2305.16291"
    assert base_id("1901.01753") == "1901.01753"


def test_enrich_cache_first_makes_no_calls(entries_v1, monkeypatch):
    for e in entries_v1:
        if e.arxiv_id:
            e.abstract = "cached"

    def boom(*a, **k):
        raise AssertionError("fetch_batch should not be called when cached")

    monkeypatch.setattr("exodia.enrich.fetch_batch", boom)
    assert enrich_entries(entries_v1, Settings()) == 0


def test_enrich_uses_fetch_batch(entries_v1, monkeypatch):
    def fake_fetch(ids, timeout=30):
        return {
            b: ArxivMeta(
                abstract="A", abstract_url=f"https://arxiv.org/abs/{b}",
                published="2019", categories=["cs.NE"],
            )
            for b in ids
        }

    monkeypatch.setattr("exodia.enrich.fetch_batch", fake_fetch)
    n = enrich_entries(entries_v1, Settings())
    assert n == 3  # POET paper, Voyager, Rainbow Teaming are the arXiv entries
    assert all(e.abstract == "A" for e in entries_v1 if e.arxiv_id)
