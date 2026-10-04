import datetime as dt

from tests.test_moi import _corpus

from exodia.config import Settings
from exodia.ideation import select_brief_entries
from exodia.models import Entry
from exodia.moi.archive import Archive
from exodia.moi.generator import build_live_landscape
from exodia.moi.live import load_archive_state, run_live
from exodia.moi.mock import mock_transport
from exodia.moi.schema import GenIdea
from exodia.summaries import summarize_entries


def _settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.moi_init_pop = 4
    s.moi_batch = 2
    s.moi_reward_pairs = 60
    return s


def _position(eid, cat, title, summary=None, year=2024):
    return Entry(eid, title, [], "", None, year, cat, {}, summary=summary)


def test_brief_reserves_slots_for_position_pieces():
    papers = _corpus()  # all have abstracts
    blog = _position("blog1", "blogs", "The Future of AI is Open-Ended", "Argues open-endedness is next.")
    talk = _position("vid1", "videos", "Novel Opportunities in Open-Endedness", "A talk on directions.")
    bare = _position("bare", "blogs", "No text at all")
    chosen = select_brief_entries(papers + [blog, talk, bare], limit=5, reserved=2)
    ids = [e.entry_id for e in chosen]
    assert len(chosen) == 5
    assert "blog1" in ids and "vid1" in ids  # reserved slots, despite having no abstract
    assert "bare" not in ids                  # nothing to say about it -> not reserved

    landscape = build_live_landscape(papers + [blog], 2026, limit=5)
    assert "Argues open-endedness is next." in landscape


def test_archive_roundtrip():
    a = Archive(n_bins=3)
    a.add(GenIdea("i1", "n", "Title one", "h", "abs", bd=(1, 2), fitness=0.4))
    b = Archive.from_dict(a.to_dict())
    assert list(b.cells) == [(1, 2)]
    assert b.cells[(1, 2)].title == "Title one"


def test_run_live_persists_and_resumes(tmp_path):
    s = _settings(tmp_path)
    corpus = _corpus()
    st1 = run_live(corpus, s, transport=mock_transport, today=dt.date(2026, 10, 1), generations=1)
    assert s.idea_archive_path.exists()
    assert st1["generation"] == 1 and len(st1["history"]) == 1
    assert st1["archive"]["elites"], "seeded archive should hold elites"

    st2 = run_live(corpus, s, transport=mock_transport, today=dt.date(2026, 10, 2), generations=2)
    assert st2["generation"] == 3                     # lineage keeps counting across runs
    assert [h["date"] for h in st2["history"]] == ["2026-10-01", "2026-10-02"]
    assert len(st2["archive"]["elites"]) >= len(st1["archive"]["elites"])  # never shrinks
    assert load_archive_state(s)["generation"] == 3


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, **kw):
        self.calls += 1
        return "A short summary of the work."


def test_summaries_only_for_abstractless_entries_with_text(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    paper = _corpus()[0]                       # has an abstract -> skipped
    blog = _position("b", "blogs", "Blog with page")
    blog.links = {"url": "https://example.org/post"}
    bare = _position("x", "blogs", "Blog without reachable text")
    monkeypatch.setattr("exodia.summaries._page_text",
                        lambda url, timeout=20: "word " * 100 if "example.org" in url else "")
    client = _FakeClient()
    n = summarize_entries([paper, blog, bare], s, client=client)
    assert n == 1 and client.calls == 1
    assert blog.summary == "A short summary of the work." and blog.summary_source == "page"
    assert paper.summary is None and bare.summary is None


def test_call_budget_stops_calls_and_live_keeps_partial_progress(tmp_path, monkeypatch):
    import pytest

    from exodia.moi.generator import spec_for_model_id
    from exodia.moi.llm import CallBudget, CallBudgetExhausted, LLMClient

    budget = CallBudget(2)
    c = LLMClient(spec_for_model_id("gemini-x"), tmp_path / "cache",
                  transport=lambda spec, payload: "[]", budget=budget)
    c.complete("s", "u1")
    c.complete("s", "u2")
    c.complete("s", "u1")  # cache hit: free
    with pytest.raises(CallBudgetExhausted):
        c.complete("s", "u3")

    # Live run whose budget runs out mid-loop still persists the seeded archive.
    s = _settings(tmp_path)
    s.moi_max_calls_per_run = 2
    import exodia.moi.live as live

    real = live.LLMClient
    monkeypatch.setattr(live, "LLMClient",
                        lambda *a, **kw: real(*a, **{**kw, "transport": mock_transport}))
    st = run_live(_corpus(), s, today=dt.date(2026, 10, 5), generations=3)
    assert st["archive"]["elites"] and st["history"][-1]["llm_calls"] <= 2
