from exodia.config import Settings
from exodia.fulltext import text_path_for
from exodia.futurework import extract_future_sections, future_work_text
from exodia.models import Entry
from exodia.plotting import future_directions_over_time, future_directions_ranked

# A realistic extracted-PDF body: intro mentions "limitations" in prose (a false
# positive trap), then a numbered conclusion/future-work heading at the end,
# then the references back-matter (which must terminate the captured section).
_INTRO = ("Open-ended learning is a broad area. Prior methods have key limitations "
          "when scaling to many tasks, as discussed below. " * 40)
_BODY = ("We present experiments across several environments and analyze results "
         "in detail over many pages of methodology and ablations. " * 80)
_FUTURE = ("\n6 Conclusion and Future Work\n"
           "We introduced a method for open-ended learning. In future work we will explore "
           "recursive self-improvement and large language model agents that design their own "
           "curricula via quality-diversity search.\n")
_REFS = "\nReferences\n[1] Some Author. A cited paper about novelty search. 2019.\n"
_PAPER = _INTRO + _BODY + _FUTURE + _REFS


def _settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    return s


def test_extract_captures_conclusion_excludes_intro_and_refs():
    out = extract_future_sections(_PAPER)
    assert "recursive self-improvement" in out  # the real future-work prose
    assert "large language model agents" in out
    assert "cited paper about novelty search" not in out  # references excluded
    assert "as discussed below" not in out  # intro false-positive excluded


def test_extract_returns_empty_without_a_heading():
    assert extract_future_sections("just some prose with no section headings " * 50) == ""
    assert extract_future_sections("") == ""


def test_future_work_text_reads_cache_and_respects_flag(tmp_path):
    s = _settings(tmp_path)
    e = Entry("1", "T", [], "", None, 2023, "papers", {}, arxiv_id="2305.16291")
    p = text_path_for(s, "2305.16291")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_PAPER, encoding="utf-8")
    assert "recursive self-improvement" in future_work_text(e, s)
    s.fulltext_analyze = False
    assert future_work_text(e, s) == ""  # disabled -> no mining


def _entry_with_future(tmp_path_settings, eid, year, future_prose, arxiv_id):
    s = tmp_path_settings
    e = Entry(eid, f"Paper {eid}", [], "", None, year, "papers", {}, arxiv_id=arxiv_id)
    p = text_path_for(s, arxiv_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("intro prose " * 60 + "\n7 Conclusion\n" + future_prose + _REFS, encoding="utf-8")
    return e


def test_future_directions_trend_and_ranked(tmp_path):
    s = _settings(tmp_path)
    es = [
        _entry_with_future(s, "1", 2019, "we will study quality-diversity and novelty search.", "1001.0001"),
        _entry_with_future(s, "2", 2023, "future work targets large language model agents.", "1001.0002"),
        _entry_with_future(s, "3", 2024, "next, llm agents with recursive self-improvement.", "1001.0003"),
    ]
    card = future_directions_over_time(es, s)
    assert card and card["id"] == "future_directions_trend"
    names = {tr.name for tr in card["fig"].data}
    assert any("LLM" in n for n in names)

    ranked = future_directions_ranked(es, s)
    assert ranked and all({"concept", "docs", "recent", "recent_share"} <= set(d) for d in ranked)
    # LLM-agent concept is flagged only in the 2023/2024 papers -> high recency.
    llm = next((d for d in ranked if "LLM" in d["concept"] or "agent" in d["concept"].lower()), None)
    assert llm and llm["recent_share"] >= 0.6


def test_future_directions_empty_without_fulltext(tmp_path):
    s = _settings(tmp_path)
    es = [Entry("1", "T", [], "", None, 2023, "papers", {}, arxiv_id="2305.16291")]
    assert future_directions_over_time(es, s) is None
    assert future_directions_ranked(es, s) == []
