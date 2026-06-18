from exodia.matching import filter_novel, match_ideas_to_papers
from exodia.models import Entry, Idea


def _entry(entry_id, title, abstract, links=None):
    return Entry(
        entry_id=entry_id, title=title, authors=[], authors_raw="", venue=None,
        year=2024, category="papers", links=links or {}, abstract=abstract,
    )


def _idea(name, **kw):
    base = dict(
        idea_id=name, name=name, title=name, short_hypothesis="", related_work="",
        abstract="", experiments="", risk_factors_and_limitations="",
        run_id="r", run_utc="2026-01-01T00:00:00Z", model="m",
    )
    base.update(kw)
    return Idea(**base)


def test_matches_idea_to_strongly_overlapping_paper():
    ideas = [
        {"idea_id": "a", "name": "open ended curriculum",
         "title": "Self-generated open-ended curriculum of goals",
         "short_hypothesis": "agents that invent their own goals sustain open-ended learning",
         "abstract": "an agent proposes goals and trains on an open-ended curriculum", "related_work": ""},
        {"idea_id": "b", "name": "unrelated",
         "title": "Quantum error correction thresholds",
         "short_hypothesis": "surface codes", "abstract": "stabilizer codes and qubits", "related_work": ""},
    ]
    new = [
        _entry("p1", "Open-ended curriculum learning via self-generated goals",
               "We train an agent that proposes its own goals to sustain open-ended curriculum learning."),
        _entry("p2", "A study of photosynthesis", "chloroplasts and light."),
    ]
    n = match_ideas_to_papers(ideas, new, threshold=0.1)
    assert n == 1
    assert ideas[0]["realized"] is True
    assert ideas[0]["realized_entry_id"] == "p1"
    assert ideas[0]["realized_score"] >= 0.1
    assert ideas[1].get("realized") in (None, False)  # unrelated idea untouched


def test_already_realized_ideas_are_skipped():
    ideas = [{"idea_id": "a", "name": "x", "title": "x", "realized": True, "abstract": "open ended"}]
    new = [_entry("p1", "open ended", "open ended")]
    assert match_ideas_to_papers(ideas, new, threshold=0.0) == 0


def test_filter_novel_drops_duplicates_keeps_new():
    existing = [{"name": "curriculum", "title": "Open-ended curriculum of self-generated goals",
                 "short_hypothesis": "", "abstract": "agents invent goals", "related_work": ""}]
    dup = _idea("curriculum2", title="Open-ended curriculum of self-generated goals",
                abstract="agents invent goals")
    fresh = _idea("novel", title="Measuring consensus drift with LLM judge ensembles",
                  abstract="aggregate many llm judgments over a corpus")
    kept = filter_novel([dup, fresh], existing, threshold=0.6)
    names = [i.name for i in kept]
    assert "novel" in names
    assert "curriculum2" not in names
