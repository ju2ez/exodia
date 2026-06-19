from exodia.analysis import analyze
from exodia.config import Settings
from exodia.models import Entry
from exodia.plotting import (
    citation_weighted_topics_over_time,
    citations_by_year,
    code_availability_over_time,
    datasets_over_time,
    make_trend_plots,
    models_over_time,
    most_cited_papers,
    providers_over_time,
)


def _e(eid, year, title, abstract="", links=None, cats=None, cites=None):
    return Entry(eid, title, [], "", None, year, "papers", links or {},
                 abstract=abstract, arxiv_categories=cats, citation_count=cites)


def test_models_over_time_detects_families_and_versions():
    es = [
        _e("1", 2020, "A study of BERT and transformers"),
        _e("2", 2023, "Using GPT-4 for planning"),
        _e("3", 2024, "GPT-4o agents with Claude", "and Llama too"),
        _e("4", 2024, "Llama-based open-ended agent"),
    ]
    card = models_over_time(es, Settings())
    assert card is not None and card["id"] == "models_trend"
    names = {tr.name for tr in card["fig"].data}
    assert any("GPT-4" in n for n in names)   # 'gpt-4' also catches 'gpt-4o'
    assert any("Claude" in n for n in names)


def test_word_boundary_avoids_false_positives():
    # "albert" must not match the BERT keyword; "scalable" must not match a model.
    es = [_e("1", 2020, "Albert designed a scalable system"),
          _e("2", 2021, "Another scalable approach")]
    card = models_over_time(es, Settings())
    assert card is None  # no real model mentions -> nothing to plot


def test_providers_and_datasets_detect():
    es = [
        _e("1", 2021, "OpenAI gym experiments in Minecraft", cats=["cs.LG"]),
        _e("2", 2023, "DeepMind agent on Atari", cats=["cs.AI"]),
        _e("3", 2024, "Anthropic Claude on MuJoCo"),
    ]
    prov = providers_over_time(es, Settings())
    data = datasets_over_time(es, Settings())
    assert prov and {"OpenAI", "Google / DeepMind", "Anthropic"} & {t.name for t in prov["fig"].data}
    assert data and {"Minecraft / MineDojo", "Atari (ALE)", "MuJoCo"} & {t.name for t in data["fig"].data}


def test_code_availability_trend():
    es = [_e("1", 2021, "x", links={"code": "http://a"}), _e("2", 2021, "y"),
          _e("3", 2023, "z", links={"code": "http://b"})]
    card = code_availability_over_time(es, Settings())
    assert card and card["id"] == "code_availability"


def test_citation_charts_need_citation_data():
    # No citation counts -> the citation charts have nothing to plot.
    es = [_e("1", 2021, "x"), _e("2", 2023, "y")]
    assert most_cited_papers(es, Settings()) is None
    assert citation_weighted_topics_over_time(es, Settings()) is None
    assert citations_by_year(es, Settings()) is None


def test_most_cited_ranks_by_citations():
    es = [_e("1", 2021, "low", cites=5),
          _e("2", 2022, "high", cites=500),
          _e("3", 2023, "mid", cites=50)]
    card = most_cited_papers(es, Settings())
    assert card and card["id"] == "most_cited"
    ys = list(card["fig"].data[0].y)  # reversed: biggest bar on top (last)
    assert ys[-1] == "high" and ys[0] == "low"


def test_citation_weighted_topics_uses_citations():
    es = [
        _e("1", 2020, "reinforcement learning policy gradient", cites=10),
        _e("2", 2023, "a large language model agent", "llm foundation model", cites=200),
        _e("3", 2024, "another language model paper", "gpt transformer", cites=100),
    ]
    card = citation_weighted_topics_over_time(es, Settings())
    assert card and card["id"] == "citation_weighted_topics"
    assert any("LLM" in tr.name for tr in card["fig"].data)


def test_citations_by_year_median_and_mean():
    es = [_e("1", 2021, "a", cites=10), _e("2", 2021, "b", cites=30),
          _e("3", 2023, "c", cites=4)]
    card = citations_by_year(es, Settings())
    assert card and card["id"] == "citations_by_year"
    names = {tr.name for tr in card["fig"].data}
    assert {"median", "mean"} <= names


def test_trend_clips_to_2017():
    from exodia.models import ThemeReport
    from exodia.plotting import _report_since, _trend_entries
    es = [_e(str(i), 2013 + i, f"agent paper {i}") for i in range(8)]  # 2013..2020
    kept = {e.year for e in _trend_entries(es)}
    assert kept == {2017, 2018, 2019, 2020}  # pre-2017 dropped
    assert all(e.year for e in _trend_entries(es))

    report = ThemeReport(generated_utc="", method="none", n_docs=0,
                         themes_by_year={"2015": {"a": 1}, "2019": {"a": 2}})
    assert set(_report_since(report, 2017).themes_by_year) == {"2019"}


def test_make_trend_plots_html_and_single_js():
    es = [_e(str(i), 2019 + i % 5,
             f"GPT-4 minecraft navigation code generation paper {i}",
             "open-ended agent reinforcement learning", links={"code": "x"} if i % 2 else None,
             cats=["cs.AI", "cs.LG"], cites=10 * (i + 1)) for i in range(15)]
    report = analyze(es, Settings())
    cards = make_trend_plots(es, report, Settings())
    assert cards, "expected several trend charts"
    for c in cards:
        assert "plotly-graph-div" in c["html"]
    assert sum("cdn.plot.ly" in c["html"] for c in cards) == 1  # plotly.js loaded once
