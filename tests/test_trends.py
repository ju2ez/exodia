from exodia.analysis import analyze
from exodia.config import Settings
from exodia.models import Entry
from exodia.plotting import (
    code_availability_over_time,
    datasets_over_time,
    make_trend_plots,
    models_over_time,
    providers_over_time,
)


def _e(eid, year, title, abstract="", links=None, cats=None):
    return Entry(eid, title, [], "", None, year, "papers", links or {},
                 abstract=abstract, arxiv_categories=cats)


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


def test_make_trend_plots_html_and_single_js():
    es = [_e(str(i), 2019 + i % 5,
             f"GPT-4 minecraft navigation code generation paper {i}",
             "open-ended agent reinforcement learning", links={"code": "x"} if i % 2 else None,
             cats=["cs.AI", "cs.LG"]) for i in range(15)]
    report = analyze(es, Settings())
    cards = make_trend_plots(es, report, Settings())
    assert cards, "expected several trend charts"
    for c in cards:
        assert "plotly-graph-div" in c["html"]
    assert sum("cdn.plot.ly" in c["html"] for c in cards) == 1  # plotly.js loaded once
