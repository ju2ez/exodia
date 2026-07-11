"""Hindsight learner: dataset building, walk-forward training, persistence."""

import json

from exodia.config import Settings
from exodia.models import Entry
from exodia.moi.features import FEATURE_NAMES, fit_novelty_space
from exodia.moi.learner import (
    HindsightModel,
    build_dataset,
    load_model,
    realization_labels,
    train_hindsight,
)
from exodia.moi.schema import CutoffResult, GenerationResult, GenIdea, MoiBacktest


def _e(eid, year, title, abstract=""):
    return Entry(eid, title, [], "", None, year, "papers", {}, abstract=abstract)


def _corpus():
    rows = [
        ("a", 2019, "Novelty search for evolution", "quality diversity map-elites novelty search"),
        ("b", 2019, "POET curricula", "paired open-ended trailblazer auto-curricula"),
        ("c", 2020, "Unsupervised environment design", "regret-based environment generation ued"),
        ("d", 2020, "Evolution strategies", "evolutionary computation genetic algorithm"),
        ("e", 2021, "Reinforcement learning agents", "reinforcement learning policy gradient"),
        ("f", 2021, "World models", "world model planning latent dynamics"),
        ("g", 2022, "LLM agents emerge", "large language model agentic in-context learning"),
        ("h", 2022, "Self-improving code", "recursive self-improvement program synthesis"),
        # the held-out "future" the 2021/2022 vintages are judged against
        ("i", 2023, "Voyager-like LLM agent", "autonomous llm agent lifelong skill minecraft"),
        ("j", 2023, "AI scientist", "automated scientific discovery llm agent pipeline"),
        ("k", 2024, "Darwin machine", "recursive self-improvement llm agent open-ended"),
    ]
    return [_e(*r) for r in rows]


def _idea(iid, title, hypothesis):
    return GenIdea(idea_id=iid, name=iid, title=title, short_hypothesis=hypothesis,
                   abstract=hypothesis)


def _hit_idea(iid):
    # Deliberately close to the post-cutoff papers -> should label as realized.
    return _idea(iid, "Autonomous LLM agents for lifelong open-ended skill discovery",
                 "an autonomous llm agent performs automated scientific discovery lifelong")


def _miss_idea(iid):
    # Deliberately far from everything published later -> should label as a miss.
    return _idea(iid, "Baroque harpsichord tuning by consensus of provincial guilds",
                 "medieval guild politics of harpsichord temperament in coastal towns")


def _backtest():
    bt = MoiBacktest(generated_utc="2026-01-01T00:00:00Z", config={})
    for year in (2021, 2022):
        cut = CutoffResult(cutoff_year=year, n_train_le_y=0, n_test_gt_y=0)
        ideas = [_hit_idea(f"h{year}{i}") for i in range(3)] + \
                [_miss_idea(f"m{year}{i}") for i in range(3)]
        cut.runs.append(GenerationResult(arm="oracle", mode="baseline", model="mock",
                                         ideas=ideas, n_llm_calls=0, n_cache_hits=0))
        bt.cutoffs.append(cut)
    return bt


def test_realization_labels_separate_hits_from_misses():
    corpus = _corpus()
    future = [e for e in corpus if e.year and e.year > 2022]
    s = Settings()
    labels = realization_labels(
        ["autonomous llm agent automated scientific discovery lifelong skill",
         "harpsichord tuning guild politics"],
        future, s)
    assert labels[0] == 1 and labels[1] == 0


def test_build_dataset_rows_carry_cutoff_features_and_labels():
    s = Settings()
    rows = build_dataset(_backtest(), _corpus(), s)
    assert len(rows) == 12  # 2 cutoffs x 6 ideas
    assert {r["cutoff"] for r in rows} == {2021, 2022}
    assert all(len(r["x"]) == len(FEATURE_NAMES) for r in rows)
    assert 0 < sum(r["y"] for r in rows) < len(rows)  # both classes present


def test_train_hindsight_walkforward_and_ranking():
    s = Settings()
    model = train_hindsight(_backtest(), _corpus(), s)
    assert model is not None
    assert model.trained_cutoffs == [2021, 2022]
    # Walk-forward eval on the later cutoff exists and beats coin-flipping on
    # this cleanly separable fixture.
    assert model.walk_forward and model.walk_forward[0]["cutoff"] == 2022
    assert model.walk_forward[0]["auc"] > 0.5
    # The model must rank a realized-style idea above an unrealizable one.
    corpus = _corpus()
    train = [e for e in corpus if e.year and e.year <= 2022]
    space = fit_novelty_space([f"{e.title} {e.abstract}" for e in train], seed=0)
    hit = model.score("autonomous llm agent automated scientific discovery", space)
    miss = model.score("harpsichord tuning guild politics of coastal towns", space)
    assert hit > miss


def test_model_json_roundtrip(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    model = train_hindsight(_backtest(), _corpus(), s)
    assert model is not None
    (tmp_path / "moi_model.json").write_text(json.dumps(model.to_dict()))
    loaded = load_model(s)
    assert loaded is not None
    phi = [0.5] * len(FEATURE_NAMES)
    assert abs(loaded.score_features(phi) - model.score_features(phi)) < 1e-9


def test_train_hindsight_needs_two_cutoffs():
    s = Settings()
    bt = _backtest()
    bt.cutoffs = bt.cutoffs[:1]
    assert train_hindsight(bt, _corpus(), s) is None


def test_load_model_rejects_stale_feature_set(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    d = HindsightModel(["old_feature"], [0.0], [1.0], [1.0], 0.0).to_dict()
    (tmp_path / "moi_model.json").write_text(json.dumps(d))
    assert load_model(s) is None
