import json

from exodia.config import Settings
from exodia.models import Entry
from exodia.moi.archive import Archive, behavior_descriptor, family_of
from exodia.moi.evaluation import concept_metrics, semantic_metrics
from exodia.moi.features import FEATURE_NAMES, fit_novelty_space, idea_features
from exodia.moi.generator import parse_gen_ideas
from exodia.moi.llm import ModelSpec
from exodia.moi.plots import moi_hitrate_by_cutoff
from exodia.moi.reward import entry_impact, train_reward
from exodia.moi.schema import GenIdea, MoiBacktest
from exodia.moi.score import score_ideas
from exodia.moi.walkforward import run_backtest, split_by_cutoff


def _e(eid, year, title, abstract="", cites=None):
    return Entry(eid, title, [], "", None, year, "papers", {},
                 abstract=abstract, citation_count=cites,
                 arxiv_published=f"{year}-06-01T00:00:00Z" if year else None)


# A small dated corpus with concept-bearing abstracts + citations for the reward label.
def _corpus():
    rows = [
        ("a", 2019, "Novelty search for evolution", "quality diversity map-elites novelty search", 80),
        ("b", 2019, "POET curricula", "paired open-ended trailblazer auto-curricula", 60),
        ("c", 2020, "Unsupervised environment design", "regret-based environment generation ued", 40),
        ("d", 2020, "Evolution strategies", "evolutionary computation genetic algorithm", 20),
        ("e", 2021, "Reinforcement learning agents", "reinforcement learning policy gradient exploration", 30),
        ("f", 2021, "World models", "world model planning latent dynamics", 25),
        ("g", 2022, "LLM agents emerge", "large language model agentic in-context learning", 50),
        ("h", 2022, "Self-improving code", "recursive self-improvement program synthesis", 15),
        # post-cutoff (test) papers, with a clear LLM-agent rise
        ("i", 2023, "Voyager-like LLM agent", "autonomous llm agent lifelong skill minecraft", 200),
        ("j", 2023, "AI scientist", "automated scientific discovery llm agent", 120),
        ("k", 2024, "Darwin machine", "recursive self-improvement llm agent open-ended", 90),
        ("l", 2024, "Foundation self-play", "foundation model self-play open-ended strategy", 30),
    ]
    return [_e(*r) for r in rows]


def _settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.moi_generations = 1
    s.moi_init_pop = 4
    s.moi_batch = 2
    s.moi_reward_pairs = 60
    return s


def test_features_novelty_and_keys():
    corpus = _corpus()
    space = fit_novelty_space([f"{e.title} {e.abstract}" for e in corpus], seed=0)
    assert space is not None
    f_dup = idea_features("quality diversity map-elites novelty search", space)
    f_new = idea_features("a completely unrelated treatise on medieval poetry", space)
    assert set(f_dup) == set(FEATURE_NAMES)
    assert f_dup["nn_sim"] > f_new["nn_sim"]      # near-duplicate is less novel
    assert f_new["novelty"] > f_dup["novelty"]


def test_entry_impact_age_normalizes():
    old = _e("x", 2019, "t", "a", cites=100)
    new = _e("y", 2022, "t", "a", cites=100)
    assert entry_impact(old, 2022) < entry_impact(new, 2022)  # older paper, more years to accrue
    assert entry_impact(_e("z", 2022, "t", "a", cites=None), 2022) is None


def test_reward_trains_and_scores_in_range():
    corpus = _corpus()
    texts = [f"{e.title} {e.abstract}" for e in corpus]
    space = fit_novelty_space(texts, seed=0)
    reward = train_reward(corpus, texts, space, n_pairs=80, ensemble=3, seed=0)
    assert reward is not None
    mean, std = reward.score_with_std("open-ended llm agent that self-improves")
    assert 0.0 <= mean <= 1.0 and std >= 0.0


def test_archive_diversity_and_replacement():
    fam = family_of(["Large language models (LLMs)"])
    assert behavior_descriptor(["Large language models (LLMs)"], 0.9)[1] == 2  # high novelty bin
    arc = Archive()
    a = GenIdea("1", "a", "A", "h", "x", concepts=["Large language models (LLMs)"], bd=(fam, 2), fitness=0.3)
    b = GenIdea("2", "b", "B", "h", "x", concepts=["Large language models (LLMs)"], bd=(fam, 2), fitness=0.7)
    assert arc.add(a) is True
    assert arc.add(b) is True          # higher fitness replaces same cell
    assert arc.cells[(fam, 2)].idea_id == "2"
    assert arc.max_fitness() == 0.7


def test_llm_client_cache_offline(tmp_path):
    from exodia.moi.llm import LLMClient

    calls = {"n": 0}

    def transport(spec, payload):
        calls["n"] += 1
        return '[{"name":"x","title":"T","short_hypothesis":"h","abstract":"a"}]'

    client = LLMClient(ModelSpec("fake"), tmp_path / "cache", transport=transport)
    a = client.complete("sys", "user one")
    b = client.complete("sys", "user one")          # identical -> cache hit
    assert a == b
    assert calls["n"] == 1 and client.n_calls == 1 and client.n_cache_hits == 1


def test_parse_gen_ideas_extracts_concepts():
    raw = ('```json\n[{"name":"llm-agents","title":"Autonomous LLM agents",'
           '"short_hypothesis":"agentic in-context learning","abstract":"large language model"}]\n```')
    ideas = parse_gen_ideas(raw)
    assert len(ideas) == 1
    assert any("LLM" in c or "agent" in c.lower() for c in ideas[0].concepts)


def test_semantic_and_concept_metrics():
    corpus = _corpus()
    train, test = split_by_cutoff(corpus, 2022)
    s = Settings()
    import numpy as np
    rng = np.random.default_rng(0)
    ideas = [GenIdea("1", "v", "Autonomous LLM agent lifelong skill learning in open worlds",
                     "autonomous llm agent", "minecraft lifelong skill", concepts=["Autonomous / LLM agents"])]
    sem, examples = semantic_metrics(ideas, test, s, rng)
    assert sem["n_ideas"] == 1 and sem["precision"] >= 0.0 and "hit_ci" in sem
    con, risen, named = concept_metrics(ideas, train, test)
    assert "concept_recall" in con and "Autonomous / LLM agents" in named


def test_score_ideas_sets_moi_score(tmp_path):
    corpus = _corpus()
    s = _settings(tmp_path)
    ideas = [{"title": "Open-ended LLM agent", "name": "x", "short_hypothesis": "llm agent",
              "abstract": "recursive self-improvement", "related_work": ""}]
    n = score_ideas(corpus, ideas, s)
    assert n == 1
    assert 0.0 <= ideas[0]["moi_score"] <= 1.0


def test_walkforward_smoke_mock(tmp_path):
    s = _settings(tmp_path)
    # Mock LLM, single cutoff, tiny budget -> fully offline, deterministic.
    out = run_backtest(s, cutoffs=[2022], arms=["honest", "oracle"],
                       modes=["steered", "baseline"], mock=True, seed=0,
                       out=str(tmp_path / "moi.json"), entries=_corpus())
    data = json.loads(open(out).read())
    bt = MoiBacktest.from_dict(data)
    assert len(bt.cutoffs) == 1
    c = bt.cutoffs[0]
    assert c.cutoff_year == 2022 and c.n_test_gt_y == 4
    assert "honest.steered" in c.metrics and "oracle.baseline" in c.metrics
    assert all(r.n_llm_calls + r.n_cache_hits > 0 for r in c.runs)


def test_moi_plot_builder_and_cdn(tmp_path):
    canned = {"cutoffs": [
        {"cutoff_year": 2021, "metrics": {
            "honest.steered": {"precision": 0.3, "hit_ci": [0.1, 0.5]},
            "oracle.steered": {"precision": 0.5, "hit_ci": [0.3, 0.7]}}},
        {"cutoff_year": 2022, "metrics": {
            "honest.steered": {"precision": 0.4, "hit_ci": [0.2, 0.6]},
            "oracle.steered": {"precision": 0.6, "hit_ci": [0.4, 0.8]}}},
    ]}
    card = moi_hitrate_by_cutoff(canned, Settings())
    assert card and card["id"] == "moi_backtest"
    assert len(card["fig"].data) == 2
    assert moi_hitrate_by_cutoff(None, Settings()) is None

    # Single-CDN invariant still holds when the MOI card is merged into trend plots.
    from exodia.analysis import analyze
    from exodia.plotting import make_trend_plots
    es = [_e(str(i), 2019 + i % 5, f"open-ended rl agent paper {i}",
             "reinforcement learning open-ended", cites=10 * (i + 1)) for i in range(12)]
    report = analyze(es, Settings())
    cards = make_trend_plots(es, report, Settings(), moi=canned)
    assert sum("cdn.plot.ly" in c["html"] for c in cards) == 1
    assert any(c["id"] == "moi_backtest" for c in cards)
