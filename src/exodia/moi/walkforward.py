"""Walk-forward driver: slide the cutoff year, run all arms × modes, persist JSON.

For each cutoff Y: split the corpus into ≤Y (train) and >Y (held-out future),
fit the leakage-free MOI on ≤Y, then for each generator arm (honest/oracle) and
mode (steered/baseline) generate predictions and score them against >Y. Results
are written to ``data/moi_backtest.json`` for the site to render.
"""

from __future__ import annotations

from ..config import Settings
from ..logging_setup import get_logger
from ..models import Entry
from ..store import load_entries
from ..util import now_utc_iso, write_json
from . import evaluation as ev
from .features import fit_novelty_space
from .generator import Generator, build_landscape, build_rag_context, oracle_spec, pick_honest_model
from .llm import LLMClient
from .reward import train_reward
from .schema import CutoffResult, GenerationResult, MoiBacktest
from .steering import run_baseline, run_steered

log = get_logger(__name__)


def split_by_cutoff(entries: list[Entry], year: int) -> tuple[list[Entry], list[Entry]]:
    """(≤Y train, >Y test) by publication year; undated entries are dropped."""
    train = [e for e in entries if e.year and e.year <= year]
    test = [e for e in entries if e.year and e.year > year]
    return train, test


def _client(spec, settings: Settings, transport, seed: int) -> LLMClient:
    return LLMClient(spec, settings.moi_cache_dir, transport=transport, seed=seed,
                     temperature=settings.moi_temperature)


def run_cutoff(entries: list[Entry], year: int, settings: Settings, *, arms, modes,
               transport, seed: int) -> CutoffResult:
    import numpy as np

    train, test = split_by_cutoff(entries, year)
    res = CutoffResult(cutoff_year=year, n_train_le_y=len(train), n_test_gt_y=len(test))
    if len(train) < 6 or len(test) < 1:
        log.warning("MOI cutoff %d: train=%d test=%d — too sparse, skipping", year, len(train), len(test))
        return res

    train_texts = [ev._light_text(e) for e in train]
    space = fit_novelty_space(train_texts, seed=seed)
    reward = train_reward(train, train_texts, space, n_pairs=settings.moi_reward_pairs,
                          ensemble=settings.moi_reward_ensemble, seed=seed) if space else None
    if reward is None:
        log.warning("MOI cutoff %d: could not train reward; skipping", year)
        return res

    landscape = build_landscape(train, year)
    concept_table_named: dict[str, set[str]] = {}

    for arm in arms:
        spec = pick_honest_model(year, settings) if arm == "honest" else oracle_spec(settings)
        rag = build_rag_context(train, landscape, space) if arm == "oracle" else ""
        for mode in modes:
            client = _client(spec, settings, transport, seed)
            gen = Generator(client, year, arm, landscape, rag)
            rng = np.random.default_rng(seed)
            if mode == "steered":
                ideas = run_steered(gen, reward, settings, rng)
            else:
                ideas = run_baseline(gen, reward, settings, rng,
                                     n_target=max(10, settings.moi_init_pop))
            sem, examples = ev.semantic_metrics(ideas, test, settings, rng)
            con, risen, named = ev.concept_metrics(ideas, train, test)
            key = f"{arm}.{mode}"
            res.metrics[key] = {**sem, **con, "model": spec.id}
            res.runs.append(GenerationResult(
                arm=arm, mode=mode, model=spec.id, ideas=ideas,
                n_llm_calls=client.n_calls, n_cache_hits=client.n_cache_hits))
            for ex in examples[:5]:
                res.examples.append({**ex, "arm": arm, "mode": mode})
            concept_table_named[key] = named
            if not res.concept_table:  # build once from the risen set
                res.concept_table = [{"concept": c, "risen": True, "named_by": []} for c in sorted(risen)]
    # Fill named_by across runs.
    for row in res.concept_table:
        row["named_by"] = [k for k, named in concept_table_named.items() if row["concept"] in named]
    return res


def run_backtest(settings: Settings, *, cutoffs: list[int], arms: list[str] | None = None,
                 modes: list[str] | None = None, mock: bool = False, seed: int = 0,
                 out: str | None = None, transport=None, entries: list[Entry] | None = None) -> str:
    """Run the walk-forward backtest and write the results JSON. Returns the path."""
    arms = arms or ["honest", "oracle"]
    modes = modes or ["steered", "baseline"]
    if mock and transport is None:
        from .mock import mock_transport
        transport = mock_transport

    entries = entries if entries is not None else load_entries(settings.kb_path)
    bt = MoiBacktest(
        generated_utc=now_utc_iso(),
        config={
            "seed": seed, "mock": mock, "arms": arms, "modes": modes,
            "honest_models": settings.moi_honest_models, "oracle_model": settings.moi_oracle_model,
            "generations": settings.moi_generations, "init_pop": settings.moi_init_pop,
            "batch": settings.moi_batch, "archive_shape": list(settings.moi_archive_shape),
            "hit_threshold": settings.moi_hit_threshold,
        },
    )
    for year in sorted(cutoffs):
        log.info("MOI backtest: cutoff %d", year)
        bt.cutoffs.append(run_cutoff(entries, year, settings, arms=arms, modes=modes,
                                     transport=transport, seed=seed))

    out_path = out or str(settings.moi_backtest_path)
    write_json(out_path, bt.to_dict())
    log.info("MOI backtest written -> %s (%d cutoffs)", out_path, len(bt.cutoffs))
    return out_path
