"""The live idea archive: the steering loop, run a little every day, never reset.

The backtest runs the MAP-Elites steering loop once per cutoff to *evaluate*
foresight. Here the same loop *produces* the site's ideas: an archive over
concept-family × novelty that persists in ``data/idea_archive.json`` and grows by a
few generations per run, whether or not the upstream list changed. A cell's idea
is replaced only by one the Model of Interestingness scores higher, so the archive
improves and diversifies instead of re-proposing the same direction.

Each run:
1. fit the novelty space + Bradley-Terry reward on the *current* knowledge base;
2. re-score the stored elites (the field moved, so novelty and reward moved) and
   re-bin them — collisions keep the fitter idea;
3. run ``generations`` rounds of mutate / crossover from sampled elites;
4. persist the archive plus a history row (coverage, QD-score, calls).

The LLM seed is the run date, so the content-addressed cache never replays
yesterday's answers.
"""

from __future__ import annotations

import datetime as dt

from ..analysis import _light_text
from ..config import Settings
from ..logging_setup import get_logger
from ..models import Entry
from ..util import now_utc_iso, read_json, write_json
from .archive import FAMILY_NAMES, Archive
from .features import fit_novelty_space
from .generator import Generator, build_live_landscape, build_rag_context, oracle_spec
from .llm import LLMClient
from .reward import train_reward
from .steering import _score, run_steered

log = get_logger(__name__)

HISTORY_LIMIT = 365


def load_archive_state(settings: Settings) -> dict:
    return read_json(settings.idea_archive_path, default=None) or {
        "schema_version": 1, "archive": None, "seen_titles": [], "generation": 0, "history": [],
    }


def run_live(entries: list[Entry], settings: Settings, *, transport=None,
             today: dt.date | None = None, generations: int | None = None) -> dict:
    """Advance the persisted archive by ``generations`` rounds; returns the new state."""
    import numpy as np

    today = today or dt.datetime.now(dt.UTC).date()
    seed = int(today.strftime("%Y%m%d"))
    state = load_archive_state(settings)

    texts = [_light_text(e) for e in entries]
    space = fit_novelty_space(texts, seed=0)
    reward = train_reward(entries, texts, space, n_pairs=settings.moi_reward_pairs,
                          ensemble=settings.moi_reward_ensemble, seed=0) if space else None
    if reward is None:
        log.warning("Live archive: could not fit the reward model; skipping")
        return state

    # Re-score and re-bin the stored elites against today's corpus.
    old = Archive.from_dict(state["archive"]) if state.get("archive") else None
    archive = Archive(n_bins=settings.moi_archive_shape[1])
    for idea in old.elites() if old else []:
        archive.add(_score(idea, reward, settings.moi_fitness_lambda))

    year = today.year
    landscape = build_live_landscape(entries, year)
    rag = build_rag_context(entries, landscape, space)
    client = LLMClient(oracle_spec(settings), settings.moi_cache_dir, transport=transport,
                       seed=seed, temperature=settings.moi_temperature, tag="live",
                       min_interval=0.0 if transport else settings.moi_min_call_interval)
    gen = Generator(client, year, "oracle", landscape, rag)

    seen = set(state.get("seen_titles") or [])
    start = int(state.get("generation") or 0)
    n_gens = settings.moi_live_generations if generations is None else generations
    before = {i.idea_id for i in archive.elites()}
    run_steered(gen, reward, settings, np.random.default_rng(seed), archive=archive,
                generations=n_gens, seen=seen, start_generation=start)
    new_ids = {i.idea_id for i in archive.elites()} - before

    stats = archive.stats()
    history = (state.get("history") or []) + [{
        "date": today.isoformat(), **stats, "new_elites": len(new_ids),
        "llm_calls": client.n_calls, "model": client.spec.id,
    }]
    state = {
        "schema_version": 1,
        "updated_utc": now_utc_iso(),
        "families": FAMILY_NAMES,
        "archive": archive.to_dict(),
        "seen_titles": sorted(seen),
        "generation": start + n_gens,
        "history": history[-HISTORY_LIMIT:],
    }
    write_json(settings.idea_archive_path, state)
    log.info("Live archive: %s, %d new elite(s), %d LLM call(s)", stats, len(new_ids), client.n_calls)
    return state
