"""The Agent of Interestingness: an evolutionary quality-diversity steering loop.

Generator proposes → MOI scores (fitness) → MAP-Elites archive keeps the diverse
best → the agent samples elites as parents and asks the generator to mutate /
recombine toward under-filled niches. Fitness subtracts ensemble disagreement
(``λ·std``) so the loop resists reward-hacking a single MOI blind spot.

``run_steered`` returns elites ranked by MOI fitness. ``run_baseline`` makes a
matched number of generations with NO MOI selection and NO archive guidance, in
generation order — so a steered-vs-baseline precision@k gap isolates the value of
the steering itself.
"""

from __future__ import annotations

from ..config import Settings
from ..logging_setup import get_logger
from ..util import normalize_title
from .archive import Archive, behavior_descriptor
from .features import idea_features
from .generator import Generator
from .reward import RewardModel
from .schema import GenIdea

log = get_logger(__name__)


def _score(idea: GenIdea, reward: RewardModel, lam: float) -> GenIdea:
    text = " ".join([idea.title, idea.short_hypothesis, idea.abstract])
    feats = idea_features(text, reward.space)
    mean, std = reward.score_with_std(text)
    idea.fitness = round(mean - lam * std, 4)
    idea.bd = behavior_descriptor(idea.concepts, feats["novelty"])
    return idea


def _dedup(ideas: list[GenIdea], seen_titles: set[str]) -> list[GenIdea]:
    out = []
    for i in ideas:
        key = normalize_title(i.title)
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        out.append(i)
    return out


def run_steered(generator: Generator, reward: RewardModel, settings: Settings, rng, *,
                archive: Archive | None = None, generations: int | None = None,
                seen: set[str] | None = None, start_generation: int = 0) -> list[GenIdea]:
    """Run the QD loop and return elites ranked by fitness.

    The backtest starts from an empty archive. The live loop passes its persisted
    ``archive`` (already re-scored), the titles it has ever seen (so dropped ideas
    are not re-proposed), and ``start_generation`` so lineage keeps counting up.
    Only an empty archive is seeded with a fresh population.
    """
    archive = archive if archive is not None else Archive(n_bins=settings.moi_archive_shape[1])
    seen = seen if seen is not None else set()
    n_gens = settings.moi_generations if generations is None else generations

    if not archive.cells:
        pop = _dedup(generator.propose(settings.moi_init_pop), seen)
        for idea in pop:
            archive.add(_score(idea, reward, settings.moi_fitness_lambda))

    from .archive import FAMILY_NAMES
    for gen in range(start_generation + 1, start_generation + n_gens + 1):
        parents = archive.sample_parents(rng, settings.moi_batch)
        children: list[GenIdea] = []
        for n, parent in enumerate(parents):
            if n % 3 == 0 and len(parents) > 1:  # occasional crossover
                mate = parents[(n + 1) % len(parents)]
                children += generator.crossover(parent, mate, gen, variant=gen * 100 + n + 1)
            else:
                hint = FAMILY_NAMES[(parent.bd[0] + 1) % len(FAMILY_NAMES)]
                children += generator.mutate(parent, hint, gen, variant=gen * 100 + n + 1)
        for idea in _dedup(children, seen):
            archive.add(_score(idea, reward, settings.moi_fitness_lambda))
        log.info("MOI steer gen %d (%d this run): %s", gen, n_gens, archive.stats())

    return sorted(archive.elites(), key=lambda i: i.fitness, reverse=True)


def run_baseline(generator: Generator, reward: RewardModel, settings: Settings, rng,
                 n_target: int) -> list[GenIdea]:
    """Matched-budget, unsteered generation: propose in batches, NO MOI selection."""
    seen: set[str] = set()
    out: list[GenIdea] = []
    batch = max(1, settings.moi_init_pop)
    batch_no = 0
    while len(out) < n_target:
        fresh = _dedup(generator.propose(batch, batch=batch_no), seen)
        batch_no += 1
        if not fresh:
            break
        out += fresh
    # Score for the record (so the chart can show MOI-vs-real correlation), but
    # keep generation order — selection is deliberately NOT MOI-guided.
    for idea in out:
        _score(idea, reward, settings.moi_fitness_lambda)
    return out[:n_target]
