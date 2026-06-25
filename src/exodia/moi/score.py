"""Score live ideas with the Model of Interestingness (the "close the loop" use).

Unlike the walk-forward backtest, this is *not* a forecast: it fits the MOI on the
**whole current corpus** and rates every generated idea by interestingness. It is
cheap (sklearn only — no LLM, no network), so the daily pipeline runs it and
persists ``moi_score`` (0–1) onto each idea in ``ideas.json``; the site shows it.
"""

from __future__ import annotations

from ..analysis import _light_text
from ..config import Settings
from ..logging_setup import get_logger
from ..matching import _idea_text
from ..models import Entry
from .features import fit_novelty_space
from .reward import train_reward

log = get_logger(__name__)


def score_ideas(entries: list[Entry], ideas: list[dict], settings: Settings, *, seed: int = 0) -> int:
    """Set ``moi_score`` on each idea dict (in place). Returns the number scored.

    Returns 0 (leaving ideas untouched) when the corpus is too small/unlabeled to
    train the reward model — the caller can then skip persisting.
    """
    if not ideas or len(entries) < 6:
        return 0
    train_texts = [_light_text(e) for e in entries]
    space = fit_novelty_space(train_texts, seed=seed)
    if space is None:
        return 0
    reward = train_reward(entries, train_texts, space, n_pairs=settings.moi_reward_pairs,
                          ensemble=settings.moi_reward_ensemble, seed=seed)
    if reward is None:
        return 0
    n = 0
    for idea in ideas:
        idea["moi_score"] = round(reward.score(_idea_text(idea)), 4)
        n += 1
    log.info("MOI: scored %d ideas (corpus n=%d)", n, len(entries))
    return n
