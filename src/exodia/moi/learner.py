"""Hindsight learning: train a realization model from the backtest's own outcomes.

The Bradley-Terry reward (:mod:`reward`) learns from ≤Y *paper impact* — it never
sees the outcome of its own predictions. This module closes that gap. Every
backtest cutoff Y leaves behind generated ideas whose fate is now known: did a
paper published after Y land close to them (the same similarity match the
backtest scores with)? Those ``(features, realized?)`` pairs form the training
set for a hindsight model ``P(idea gets realized)``.

Evaluation is strictly walk-forward — train on cutoffs < C, evaluate on C — so
the reported numbers never peek at the future they are judged on. Labels and
features are recomputed deterministically from the persisted backtest and the
knowledge base: training costs **no LLM calls**.

Honest caveats, by design:

- A "realized" label means realized *within the curated list*. An idea realized
  by a paper the upstream curator never added counts as a miss, so the label
  inherits the curator's taste.
- Generic ideas match *something* later more easily than sharp bets do. The
  novelty features are part of ``phi`` precisely so the model has to trade
  novelty against hit probability; judge it against the per-arm baselines, not
  in isolation.

The model persists as plain JSON (weights + normalization) — never pickle, which
would be an arbitrary-code-execution hazard in a public repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..analysis import _light_text
from ..config import Settings
from ..logging_setup import get_logger
from ..matching import _idea_text, _paper_text, _similarity_matrix
from ..models import Entry
from ..util import read_json, write_json
from .evaluation import _idea_dict
from .features import FEATURE_NAMES, NoveltySpace, feature_vector, fit_novelty_space
from .schema import MoiBacktest
from .walkforward import split_by_cutoff

log = get_logger(__name__)


def realization_labels(idea_texts: list[str], test_entries: list[Entry],
                       settings: Settings) -> list[int]:
    """1 if the idea's best match among >Y papers clears the hit threshold."""
    if not idea_texts or not test_entries:
        return [0] * len(idea_texts)
    sims = _similarity_matrix(idea_texts, [_paper_text(e) for e in test_entries])
    return [1 if max(row) >= settings.moi_hit_threshold else 0 for row in sims]


def build_dataset(bt: MoiBacktest, entries: list[Entry], settings: Settings,
                  *, seed: int = 0) -> list[dict]:
    """Rows of ``{cutoff, arm, mode, x, y}`` from a persisted backtest.

    Features are computed against the ≤Y novelty space of the idea's own cutoff
    (leakage-free); labels against the >Y half of the corpus.
    """
    rows: list[dict] = []
    for cut in bt.cutoffs:
        train, test = split_by_cutoff(entries, cut.cutoff_year)
        if len(train) < 2 or not test or not cut.runs:
            continue
        space = fit_novelty_space([_light_text(e) for e in train], seed=seed)
        if space is None:
            continue
        for run in cut.runs:
            texts = [_idea_text(_idea_dict(i)) for i in run.ideas]
            labels = realization_labels(texts, test, settings)
            for text, y in zip(texts, labels, strict=True):
                rows.append({
                    "cutoff": cut.cutoff_year, "arm": run.arm, "mode": run.mode,
                    "x": feature_vector(text, space), "y": y,
                })
    return rows


@dataclass
class HindsightModel:
    """Logistic realization model over the MOI features, JSON-serializable."""

    feature_names: list[str]
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float
    trained_cutoffs: list[int] = field(default_factory=list)
    n_rows: int = 0
    n_pos: int = 0
    walk_forward: list[dict] = field(default_factory=list)

    def score_features(self, phi: list[float]) -> float:
        import numpy as np

        x = (np.array(phi) - np.array(self.mean)) / np.array(self.std)
        z = float(x @ np.array(self.coef) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))

    def score(self, text: str, space: NoveltySpace) -> float:
        return self.score_features(feature_vector(text, space))

    def to_dict(self) -> dict:
        return {
            "schema_version": 1, "feature_names": self.feature_names,
            "mean": self.mean, "std": self.std, "coef": self.coef,
            "intercept": self.intercept, "trained_cutoffs": self.trained_cutoffs,
            "n_rows": self.n_rows, "n_pos": self.n_pos,
            "walk_forward": self.walk_forward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HindsightModel:
        return cls(
            feature_names=d["feature_names"], mean=d["mean"], std=d["std"],
            coef=d["coef"], intercept=d["intercept"],
            trained_cutoffs=d.get("trained_cutoffs", []),
            n_rows=d.get("n_rows", 0), n_pos=d.get("n_pos", 0),
            walk_forward=d.get("walk_forward", []),
        )


def load_model(settings: Settings) -> HindsightModel | None:
    d = read_json(settings.moi_model_path, default=None)
    if not d:
        return None
    if d.get("feature_names") != FEATURE_NAMES:
        log.warning("Hindsight model features don't match current FEATURE_NAMES; ignoring it")
        return None
    return HindsightModel.from_dict(d)


def _fit(rows: list[dict]):
    """Standardize + balanced logistic regression. Returns (mean, std, coef, b)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.array([r["x"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    # Balanced: positives are rare (most guesses are never realized), and the
    # useful output is a *ranking*, not a calibrated probability.
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(Xn, y)
    return list(mean), list(std), list(clf.coef_[0]), float(clf.intercept_[0])


def _eval(rows_train: list[dict], rows_test: list[dict]) -> dict | None:
    """AUC + precision@10 of a model trained on ``rows_train``, on ``rows_test``."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    ys_tr = {r["y"] for r in rows_train}
    ys_te = {r["y"] for r in rows_test}
    if len(ys_tr) < 2 or len(ys_te) < 2:
        return None  # single-class on either side: AUC undefined
    mean, std, coef, b = _fit(rows_train)
    m = HindsightModel(FEATURE_NAMES, mean, std, coef, b)
    scores = [m.score_features(r["x"]) for r in rows_test]
    labels = [r["y"] for r in rows_test]
    order = np.argsort(scores)[::-1]
    k = min(10, len(order))
    p_at_10 = sum(labels[i] for i in order[:k]) / k if k else 0.0
    return {
        "n_train": len(rows_train), "n_test": len(rows_test),
        "base_rate": round(sum(labels) / len(labels), 3),
        "auc": round(float(roc_auc_score(labels, scores)), 3),
        "precision_at_10": round(float(p_at_10), 3),
    }


def train_hindsight(bt: MoiBacktest, entries: list[Entry], settings: Settings,
                    *, seed: int = 0) -> HindsightModel | None:
    """Walk-forward-evaluate, then fit the final model on all cutoffs.

    Returns ``None`` when the backtest doesn't support learning yet (fewer than
    two cutoffs with ideas, or labels are all one class).
    """
    rows = build_dataset(bt, entries, settings, seed=seed)
    if not rows:
        log.warning("Hindsight: backtest contains no usable idea populations")
        return None
    cutoffs = sorted({r["cutoff"] for r in rows})
    if len(cutoffs) < 2:
        log.warning("Hindsight: need >= 2 cutoffs to evaluate walk-forward (have %d)", len(cutoffs))
        return None
    if len({r["y"] for r in rows}) < 2:
        log.warning("Hindsight: labels are all one class; nothing to learn")
        return None

    walk_forward = []
    for c in cutoffs[1:]:
        tr = [r for r in rows if r["cutoff"] < c]
        te = [r for r in rows if r["cutoff"] == c]
        m = _eval(tr, te)
        if m:
            walk_forward.append({"cutoff": c, **m})

    mean, std, coef, b = _fit(rows)
    model = HindsightModel(
        feature_names=FEATURE_NAMES, mean=mean, std=std, coef=coef, intercept=b,
        trained_cutoffs=cutoffs, n_rows=len(rows), n_pos=sum(r["y"] for r in rows),
        walk_forward=walk_forward,
    )
    log.info("Hindsight: trained on %d rows (%d realized) across cutoffs %s",
             model.n_rows, model.n_pos, cutoffs)
    return model


def train_and_save(settings: Settings, *, seed: int = 0) -> HindsightModel | None:
    """CLI entry: load backtest + KB, train, persist ``data/moi_model.json``."""
    from ..store import load_entries

    raw = read_json(settings.moi_backtest_path, default=None)
    if not raw:
        log.warning("Hindsight: no backtest at %s — run `exodia moi-backtest` first",
                    settings.moi_backtest_path)
        return None
    bt = MoiBacktest.from_dict(raw)
    entries = load_entries(settings.kb_path)
    model = train_hindsight(bt, entries, settings, seed=seed)
    if model is None:
        return None
    write_json(settings.moi_model_path, model.to_dict())
    log.info("Hindsight model written -> %s", settings.moi_model_path)
    return model
