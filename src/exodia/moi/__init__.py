"""Agent of Interestingness (AoI) — a walk-forward forecasting backtest.

Can the *future* of open-endedness research be predicted from its *past*? This
package trains a leakage-free **Model of Interestingness** (MOI) on the corpus up
to a cutoff year Y, then has an LLM — constrained to a year-Y knowledge horizon —
generate "what comes next" research directions, *steered* by the MOI through an
evolutionary quality-diversity loop (MAP-Elites). The predicted directions are
scored against the papers that actually appeared after Y (semantic hit-rate +
concept-level), comparing a steered agent against an unsteered baseline and an
honest (genuine ≤Y cutoff) generator against an oracle (modern model, leakage
upper bound).

This is an **offline research artifact**: it is run manually (``exodia
moi-backtest``), costs LLM calls, and is never invoked by the daily pipeline. Its
results are persisted to ``data/moi_backtest.json`` and rendered as a chart +
writeup on the Trends page; the site renders that committed JSON with zero LLM
calls. See ``walkforward.run_backtest``.
"""

from __future__ import annotations

from .schema import MoiBacktest
from .walkforward import run_backtest

__all__ = ["MoiBacktest", "run_backtest"]
