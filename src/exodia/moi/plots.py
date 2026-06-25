"""Plotly builder for the MOI backtest (reads the committed JSON; no LLM)."""

from __future__ import annotations

import plotly.graph_objects as go

from ..plotting import _ACCENT, _ACCENT2, _card

_SERIES = [
    ("honest.steered", "Honest · steered", _ACCENT, "solid"),
    ("honest.baseline", "Honest · baseline", _ACCENT, "dot"),
    ("oracle.steered", "Oracle · steered", _ACCENT2, "solid"),
    ("oracle.baseline", "Oracle · baseline", _ACCENT2, "dot"),
]


def moi_hitrate_by_cutoff(backtest: dict | None, settings=None, metric: str = "mean_best_sim") -> dict | None:
    """How close predictions get to the real post-Y papers, vs. cutoff year.

    Headline metric is ``mean_best_sim`` — the mean cosine of each predicted idea to
    its nearest real post-Y paper. It's continuous and fair across arms/modes (binary
    hit-rate at a fixed threshold is too sparse for short, visionary ideas to be
    informative; precision/recall/concept-recall remain in the committed JSON).
    """
    if not backtest:
        return None
    cutoffs = [c for c in backtest.get("cutoffs", []) if c.get("metrics")]
    if not cutoffs:
        return None
    years = [c["cutoff_year"] for c in cutoffs]
    is_rate = metric in ("precision", "recall", "concept_recall")
    fig = go.Figure()
    plotted = 0
    for key, label, color, dash in _SERIES:
        xs, ys = [], []
        for c in cutoffs:
            m = c["metrics"].get(key)
            if not m:
                continue
            xs.append(c["cutoff_year"])
            ys.append(m.get(metric, 0.0))
        if not xs:
            continue
        plotted += 1
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=label,
            line=dict(color=color, dash=dash, width=3 if "steered" in key else 2),
            hovertemplate=f"{label} — cutoff %{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    if not plotted:
        return None
    fig.update_xaxes(title="Training cutoff year (Y)", dtick=1, tickvals=years)
    fig.update_yaxes(title=("Forecast hit-rate" if is_rate
                            else "Mean similarity of predictions to nearest real post-Y paper"),
                     tickformat=".0%" if is_rate else ".2f", rangemode="tozero")
    return _card(
        "moi_backtest", "Forecasting the field: how close were predictions to what actually came?",
        "Each line trains a Model of Interestingness on papers up to year Y, has an LLM "
        "(constrained to a year-Y horizon) generate “what comes next” steered by the MOI via an "
        "evolutionary quality-diversity loop, and measures how close those predictions land to the "
        "papers that <i>actually</i> appeared after Y (mean cosine to the nearest real paper). "
        "<b>Steered</b> = MOI-guided; <b>baseline</b> = unsteered. <b>Oracle</b> = a modern model "
        "fed only ≤Y context (a leakage upper bound — it already knows the future). Small samples, "
        "lexical (TF-IDF) similarity → a conservative signal; read the trend across cutoffs. "
        "Note: no genuinely ≤Y-cutoff model is available below 2024, so these are oracle "
        "reconstructions, not honest forecasts.", fig, 520)


def moi_examples(backtest: dict | None, top_n: int = 12) -> list[dict]:
    """Flatten the best correct predictions across cutoffs for the writeup table."""
    if not backtest:
        return []
    rows = []
    for c in backtest.get("cutoffs", []):
        for ex in c.get("examples", []):
            rows.append({**ex, "cutoff_year": c["cutoff_year"]})
    rows.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return rows[:top_n]
