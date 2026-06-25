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


def moi_hitrate_by_cutoff(backtest: dict | None, settings=None, metric: str = "precision") -> dict | None:
    """Hit-rate (precision) vs. cutoff year, one line per arm×mode (with CIs)."""
    if not backtest:
        return None
    cutoffs = [c for c in backtest.get("cutoffs", []) if c.get("metrics")]
    if not cutoffs:
        return None
    years = [c["cutoff_year"] for c in cutoffs]
    fig = go.Figure()
    plotted = 0
    for key, label, color, dash in _SERIES:
        xs, ys, los, his = [], [], [], []
        for c in cutoffs:
            m = c["metrics"].get(key)
            if not m:
                continue
            xs.append(c["cutoff_year"])
            ys.append(m.get(metric, 0.0))
            ci = m.get("hit_ci") or [m.get(metric, 0.0), m.get(metric, 0.0)]
            los.append(max(0.0, m.get(metric, 0.0) - ci[0]))
            his.append(max(0.0, ci[1] - m.get(metric, 0.0)))
        if not xs:
            continue
        plotted += 1
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=label,
            line=dict(color=color, dash=dash, width=3 if "steered" in key else 2),
            error_y=dict(type="data", symmetric=False, array=his, arrayminus=los, thickness=1),
            hovertemplate=f"{label} — cutoff %{{x}}: %{{y:.0%}}<extra></extra>",
        ))
    if not plotted:
        return None
    fig.update_xaxes(title="Training cutoff year (Y)", dtick=1, tickvals=years)
    fig.update_yaxes(title="Forecast hit-rate (predictions matching real post-Y papers)",
                     tickformat=".0%", rangemode="tozero")
    return _card(
        "moi_backtest", "Forecasting the field: predicting post-Y papers from ≤Y knowledge",
        "Each line trains a Model of Interestingness on papers up to year Y, has an LLM "
        "(constrained to a year-Y horizon) generate “what comes next,” and scores those "
        "predictions against the papers that actually appeared after Y. <b>Steered</b> = guided "
        "by the MOI via an evolutionary quality-diversity loop; <b>baseline</b> = unsteered. "
        "<b>Honest</b> = a genuine ≤Y-cutoff model; <b>oracle</b> = a modern model (leakage "
        "upper bound — it already knows the future). Bars are bootstrap 95% CIs; samples are "
        "small, so read the trend across cutoffs, not single points.", fig, 520)


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
