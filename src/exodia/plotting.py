"""Interactive overview charts for the site, built with Plotly.

Each builder returns a card ``{id, title, caption, fig}`` (or ``None`` when there
is nothing to plot); :func:`make_plots` turns the figures into self-contained,
responsive HTML ``<div>``s that are embedded directly into the page (no PNG
files). Plotly.js is loaded once (from the CDN) by the first chart on the page.

The charts are intentionally large and interactive — hover for exact values,
zoom, and toggle series. The venue chart reports the *real* publication venue
(``venue_display``), so arXiv-only preprints are excluded rather than miscounted.
"""

from __future__ import annotations

from collections import Counter

import plotly.graph_objects as go

from .config import Settings
from .logging_setup import get_logger
from .models import CATEGORIES, Entry, ThemeReport
from .venues import PREPRINT_LABEL, resolve_venue

log = get_logger(__name__)

# Site palette (matches assets/css/style.css).
_ACCENT = "#2f855a"
_ACCENT2 = "#2b6cb0"
_SEQ = "Viridis"
_PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
_FONT = dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=14)

# Topics tracked for the "prevalence over time" trend (title+abstract keyword match).
# Deliberately excludes "open-endedness" itself — that's ~the whole corpus.
_TREND_TOPICS = {
    "LLMs / foundation models": ["language model", "llm", "foundation model", "gpt", "transformer", "in-context", "prompt"],
    "Reinforcement learning": ["reinforcement learning", "policy gradient", "reward", "q-learning", " rl "],
    "Evolution / quality-diversity": ["evolution", "quality diversity", "quality-diversity", "novelty search", "map-elites", "genetic", "coevolution"],
    "Agents / embodiment": ["agent", "embodied", "robot", "minecraft", "manipulation"],
    "Multi-agent / self-play": ["multi-agent", "multi agent", "self-play", "self play", "population", "competitive"],
    "Generative / world models": ["generative", "diffusion", "world model", "gan", "autoencoder"],
}


def _layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=height,
        margin=dict(l=10, r=20, t=56, b=40),
        font=_FONT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(font_size=13),
        autosize=True,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False)
    return fig


def _card(card_id: str, title: str, caption: str, fig: go.Figure, height: int = 440) -> dict:
    return {"id": card_id, "title": title, "caption": caption, "fig": _layout(fig, title, height)}


def papers_per_year(entries: list[Entry], settings: Settings) -> dict | None:
    years = [e.year for e in entries if e.year]
    if not years:
        return None
    counts = Counter(years)
    xs = sorted(counts)
    fig = go.Figure(go.Bar(
        x=xs, y=[counts[y] for y in xs], marker_color=_ACCENT,
        hovertemplate="%{x}: %{y} entries<extra></extra>",
    ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="Entries")
    return _card("papers_per_year", "Entries per year",
                 "How many curated entries were published each year.", fig)


def cumulative_growth(entries: list[Entry], settings: Settings) -> dict | None:
    years = sorted(e.year for e in entries if e.year)
    if not years:
        return None
    counts = Counter(years)
    xs = sorted(counts)
    cum, total = [], 0
    for y in xs:
        total += counts[y]
        cum.append(total)
    fig = go.Figure(go.Scatter(
        x=xs, y=cum, mode="lines+markers", line=dict(color=_ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(47,133,90,0.12)",
        hovertemplate="by %{x}: %{y} entries<extra></extra>",
    ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="Cumulative entries")
    return _card("cumulative_growth", "Cumulative growth of the field",
                 "Cumulative count of entries over time — field momentum.", fig)


def category_distribution(entries: list[Entry], settings: Settings) -> dict | None:
    counts = Counter(e.category for e in entries)
    if not counts:
        return None
    labels = [CATEGORIES.get(k, k) for k in CATEGORIES if counts.get(k)]
    values = [counts[k] for k in CATEGORIES if counts.get(k)]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=_ACCENT2,
        hovertemplate="%{y}: %{x} entries<extra></extra>",
    ))
    fig.update_xaxes(title="Entries")
    fig.update_yaxes(autorange="reversed")
    return _card("category_distribution", "Entries by category",
                 "Distribution of entries across the curated categories.", fig)


def venue_distribution(entries: list[Entry], settings: Settings, top_n: int = 12) -> dict | None:
    real: Counter[str] = Counter()
    preprints = unpublished = 0
    for e in entries:
        v = e.venue_display or resolve_venue(e)
        if v == PREPRINT_LABEL:
            preprints += 1
        elif v == "Unpublished":
            unpublished += 1
        else:
            real[v] += 1
    if not real:
        return None
    common = real.most_common(top_n)
    labels = [v for v, _ in common][::-1]
    values = [c for _, c in common][::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=values, colorscale=_SEQ),
        hovertemplate="%{y}: %{x} papers<extra></extra>",
    ))
    fig.update_xaxes(title="Papers")
    height = max(360, 30 * len(labels) + 120)
    return _card(
        "venue_distribution", f"Top {len(labels)} publication venues",
        f"Where the work was actually published — arXiv is a preprint server, not a "
        f"venue, so {preprints} arXiv-only preprints"
        + (f" and {unpublished} items without a known venue" if unpublished else "")
        + " are excluded.",
        fig, height,
    )


def theme_keyword_frequency(report: ThemeReport, settings: Settings, top_n: int = 15) -> dict | None:
    kp = report.top_keyphrases[:top_n]
    if not kp:
        return None
    labels = [k["phrase"] for k in kp][::-1]
    values = [k["score"] for k in kp][::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=values, colorscale=_SEQ),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="TF-IDF weight")
    height = max(380, 28 * len(labels) + 120)
    return _card("theme_keyword_frequency", "Dominant keyphrases (consensus vocabulary)",
                 "Highest-weighted phrases across all abstracts/titles.", fig, height)


def cluster_sizes(report: ThemeReport, settings: Settings) -> dict | None:
    clusters = report.clusters
    if not clusters:
        return None
    labels = [c["label"] for c in clusters][::-1]
    values = [c["size"] for c in clusters][::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=values, colorscale=_SEQ),
        hovertemplate="%{y}: %{x} entries<extra></extra>",
    ))
    fig.update_xaxes(title="Entries in cluster")
    height = max(360, 36 * len(labels) + 120)
    return _card("cluster_sizes", "Theme clusters (the 'majority vote')",
                 "Sizes of the discovered theme clusters — what the field works on most.", fig, height)


def themes_over_time(report: ThemeReport, settings: Settings, top_k: int = 5) -> dict | None:
    tby = report.themes_by_year
    if not tby:
        return None
    phrase_totals: Counter[str] = Counter()
    for year_map in tby.values():
        for p, c in year_map.items():
            phrase_totals[p] += c
    top_phrases = [p for p, _ in phrase_totals.most_common(top_k)]
    if not top_phrases:
        return None
    years = sorted(tby, key=int)
    fig = go.Figure()
    for p in top_phrases:
        fig.add_trace(go.Scatter(
            x=years, y=[tby[y].get(p, 0) for y in years], mode="lines+markers", name=p,
            hovertemplate=f"{p} — %{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="Mentions")
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return _card("themes_over_time", "Top themes over time",
                 "How the leading themes wax and wane year over year.", fig, 480)


def abstract_coverage(entries: list[Entry], settings: Settings) -> dict | None:
    if not entries:
        return None
    with_abs = sum(1 for e in entries if e.abstract)
    without = len(entries) - with_abs
    fig = go.Figure(go.Pie(
        labels=["With abstract", "No abstract"], values=[with_abs, without], hole=0.55,
        marker=dict(colors=[_ACCENT, "#e2e8f0"]),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    return _card("abstract_coverage", "Abstract coverage",
                 "How many entries have an arXiv abstract attached (data quality).", fig, 400)


def topic_prevalence_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    """Share of each year's entries mentioning each tracked topic (title+abstract)."""
    year_total: Counter[int] = Counter()
    topic_year = {t: Counter() for t in _TREND_TOPICS}
    for e in entries:
        if not e.year:
            continue
        year_total[e.year] += 1
        text = f"{e.title} {e.abstract or ''}".lower()
        for topic, kws in _TREND_TOPICS.items():
            if any(k in text for k in kws):
                topic_year[topic][e.year] += 1
    years = sorted(year_total)
    if len(years) < 2:
        return None
    fig = go.Figure()
    for topic, yc in topic_year.items():
        if sum(yc.values()) == 0:
            continue
        ys = [100.0 * yc.get(y, 0) / year_total[y] for y in years]
        fig.add_trace(go.Scatter(
            x=years, y=ys, mode="lines+markers", name=topic,
            hovertemplate=f"{topic} — %{{x}}: %{{y:.0f}}%<extra></extra>",
        ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="% of that year's entries", ticksuffix="%")
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return _card("topic_prevalence", "Topic prevalence over time",
                 "Share of each year's entries mentioning a topic — watch the shift toward "
                 "LLM/agent work as the field's vocabulary changes.", fig, 520)


def rising_falling_keyphrases(report: ThemeReport, settings: Settings, top: int = 8) -> dict | None:
    """Linear trend (slope) of each consensus keyphrase's yearly mentions."""
    tby = report.themes_by_year
    if not tby or len(tby) < 2:
        return None
    years = sorted(int(y) for y in tby)
    phrases: set[str] = set()
    for ym in tby.values():
        phrases.update(ym)
    slopes: dict[str, float] = {}
    try:
        import numpy as np
        xs = np.array(years, dtype=float)
        for p in phrases:
            ys = np.array([tby[str(y)].get(p, 0) for y in years], dtype=float)
            if ys.sum() == 0:
                continue
            slopes[p] = float(np.polyfit(xs, ys, 1)[0])
    except Exception:  # pragma: no cover - numpy always present here
        span = max(1, years[-1] - years[0])
        for p in phrases:
            slopes[p] = (tby[str(years[-1])].get(p, 0) - tby[str(years[0])].get(p, 0)) / span
    if not slopes:
        return None
    ranked = sorted(slopes.items(), key=lambda kv: kv[1])
    fallers = [kv for kv in ranked if kv[1] < 0][: max(1, top // 2)]
    risers = [kv for kv in ranked if kv[1] > 0][-top:]
    items = fallers + risers
    if not items:
        return None
    labels = [k for k, _ in items]
    vals = [round(v, 3) for _, v in items]
    colors = ["#e53e3e" if v < 0 else "#2f855a" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:+.2f} mentions/yr<extra></extra>",
    ))
    fig.update_xaxes(title="Trend (linear slope, mentions per year)")
    height = max(360, 30 * len(labels) + 120)
    return _card("rising_falling", "Rising &amp; fading themes",
                 "Linear trend of each consensus keyphrase's yearly mentions — "
                 "green is rising, red is fading.", fig, height)


def venue_mix_over_time(entries: list[Entry], settings: Settings, top_n: int = 6) -> dict | None:
    """Stacked yearly counts for the most common venues (incl. the arXiv-preprint share)."""
    totals: Counter[str] = Counter()
    for e in entries:
        if e.year:
            totals[e.venue_display or resolve_venue(e)] += 1
    top = [v for v, _ in totals.most_common(top_n)]
    years = sorted({e.year for e in entries if e.year})
    if not top or len(years) < 2:
        return None
    yindex = {y: i for i, y in enumerate(years)}
    data = {v: [0] * len(years) for v in top}
    for e in entries:
        v = e.venue_display or resolve_venue(e)
        if e.year and v in data:
            data[v][yindex[e.year]] += 1
    fig = go.Figure()
    for v in top:
        fig.add_trace(go.Scatter(
            x=years, y=data[v], mode="lines", stackgroup="one", name=v,
            hovertemplate=f"{v} — %{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="Entries")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _card("venue_mix", "Venue mix over time",
                 "How the leading venues' yearly counts stack up (the arXiv-preprint share "
                 "shows how much work stays unpublished).", fig, 520)


def category_mix_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    """Stacked yearly counts per curated category."""
    years = sorted({e.year for e in entries if e.year})
    if len(years) < 2:
        return None
    yindex = {y: i for i, y in enumerate(years)}
    data: dict[str, list[int]] = {k: [0] * len(years) for k in CATEGORIES}
    for e in entries:
        if e.year:
            data.setdefault(e.category, [0] * len(years))[yindex[e.year]] += 1
    fig = go.Figure()
    for k in CATEGORIES:
        if sum(data[k]) == 0:
            continue
        label = CATEGORIES.get(k, k)
        fig.add_trace(go.Scatter(
            x=years, y=data[k], mode="lines", stackgroup="one", name=label,
            hovertemplate=f"{label} — %{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="Entries")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _card("category_mix", "Category mix over time",
                 "Entries by curated category each year.", fig, 480)


def _fig_html(fig: go.Figure, include_js: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config=_PLOT_CONFIG,
    )


def _assemble(builders: list) -> list[dict]:
    """Run chart builders; return responsive HTML cards for those with data.

    Plotly.js is included (from the CDN) only by the first card on the page, so a
    page that embeds this list is self-contained.
    """
    cards: list[dict] = []
    for build in builders:
        try:
            card = build()
        except Exception as ex:  # never let one chart break the run
            log.warning("Plot failed: %s", ex)
            continue
        if not card:
            continue
        fig = card.pop("fig")
        card["html"] = _fig_html(fig, include_js=not cards)  # first card carries plotly.js
        cards.append(card)
    return cards


def make_plots(entries: list[Entry], report: ThemeReport, settings: Settings) -> list[dict]:
    """Home overview charts (current state of the corpus)."""
    cards = _assemble([
        lambda: papers_per_year(entries, settings),
        lambda: cumulative_growth(entries, settings),
        lambda: category_distribution(entries, settings),
        lambda: venue_distribution(entries, settings),
        lambda: theme_keyword_frequency(report, settings),
        lambda: cluster_sizes(report, settings),
        lambda: abstract_coverage(entries, settings),
    ])
    log.info("Plots: produced %d interactive figures", len(cards))
    return cards


def make_trend_plots(entries: list[Entry], report: ThemeReport, settings: Settings) -> list[dict]:
    """Trends page charts (how the field changes over the years)."""
    cards = _assemble([
        lambda: topic_prevalence_over_time(entries, settings),
        lambda: themes_over_time(report, settings),
        lambda: rising_falling_keyphrases(report, settings),
        lambda: venue_mix_over_time(entries, settings),
        lambda: category_mix_over_time(entries, settings),
    ])
    log.info("Trends: produced %d interactive figures", len(cards))
    return cards
