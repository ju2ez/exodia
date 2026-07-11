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

import statistics
from collections import Counter
from dataclasses import replace

import plotly.graph_objects as go

from .concepts import CONCEPTS, compile_gazetteer, concept_matchers
from .config import Settings
from .corpus import corpus_text
from .enrich import base_id
from .futurework import future_work_text
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

# The Trends page covers the modern era of the field: every trend starts here.
_TREND_MIN_YEAR = 2017

# Concept trends are driven by the curated gazetteer in concepts.py (CONCEPTS),
# so what we plot is always a sensible research concept, not TF-IDF noise.

# Curated gazetteers for entity trends (matched as whole words against title+abstract).
_MODELS = {
    "GPT-4 / 4o": ["gpt-4", "gpt-4o"],
    "GPT-3 / 3.5 / ChatGPT": ["gpt-3", "chatgpt", "instructgpt"],
    "GPT-2": ["gpt-2"],
    "Claude": ["claude"],
    "Gemini": ["gemini"],
    "PaLM": ["palm"],
    "Llama": ["llama"],
    "Mistral / Mixtral": ["mistral", "mixtral"],
    "Qwen": ["qwen"],
    "DeepSeek": ["deepseek"],
    "BERT": ["bert"],
    "T5": ["t5"],
    "Diffusion models": ["diffusion"],
    "Dreamer / world model": ["dreamer", "world model*"],
    "MuZero / AlphaZero": ["muzero", "alphazero", "alphago"],
    "PPO / DQN (RL)": ["ppo", "dqn", "impala", "a3c"],
}
_PROVIDERS = {
    "OpenAI": ["openai"],
    "Google / DeepMind": ["deepmind", "google"],
    "Meta AI": ["meta ai", "facebook ai"],
    "Anthropic": ["anthropic"],
    "Microsoft": ["microsoft"],
    "Mistral AI": ["mistral ai"],
    "Hugging Face": ["hugging face", "huggingface"],
    "Stability AI": ["stability ai"],
    "Sakana AI": ["sakana"],
    "Cohere": ["cohere"],
    "Alibaba (Qwen)": ["alibaba"],
    "NVIDIA": ["nvidia"],
}
_DATASETS = {
    "Minecraft / MineDojo": ["minecraft", "minedojo", "minerl", "malmo"],
    "Atari (ALE)": ["atari", "arcade learning environment"],
    "MuJoCo": ["mujoco"],
    "NetHack": ["nethack", "minihack"],
    "Crafter": ["crafter"],
    "Procgen": ["procgen"],
    "XLand": ["xland"],
    "MiniGrid / BabyAI": ["minigrid", "babyai"],
    "Meta-World": ["meta-world", "metaworld"],
    "DM Control / DMLab": ["dm control", "dmcontrol", "deepmind lab", "dmlab", "control suite"],
    "Gym / Gymnasium": ["openai gym", "gymnasium"],
    "MMLU": ["mmlu"],
    "HumanEval": ["humaneval"],
    "GSM8K": ["gsm8k"],
    "BIG-bench": ["big-bench", "bigbench"],
    "ImageNet": ["imagenet"],
    "StarCraft (SMAC)": ["starcraft", "smac"],
}
_TASKS = {
    "Code generation": ["code generation", "program synthesis", "coding"],
    "Math reasoning": ["math reasoning", "mathematical reasoning", "theorem proving"],
    "Question answering": ["question answering", "question-answering"],
    "Dialogue / chat": ["dialogue", "conversational", "chatbot*"],
    "Summarization": ["summarization"],
    "Planning": ["planning"],
    "Tool use": ["tool use", "tool-use", "function calling"],
    "Instruction following": ["instruction following", "instruction-following", "instruction tuning"],
    "Navigation": ["navigation", "maze*"],
    "Manipulation / locomotion": ["manipulation", "grasping", "locomotion"],
    "Game playing": ["game playing", "game-playing"],
    "Image generation": ["image generation", "text-to-image", "text to image"],
    "Exploration": ["exploration"],
}
_ARXIV_CAT_NAMES = {
    "cs.AI": "cs.AI · AI", "cs.LG": "cs.LG · ML", "cs.NE": "cs.NE · neural/evo",
    "cs.RO": "cs.RO · robotics", "cs.CL": "cs.CL · NLP", "cs.CV": "cs.CV · vision",
    "cs.MA": "cs.MA · multi-agent", "stat.ML": "stat.ML", "cs.GT": "cs.GT · game theory",
    "cs.HC": "cs.HC · HCI", "cs.SY": "cs.SY · systems",
}


def _layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(
        # Left-aligned title with its own auto-margin so long titles never clip.
        title=dict(text=title, font=dict(size=16), x=0.01, xanchor="left", automargin=True),
        height=height,
        # Generous margins; automargin on the axes grows l/b further to fit labels.
        margin=dict(l=16, r=28, t=64, b=64),
        font=_FONT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(font_size=13),
        autosize=True,
    )
    # automargin = expand the margin to fit tick labels (long venue / concept /
    # paper-title labels on horizontal bars were being cut off on the left).
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f2", zeroline=False, automargin=True)
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
    # Only paper-like entries have a venue; blogs/videos would inflate "unpublished".
    for e in entries:
        if e.category in ("blogs", "videos"):
            continue
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
    return _card("themes_over_time", "Top themes over time",
                 "How the leading themes wax and wane year over year.", fig, 480)


def abstract_coverage(entries: list[Entry], settings: Settings) -> dict | None:
    """Abstract coverage among paper-like entries only.

    Blog posts and videos have no abstract by nature, so counting them would
    understate coverage — they're excluded from the denominator.
    """
    scholarly = [e for e in entries if e.category not in ("blogs", "videos")]
    if not scholarly:
        return None
    with_abs = sum(1 for e in scholarly if e.abstract)
    without = len(scholarly) - with_abs
    fig = go.Figure(go.Pie(
        labels=["With abstract", "No abstract"], values=[with_abs, without], hole=0.55,
        marker=dict(colors=[_ACCENT, "#e2e8f0"]),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    return _card("abstract_coverage", "Abstract coverage (papers)",
                 "Share of paper-like entries with an arXiv abstract attached — blog posts and "
                 "videos are excluded since they have none (an enrichment/data-quality signal).",
                 fig, 400)


def concepts_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    """Share of each year's entries engaging the top curated concepts over time."""
    return _entity_trend(
        entries, CONCEPTS, settings, card_id="concepts_trend",
        title="Concepts over time", top_n=8, height=520,
        caption="Share of each year's entries engaging each curated open-endedness concept "
                "(matched across full text) — watch the shift toward LLM / agent work.",
    )


def top_concepts(report: ThemeReport, settings: Settings, top_n: int = 18) -> dict | None:
    """Curated concepts ranked by how many curated works engage them."""
    cs = report.concepts[:top_n][::-1]
    if not cs:
        return None
    labels = [c["concept"] for c in cs]
    vals = [c["docs"] for c in cs]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker=dict(color=vals, colorscale=_SEQ),
        hovertemplate="%{y}: %{x} works<extra></extra>",
    ))
    fig.update_xaxes(title="Works engaging the concept")
    height = max(380, 26 * len(cs) + 120)
    return _card("top_concepts", "Key concepts in the field",
                 "Curated open-endedness concepts ranked by how many curated works engage them "
                 "(detected across full text + abstracts).", fig, height)


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
    return _card("category_mix", "Category mix over time",
                 "Entries by curated category each year.", fig, 480)


def _entity_trend(
    entries: list[Entry], gazetteer: dict[str, list[str]], settings: Settings, *, card_id: str,
    title: str, caption: str, top_n: int = 8, height: int = 520, share: bool = True,
) -> dict | None:
    """Per-year prevalence of named entities (keyword gazetteer) in an entry's text.

    Matches each keyword as a whole word/phrase (both boundaries; a trailing ``*``
    marks a deliberate prefix-stem — see :func:`exodia.concepts.alias_pattern`)
    against the full corpus text (title + abstract + any cached transcript / PDF
    full text). Plots the top-N entities by total mentions as lines: % of that
    year's entries (``share``) or absolute mention counts.
    """
    matchers = compile_gazetteer(gazetteer)
    year_total: Counter[int] = Counter()
    ent_year: dict[str, Counter] = {label: Counter() for label in gazetteer}
    totals: Counter[str] = Counter()
    for e in entries:
        if not e.year:
            continue
        year_total[e.year] += 1
        text = corpus_text(e, settings)
        for label, rx in matchers.items():
            if rx.search(text):
                ent_year[label][e.year] += 1
                totals[label] += 1
    years = sorted(year_total)
    top = [label for label, n in totals.most_common(top_n) if n > 0]
    if len(years) < 2 or not top:
        return None
    fig = go.Figure()
    for label in top:
        yc = ent_year[label]
        if share:
            ys = [100.0 * yc.get(y, 0) / year_total[y] for y in years]
            tmpl = f"{label} — %{{x}}: %{{y:.0f}}%<extra></extra>"
        else:
            ys = [yc.get(y, 0) for y in years]
            tmpl = f"{label} — %{{x}}: %{{y}}<extra></extra>"
        fig.add_trace(go.Scatter(x=years, y=ys, mode="lines+markers", name=label, hovertemplate=tmpl))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="% of that year's entries" if share else "Entries mentioning",
                     ticksuffix="%" if share else "")
    return _card(card_id, title, caption, fig, height)


def models_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    return _entity_trend(
        entries, _MODELS, settings, card_id="models_trend", title="Models used over time",
        caption="Share of each year's entries mentioning a model / algorithm family "
                "(keyword-based, approximate) — the LLM families climb fast.",
    )


def providers_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    return _entity_trend(
        entries, _PROVIDERS, settings, card_id="providers_trend",
        title="Model providers over time",
        caption="Share of each year's entries mentioning an AI lab / model provider "
                "(keyword-based, approximate).",
    )


def datasets_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    return _entity_trend(
        entries, _DATASETS, settings, card_id="datasets_trend",
        title="Datasets, benchmarks & environments over time",
        caption="Share of each year's entries mentioning a dataset / benchmark / environment "
                "(keyword-based, approximate).",
    )


def tasks_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    return _entity_trend(
        entries, _TASKS, settings, card_id="tasks_trend", title="Tasks studied over time",
        caption="Share of each year's entries mentioning a task type (keyword-based, approximate).",
    )


def arxiv_categories_over_time(entries: list[Entry], settings: Settings, top_n: int = 8) -> dict | None:
    """Primary arXiv subject categories of the papers, by year (from arxiv metadata)."""
    year_total: Counter[int] = Counter()
    cat_year: dict[str, Counter] = {}
    totals: Counter[str] = Counter()
    for e in entries:
        if not e.year or not e.arxiv_categories:
            continue
        year_total[e.year] += 1
        for c in dict.fromkeys(e.arxiv_categories):  # de-dup within an entry
            cat_year.setdefault(c, Counter())[e.year] += 1
            totals[c] += 1
    years = sorted(year_total)
    top = [c for c, _ in totals.most_common(top_n)]
    if len(years) < 2 or not top:
        return None
    fig = go.Figure()
    for c in top:
        yc = cat_year[c]
        name = _ARXIV_CAT_NAMES.get(c, c)
        fig.add_trace(go.Scatter(
            x=years, y=[yc.get(y, 0) for y in years], mode="lines+markers", name=name,
            hovertemplate=f"{name} — %{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="Papers")
    return _card("arxiv_categories", "arXiv subfields over time",
                 "Primary arXiv categories of the papers each year — which subfields drive "
                 "open-endedness (from arXiv metadata, not keywords).", fig)


def code_availability_over_time(entries: list[Entry], settings: Settings) -> dict | None:
    """Share of each year's *papers* that link to code — a reproducibility signal.

    Only paper-like entries can ship code, so blog posts and videos are excluded
    from the denominator (a video that obviously can't release a repo would
    otherwise understate the share), matching :func:`abstract_coverage`.
    """
    year_total: Counter[int] = Counter()
    year_code: Counter[int] = Counter()
    for e in entries:
        if not e.year or e.category in ("blogs", "videos"):
            continue
        year_total[e.year] += 1
        if "code" in (e.links or {}):
            year_code[e.year] += 1
    years = sorted(year_total)
    if len(years) < 2:
        return None
    ys = [100.0 * year_code.get(y, 0) / year_total[y] for y in years]
    fig = go.Figure(go.Scatter(
        x=years, y=ys, mode="lines+markers", line=dict(color=_ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(47,133,90,0.12)",
        hovertemplate="%{x}: %{y:.0f}% link to code<extra></extra>",
    ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="% of papers with a code link", ticksuffix="%")
    return _card("code_availability", "Code availability over time",
                 "Share of each year's papers that link to code — a reproducibility signal. "
                 "Blog posts and videos are excluded since they can't ship a repository.", fig, 440)


def _paper_link(e: Entry) -> str:
    if e.links and e.links.get("paper"):
        return e.links["paper"]
    return e.abstract_url or (next(iter(e.links.values()), "") if e.links else "")


def most_cited_list(entries: list[Entry], top_n: int = 15) -> list[dict]:
    """Top-N entries by citation count, as data for an HTML list.

    Rendered as HTML (not a Plotly chart) so the full paper titles wrap and link
    out — a horizontal-bar chart clips long titles into its y-axis margin. Deduped
    by arXiv id so two entries sharing one preprint (e.g. a paper's v1/v2) don't
    both occupy a slot with the same citation count.
    """
    ordered = sorted((e for e in entries if e.citation_count),
                     key=lambda e: e.citation_count or 0, reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for e in ordered:
        key = base_id(e.arxiv_id) if e.arxiv_id else e.entry_id
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": e.title, "url": _paper_link(e), "citations": e.citation_count,
                    "venue": e.venue_display, "year": e.year})
        if len(out) >= top_n:
            break
    return out


def citation_weighted_concepts_over_time(entries: list[Entry], settings: Settings,
                                         top_n: int = 7) -> dict | None:
    """Share of each publication-year's *citations* captured by each concept (impact-weighted)."""
    matchers = concept_matchers()
    year_cit: Counter[int] = Counter()
    concept_year_cit: dict[str, Counter] = {label: Counter() for label in CONCEPTS}
    totals: Counter[str] = Counter()
    for e in entries:
        if not e.year or not e.citation_count:
            continue
        year_cit[e.year] += e.citation_count
        text = corpus_text(e, settings)
        for label, rx in matchers.items():
            if rx.search(text):
                concept_year_cit[label][e.year] += e.citation_count
                totals[label] += e.citation_count
    years = sorted(y for y in year_cit if year_cit[y] > 0)
    top = [label for label, c in totals.most_common(top_n) if c > 0]
    if len(years) < 2 or not top:
        return None
    fig = go.Figure()
    for label in top:
        yc = concept_year_cit[label]
        ys = [100.0 * yc.get(y, 0) / year_cit[y] for y in years]
        fig.add_trace(go.Scatter(
            x=years, y=ys, mode="lines+markers", name=label,
            hovertemplate=f"{label} — %{{x}}: %{{y:.0f}}% of citations<extra></extra>",
        ))
    fig.update_xaxes(title="Year of publication", dtick=1)
    fig.update_yaxes(title="% of that year's citations", ticksuffix="%")
    return _card("citation_weighted_concepts", "Research impact by concept (citation-weighted)",
                 "Share of each publication-year's total citations captured by each concept — "
                 "weights every entry by how often it's cited, so it tracks where impact "
                 "concentrates, not merely how many papers appear.", fig, 540)


def citations_by_year(entries: list[Entry], settings: Settings) -> dict | None:
    """Median (bars) and mean (line) citations per paper by publication year."""
    by_year: dict[int, list[int]] = {}
    for e in entries:
        if e.year and e.citation_count is not None:
            by_year.setdefault(e.year, []).append(e.citation_count)
    years = sorted(by_year)
    if len(years) < 2 or not any(by_year[y] for y in years):
        return None
    medians = [statistics.median(by_year[y]) for y in years]
    means = [round(statistics.fmean(by_year[y]), 1) for y in years]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=medians, name="median", marker_color=_ACCENT,
                         hovertemplate="%{x}: median %{y} citations<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=means, name="mean", mode="lines+markers",
                             line=dict(color=_ACCENT2, width=3),
                             hovertemplate="%{x}: mean %{y} citations<extra></extra>"))
    fig.update_xaxes(title="Year of publication", dtick=1)
    fig.update_yaxes(title="Citations per paper")
    return _card("citations_by_year", "Citations per paper by year",
                 "Median (bars) and mean (line) citations per paper by publication year. "
                 "Recent years are necessarily under-counted — citations accrue with age.", fig, 460)


def _future_concepts_by_entry(entries: list[Entry], settings: Settings) -> list[tuple[Entry, list[str]]]:
    """For each entry with a forward-looking section, the curated concepts it proposes.

    Runs the concept gazetteer over *only* the "future work / conclusion /
    limitations" text (:mod:`futurework`), so a hit means the paper explicitly
    flags that concept as a next step — not merely that it studied it.
    """
    matchers = concept_matchers()
    out: list[tuple[Entry, list[str]]] = []
    for e in entries:
        txt = future_work_text(e, settings)
        if not txt:
            continue
        hits = [label for label, rx in matchers.items() if rx.search(txt)]
        if hits:
            out.append((e, hits))
    return out


def future_directions_over_time(entries: list[Entry], settings: Settings,
                                top_n: int = 8, height: int = 540) -> dict | None:
    """Share of each year's papers whose forward-looking sections propose each concept.

    Mines every paper's "future work / conclusion / limitations" sections, detects
    which curated concepts authors flag there, and trends them by year — concepts
    the field keeps (and increasingly) names as open are the emerging hot topics.
    """
    fce = _future_concepts_by_entry(entries, settings)
    if not fce:
        return None
    year_total: Counter[int] = Counter()
    concept_year: dict[str, Counter] = {label: Counter() for label in CONCEPTS}
    totals: Counter[str] = Counter()
    for e, hits in fce:
        if not e.year:
            continue
        year_total[e.year] += 1
        for label in hits:
            concept_year[label][e.year] += 1
            totals[label] += 1
    years = sorted(year_total)
    top = [label for label, n in totals.most_common(top_n) if n > 0]
    if len(years) < 2 or not top:
        return None
    fig = go.Figure()
    for label in top:
        yc = concept_year[label]
        ys = [100.0 * yc.get(y, 0) / year_total[y] if year_total[y] else 0 for y in years]
        fig.add_trace(go.Scatter(
            x=years, y=ys, mode="lines+markers", name=label,
            hovertemplate=f"{label} — %{{x}}: %{{y:.0f}}% of that year's papers<extra></extra>",
        ))
    fig.update_xaxes(title="Year", dtick=1)
    fig.update_yaxes(title="% of papers proposing it as future work", ticksuffix="%")
    return _card(
        "future_directions_trend", "Future directions over time — the next hot things",
        "Share of each year's papers whose “future work / conclusion / limitations” "
        "sections explicitly propose each curated concept (detected only in those forward-looking "
        "sections). Concepts that papers keep naming as open — increasingly so — are the "
        "emerging hot topics.", fig, height,
    )


def future_directions_ranked(entries: list[Entry], settings: Settings,
                             top_n: int = 12, recent_years: int = 3) -> list[dict]:
    """Concepts most often proposed as future work, with a recency (momentum) signal.

    Returns data for an HTML "what the field says comes next" summary: per concept,
    how many papers flag it as a future direction, its share of those papers, and
    how concentrated those mentions are in the most recent ``recent_years`` years.
    """
    fce = _future_concepts_by_entry(entries, settings)
    if not fce:
        return []
    years = [e.year for e, _ in fce if e.year]
    if not years:
        return []
    cutoff = max(years) - recent_years + 1
    docs: Counter[str] = Counter()
    recent: Counter[str] = Counter()
    total = sum(1 for e, _ in fce) or 1
    for e, hits in fce:
        is_recent = bool(e.year and e.year >= cutoff)
        for label in hits:
            docs[label] += 1
            if is_recent:
                recent[label] += 1
    out: list[dict] = []
    for label, n in docs.most_common(top_n):
        out.append({
            "concept": label,
            "docs": n,
            "share": round(n / total, 4),
            "recent": recent[label],
            "recent_share": round(recent[label] / n, 4) if n else 0.0,
        })
    return out


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
        lambda: top_concepts(report, settings),
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


def _trend_entries(entries: list[Entry], min_year: int = _TREND_MIN_YEAR) -> list[Entry]:
    """Entries dated on/after ``min_year`` (undated entries are kept; builders skip them)."""
    return [e for e in entries if not e.year or e.year >= min_year]


def _report_since(report: ThemeReport, min_year: int) -> ThemeReport:
    """Copy of ``report`` with its year-keyed theme tallies clipped to >= min_year."""
    tby = {y: m for y, m in report.themes_by_year.items() if y.isdigit() and int(y) >= min_year}
    return replace(report, themes_by_year=tby)


def make_trend_plots(entries: list[Entry], report: ThemeReport, settings: Settings,
                     moi: dict | None = None) -> list[dict]:
    """Trends page charts (how the field changes over the years).

    The whole page covers the modern era: entries (and the year-keyed theme data)
    are clipped to >= ``_TREND_MIN_YEAR`` so every chart starts at the same year.
    ``moi`` is the committed MOI backtest dict (or None); its forecasting chart is
    appended here so the single-plotly.js-load invariant is preserved.
    """
    from .moi.plots import moi_hitrate_by_cutoff  # lazy: moi.plots imports from this module

    entries = _trend_entries(entries)
    report = _report_since(report, _TREND_MIN_YEAR)
    cards = _assemble([
        lambda: concepts_over_time(entries, settings),
        lambda: future_directions_over_time(entries, settings),
        lambda: citation_weighted_concepts_over_time(entries, settings),
        lambda: models_over_time(entries, settings),
        lambda: providers_over_time(entries, settings),
        lambda: datasets_over_time(entries, settings),
        lambda: tasks_over_time(entries, settings),
        lambda: arxiv_categories_over_time(entries, settings),
        lambda: code_availability_over_time(entries, settings),
        lambda: citations_by_year(entries, settings),
        lambda: themes_over_time(report, settings),
        lambda: rising_falling_keyphrases(report, settings),
        lambda: venue_mix_over_time(entries, settings),
        lambda: category_mix_over_time(entries, settings),
        lambda: moi_hitrate_by_cutoff(moi, settings),
    ])
    log.info("Trends: produced %d interactive figures", len(cards))
    return cards
