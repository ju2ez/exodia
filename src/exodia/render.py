"""Render the static site from the committed knowledge base and run artifacts.

``render_site`` is self-contained: it loads the knowledge base, themes, ideas,
changelog, and state from disk, (re)generates the plots, and writes the HTML
pages + assets into the site directory. Every page carries the upstream credit
(via the footer) and a prominent "last updated" bar; the ideas page carries the
mandatory AI-generated-content disclaimer.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import paths
from .config import Settings
from .logging_setup import get_logger
from .models import CATEGORIES, State, ThemeReport
from .plotting import make_plots, make_trend_plots
from .store import load_entries
from .util import read_json
from .venues import resolve_venues

log = get_logger(__name__)

PAGES = ["index.html", "concepts.html", "trends.html", "papers.html", "ideas.html", "changelog.html", "about.html"]


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(paths.TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _copy_assets(settings: Settings) -> None:
    # Copy the whole asset tree (css + js); plots are now inlined, not files.
    for sub in ("css", "js"):
        src = paths.ASSETS_DIR / sub
        if src.exists():
            shutil.copytree(src, settings.site_dir / "assets" / sub, dirs_exist_ok=True)


def site_context(settings: Settings) -> tuple[Environment, dict, dict]:
    """Load all data and build the (env, global_ctx, per-page_ctx) used to render.

    Shared by the static renderer and the dynamic FastAPI app so both produce
    identical pages; the web app augments the ideas context with live data.
    """
    entries = load_entries(settings.kb_path)
    resolve_venues(entries)  # ensure venue_display is present even on an older KB
    themes_raw = read_json(settings.themes_path, default=None)
    report = (
        ThemeReport.from_dict(themes_raw)
        if themes_raw
        else ThemeReport(generated_utc="", method="none", n_docs=len(entries))
    )
    ideas = read_json(settings.ideas_path, default=[]) or []
    changelog = read_json(settings.changelog_path, default=[]) or []
    state = State.from_dict(
        read_json(settings.state_path, default={"upstream_repo": settings.upstream_repo})
    )

    plot_cards = make_plots(entries, report, settings)
    trend_cards = make_trend_plots(entries, report, settings)

    grouped: dict[str, list] = {k: [] for k in CATEGORIES}
    for e in entries:
        grouped.setdefault(e.category, []).append(e)

    ideas_sorted = sorted(ideas, key=lambda d: d.get("run_utc", ""), reverse=True)
    # Best ideas ranked by GitHub-Issues votes (only those that have an issue yet).
    voted = [d for d in ideas_sorted if d.get("issue_url")]
    best_ideas = sorted(
        voted,
        key=lambda d: (d.get("score", 0), d.get("votes_up", 0), bool(d.get("realized")), d.get("run_utc", "")),
        reverse=True,
    )[:10]
    # Only show runs that actually changed something (skip no-op polls/forced reruns).
    def _nonempty(c: dict) -> bool:
        counts = c.get("counts") or {}
        return bool(
            counts.get("added") or counts.get("removed") or counts.get("changed")
            or c.get("new_ideas")
        )

    changelog_sorted = [c for c in reversed(changelog) if _nonempty(c)]
    last_updated = state.last_run_utc or _now_iso()

    global_ctx = {
        "site_title": settings.site_title,
        "last_updated_utc": last_updated,
        "last_updated_date": last_updated[:10],
        "upstream_repo": settings.upstream_repo,
        "upstream_url": settings.upstream_url,
        "upstream_sha_short": (state.upstream_sha or "")[:7] or "—",
        "run_count": state.run_count,
        "current_year": datetime.now(UTC).year,
        "dynamic": False,  # the web app flips this on to show vote/feedback widgets
    }

    page_ctx = {
        "index.html": {
            "plot_cards": plot_cards,
            "n_entries": len(entries),
            "n_ideas": len(ideas),
            "n_clusters": len(report.clusters),
        },
        "concepts.html": {"report": report},
        "trends.html": {"plot_cards": trend_cards, "n_entries": len(entries)},
        "papers.html": {
            "grouped": grouped,
            "categories": list(CATEGORIES.items()),
            "n_entries": len(entries),
        },
        "ideas.html": {"ideas": ideas_sorted, "n_ideas": len(ideas), "best_ideas": best_ideas},
        "changelog.html": {"changelog": changelog_sorted},
        "about.html": {"report": report, "n_entries": len(entries)},
    }
    return _env(), global_ctx, page_ctx


def render_site(settings: Settings) -> Path:
    env, global_ctx, page_ctx = site_context(settings)
    settings.site_dir.mkdir(parents=True, exist_ok=True)
    for name in PAGES:
        html = env.get_template(name).render(**global_ctx, **page_ctx[name])
        (settings.site_dir / name).write_text(html, encoding="utf-8")

    _copy_assets(settings)
    # Disable Jekyll so asset folders (e.g. _-prefixed) are served verbatim.
    (settings.site_dir / ".nojekyll").write_text("", encoding="utf-8")

    log.info("Rendered %d pages -> %s", len(PAGES), settings.site_dir)
    return settings.site_dir


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
