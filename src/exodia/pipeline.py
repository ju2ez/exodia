"""Orchestrate the full Exodia loop.

``gate`` is the cheap SHA check the CI uses to decide whether to run. ``run_all``
executes the whole loop: fetch -> parse -> merge -> enrich -> analyze -> ideate
-> changelog -> persist state -> render. It early-exits when upstream is unchanged
(unless forced), so empty runs cost nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from . import upstream as up
from .analysis import analyze, save_themes
from .config import Settings
from .diffing import append_changelog, build_changelog_entry
from .enrich import enrich_entries
from .ideation import ideate, load_ideas, store_ideas
from .logging_setup import get_logger
from .matching import filter_novel, match_ideas_to_papers
from .models import State
from .parser import parse_readme
from .pdfs import download_pdfs
from .render import render_site
from .store import load_entries, merge, save_kb
from .util import now_utc_iso, write_json
from .venues import resolve_venues

log = get_logger(__name__)


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def gate(settings: Settings, force: bool = False) -> tuple[bool, str | None, State]:
    """Return (changed, latest_sha, state). First run or SHA mismatch => changed."""
    state = up.load_state(settings.state_path, settings.upstream_repo)
    try:
        latest, _ = up.resolve_change_token(settings.upstream_repo, settings.upstream_branch)
    except Exception as ex:
        log.warning("Could not resolve upstream change token: %s", ex)
        latest = None
    if force:
        changed = True
    elif latest is None:
        changed = False  # cannot confirm a change; stay put
    else:
        changed = up.is_changed(state, latest)
    log.info("Gate: changed=%s latest=%s stored=%s", changed, latest, state.upstream_sha)
    return changed, latest, state


def _write_run_meta(settings: Settings, run_id: str, diff: dict, n_ideas: int, sha: str | None) -> None:
    write_json(
        settings.runs_dir / f"{run_id}.json",
        {
            "run_id": run_id,
            "run_utc": now_utc_iso(),
            "upstream_sha": sha,
            "added": len(diff["added"]),
            "removed": len(diff["removed"]),
            "changed": len(diff["changed"]),
            "new_ideas": n_ideas,
        },
    )


def run_all(
    settings: Settings,
    *,
    force: bool = False,
    dry_run: bool = False,
    readme_path: str | None = None,
    ais_dir: str | None = None,
    no_enrich: bool = False,
) -> dict:
    """Execute the full pipeline. Returns a small result summary."""
    state = up.load_state(settings.state_path, settings.upstream_repo)
    from_sha = state.upstream_sha
    run_id = _run_id()
    local = readme_path is not None

    if local:
        md = Path(readme_path).read_text(encoding="utf-8")
        to_sha = from_sha
        log.info("Local run from %s (upstream SHA not advanced)", readme_path)
    else:
        md, to_sha = up.fetch_upstream(settings.upstream_repo, settings.upstream_branch)
        if not (force or up.is_changed(state, to_sha)):
            log.info("No upstream change (%s); skipping.", to_sha[:12])
            return {"changed": False, "sha": to_sha}

    parsed, _ = parse_readme(md, settings.upstream_repo)
    existing = load_entries(settings.kb_path)
    is_first = not existing
    merged, diff = merge(existing, parsed)

    if not no_enrich:
        try:
            enrich_entries(merged, settings)
        except Exception as ex:
            log.warning("Enrichment skipped due to error: %s", ex)

    if settings.pdf_fetch:
        try:
            download_pdfs(merged, settings)
        except Exception as ex:
            log.warning("PDF download skipped due to error: %s", ex)

    # Resolve the real publication venue (arXiv is a preprint server, not a venue).
    resolve_venues(merged)

    save_kb(settings, merged)

    report = analyze(merged, settings)
    save_themes(settings, report)

    try:
        ideas = ideate(merged, report, settings, run_id, to_sha, dry_run=dry_run, ais_dir=ais_dir)
        ideas = filter_novel(ideas, load_ideas(settings.ideas_path), settings.novelty_threshold)
    except Exception as ex:
        log.warning("Idea generation failed (%s); continuing without new ideas", ex)
        ideas = []
    store_ideas(settings, ideas)

    # Star-match: did a newly-added upstream paper essentially realize a prior idea?
    try:
        all_ideas = load_ideas(settings.ideas_path)
        n_marked = match_ideas_to_papers(all_ideas, diff["added"], settings.match_threshold)
        if n_marked:
            write_json(settings.ideas_path, all_ideas)
            log.info("Star-matched %d idea(s) to newly added papers", n_marked)
    except Exception as ex:
        log.warning("Idea<->paper matching skipped due to error: %s", ex)

    note = "Initial build." if is_first else ""
    compare = up.compare_url(settings.upstream_repo, from_sha, None if local else to_sha)
    changelog_entry = build_changelog_entry(
        run_id,
        diff,
        from_sha=from_sha,
        to_sha=None if local else to_sha,
        compare_url=compare,
        total_entries=len(merged),
        new_ideas=len(ideas),
        themes_changed=True,
        note=note,
    )
    append_changelog(settings.changelog_path, changelog_entry)

    if not local:
        state.upstream_sha = to_sha
    state.last_run_utc = now_utc_iso()
    state.run_count += 1
    up.save_state(state, settings.state_path)

    _write_run_meta(settings, run_id, diff, len(ideas), to_sha)
    render_site(settings)

    log.info(
        "Run %s complete: +%d / -%d / ~%d, %d ideas",
        run_id, len(diff["added"]), len(diff["removed"]), len(diff["changed"]), len(ideas),
    )
    return {
        "changed": True,
        "sha": to_sha,
        "added": len(diff["added"]),
        "removed": len(diff["removed"]),
        "changed_count": len(diff["changed"]),
        "ideas": len(ideas),
    }
