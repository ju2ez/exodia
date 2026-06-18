"""Idea generation by invoking AI-Scientist-v2 as a subprocess.

We do NOT vendor AI-Scientist-v2. At run time its repo is cloned (in CI) or
pointed at locally via EXODIA_AISCIENTIST_DIR, and we shell out to::

    python ai_scientist/perform_ideation_temp_free.py \
        --workshop-file <topic.md> --model <m> \
        --max-num-generations <n> --num-reflections <r>

That script reads the whole markdown as freeform context and writes its ideas to
``<topic stem>.json`` (verified behavior). We synthesize the topic file from the
knowledge base + consensus themes (the "with open-endedness" arm), then parse the
resulting JSON (keys: Name, Title, Short Hypothesis, Related Work, Abstract,
Experiments, Risk Factors and Limitations) into Idea records with provenance.

``--dry-run`` skips the subprocess and loads a fixture so the rest of the
pipeline can be exercised offline with no API cost.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import paths
from .config import Settings
from .logging_setup import get_logger
from .models import Entry, Idea, ThemeReport
from .util import now_utc_iso, read_json, stable_id, write_json

log = get_logger(__name__)

IDEATION_SCRIPT = "ai_scientist/perform_ideation_temp_free.py"
FIXTURE_IDEAS = paths.REPO_ROOT / "tests" / "fixtures" / "aiscientist_ideas.json"

_IDEA_KEYS = {
    "name": "Name",
    "title": "Title",
    "short_hypothesis": "Short Hypothesis",
    "related_work": "Related Work",
    "abstract": "Abstract",
    "experiments": "Experiments",
    "risk_factors_and_limitations": "Risk Factors and Limitations",
}


def _select_entries(entries: list[Entry], limit: int = 25) -> list[Entry]:
    """Prefer entries that have abstracts and are recent."""
    return sorted(
        entries, key=lambda e: (1 if e.abstract else 0, e.year or 0), reverse=True
    )[:limit]


def synthesize_topic_md(
    entries: list[Entry], report: ThemeReport, settings: Settings, out_path: Path
) -> Path:
    """Write a workshop-style topic markdown distilled from the corpus."""
    keyphrases = ", ".join(k["phrase"] for k in report.top_keyphrases[:12]) or "open-endedness"
    lines: list[str] = []
    lines.append("# Open-Endedness: State of the Art and Open Problems")
    lines.append("")
    lines.append("## Keywords")
    lines.append(keyphrases)
    lines.append("")
    lines.append("## TL;DR")
    lines.append(
        "A workshop brief distilled from the awesome-open-ended reading list "
        "(curated by Jenny Zhang). Propose novel, impactful research ideas that "
        "advance open-ended learning, generation, and discovery."
    )
    lines.append("")
    lines.append("## Abstract")
    lines.append(
        "Open-endedness studies systems that endlessly generate novel and "
        "increasingly complex artifacts, behaviors, or problems. This brief "
        f"summarizes {report.n_docs} curated works. The dominant vocabulary across "
        f"the corpus is: {keyphrases}."
    )
    lines.append("")

    if report.clusters:
        lines.append("## Dominant themes (consensus across the corpus)")
        for c in report.clusters:
            reps = "; ".join(c.get("representative_titles", [])[:2])
            lines.append(f"- **{c['label']}** ({c['size']} works). e.g. {reps}")
        lines.append("")

    lines.append("## Representative work")
    for e in _select_entries(entries):
        meta = ", ".join(filter(None, [e.venue, str(e.year) if e.year else None]))
        head = f"- **{e.title}**" + (f" ({meta})" if meta else "")
        if e.abstract:
            abstract = e.abstract[:500] + ("…" if len(e.abstract) > 500 else "")
            head += f" — {abstract}"
        lines.append(head)
    lines.append("")
    lines.append("## Goal")
    lines.append(
        "Generate research proposals that extend the state of the art of "
        "open-endedness, building on the consensus themes above while seeking "
        "genuinely novel directions."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote ideation topic file: %s", out_path)
    return out_path


def run_ideation(topic_md: Path, settings: Settings, ais_dir: Path, env: dict | None = None) -> Path:
    """Invoke AI-Scientist-v2's ideation script; return the output JSON path."""
    topic_md = topic_md.resolve()
    script = ais_dir / IDEATION_SCRIPT
    if not script.exists():
        raise FileNotFoundError(f"AI-Scientist-v2 ideation script not found: {script}")
    cmd = [
        sys.executable,
        IDEATION_SCRIPT,
        "--workshop-file", str(topic_md),
        "--model", settings.ideation_model,
        "--max-num-generations", str(settings.ideation_max_num_generations),
        "--num-reflections", str(settings.ideation_num_reflections),
    ]
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    log.info("Invoking AI-Scientist-v2: %s (cwd=%s)", " ".join(cmd), ais_dir)
    subprocess.run(
        cmd, cwd=str(ais_dir), env=run_env, check=True, timeout=settings.ideation_timeout_seconds
    )
    out = topic_md.with_suffix(".json")
    if not out.exists():
        raise FileNotFoundError(f"Expected ideation output not found: {out}")
    return out


def parse_ideas(json_path: str | Path, run_id: str, model: str, source_sha: str | None) -> list[Idea]:
    """Parse AI-Scientist-v2 ideation JSON into Idea records with provenance."""
    data = read_json(json_path, default=[]) or []
    now = now_utc_iso()
    ideas: list[Idea] = []
    for d in data:
        fields = {attr: (d.get(key) or "").strip() for attr, key in _IDEA_KEYS.items()}
        name = fields["name"] or fields["title"]
        if not name:
            continue
        ideas.append(
            Idea(
                # Content-based id: the SAME idea regenerated in a later run keeps
                # one identity, so it dedupes in store_ideas and its votes persist.
                idea_id=stable_id(name, fields["title"]),
                run_id=run_id,
                run_utc=now,
                model=model,
                source_sha=source_sha,
                **fields,
            )
        )
    log.info("Parsed %d ideas from %s", len(ideas), json_path)
    return ideas


def load_ideas(path: str | Path) -> list[dict]:
    return read_json(path, default=[]) or []


def store_ideas(settings: Settings, ideas: list[Idea]) -> list[dict]:
    """Append new ideas (deduped by idea_id) to ideas.json."""
    existing = load_ideas(settings.ideas_path)
    seen = {d.get("idea_id") for d in existing}
    for idea in ideas:
        if idea.idea_id in seen:
            continue
        existing.append(idea.to_dict())
        seen.add(idea.idea_id)
    write_json(settings.ideas_path, existing)
    return existing


def ideate(
    entries: list[Entry],
    report: ThemeReport,
    settings: Settings,
    run_id: str,
    source_sha: str | None,
    *,
    dry_run: bool = False,
    ais_dir: str | Path | None = None,
    workdir: str | Path | None = None,
) -> list[Idea]:
    """Synthesize the topic file and produce ideas (real or dry-run fixture)."""
    workdir = Path(workdir) if workdir else (paths.REPO_ROOT / ".exodia-work")
    topic_md = (workdir / "exodia_topic.md").resolve()
    synthesize_topic_md(entries, report, settings, topic_md)

    if dry_run:
        log.info("Dry-run: using idea fixture instead of invoking AI-Scientist-v2")
        return parse_ideas(FIXTURE_IDEAS, run_id, f"{settings.ideation_model} (dry-run)", source_sha)

    ais = Path(ais_dir or os.environ.get("EXODIA_AISCIENTIST_DIR", "")).expanduser()
    if not str(ais) or not ais.exists():
        log.warning("AI-Scientist-v2 dir not available (%s); skipping idea generation", ais)
        return []
    out = run_ideation(topic_md, settings, ais)
    return parse_ideas(out, run_id, settings.ideation_model, source_sha)
