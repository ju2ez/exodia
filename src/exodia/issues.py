"""GitHub Issues as the community voting / feedback backend.

Each generated idea maps to one GitHub Issue on the project repo:

* 👍 / 👎 **reactions** on the issue are up / down votes,
* issue **comments** are feedback.

The pipeline syncs ideas -> issues (creating any that are missing) and reads the
reaction + comment tallies back into ``ideas.json``. The static site then shows
the counts, ranks a "best ideas" list, and links out to each issue for voting and
discussion — so the whole interaction lives on GitHub and the site can be hosted on
GitHub Pages with no separate backend.

Auth: a token with ``issues: write`` (the Actions ``GITHUB_TOKEN`` or a local PAT
in ``GITHUB_TOKEN``). Refreshing tallies needs only read access.
"""

from __future__ import annotations

import requests

from .config import Settings
from .ideation import load_ideas
from .logging_setup import get_logger
from .util import write_json

log = get_logger(__name__)

API = "https://api.github.com"
MARKER = "exodia-idea"  # HTML-comment marker tying an issue to an idea id
ISSUE_LABEL = "exodia-idea"
_TIMEOUT = 30


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "exodia",
    }


def _issue_body(idea: dict) -> str:
    return (
        f"<!-- {MARKER}: {idea['idea_id']} -->\n\n"
        "> **AI-generated research idea** (via AI-Scientist-v2). "
        "Vote with 👍 / 👎 reactions and leave feedback in the comments.\n\n"
        f"**Hypothesis.** {idea.get('short_hypothesis', '')}\n\n"
        f"**Abstract.** {idea.get('abstract', '')}\n\n"
        "_Tracked by Exodia — please keep the marker comment above intact._"
    )


def create_issue(repo: str, idea: dict, token: str) -> tuple[int, str]:
    """Open a discussion issue for an idea; returns (issue_number, html_url)."""
    resp = requests.post(
        f"{API}/repos/{repo}/issues",
        headers=_headers(token),
        json={"title": idea["title"], "body": _issue_body(idea), "labels": [ISSUE_LABEL]},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    d = resp.json()
    return d["number"], d["html_url"]


def fetch_issue(repo: str, number: int, token: str) -> dict:
    resp = requests.get(f"{API}/repos/{repo}/issues/{number}", headers=_headers(token), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def tally_from_issue(d: dict) -> dict:
    """Extract vote/feedback tallies from a GitHub issue object."""
    rx = d.get("reactions") or {}
    up, down = int(rx.get("+1", 0)), int(rx.get("-1", 0))
    return {
        "votes_up": up,
        "votes_down": down,
        "score": up - down,
        "feedback_count": int(d.get("comments", 0)),
        "issue_url": d.get("html_url"),
    }


def sync_ideas(ideas: list[dict], repo: str, token: str, create_missing: bool = True) -> tuple[int, int]:
    """Ensure each idea has an issue and refresh its tallies (mutates ``ideas``).

    Returns ``(created, synced)``. Idempotent: ideas that already carry an
    ``issue_number`` are not re-created, just refreshed.
    """
    created = synced = 0
    for idea in ideas:
        number = idea.get("issue_number")
        if not number:
            if not create_missing:
                continue
            try:
                number, url = create_issue(repo, idea, token)
            except Exception as ex:
                log.warning("Could not create issue for idea '%s': %s", idea.get("name"), ex)
                continue
            idea["issue_number"] = number
            idea["issue_url"] = url
            created += 1
            log.info("Opened issue #%s for idea '%s'", number, idea.get("name"))
        try:
            d = fetch_issue(repo, number, token)
        except Exception as ex:
            log.warning("Could not fetch issue #%s: %s", number, ex)
            continue
        idea.update(tally_from_issue(d))
        synced += 1
    return created, synced


def sync(settings: Settings, repo: str, token: str, create_missing: bool = True) -> tuple[int, int]:
    """Load ideas.json, sync to GitHub Issues, and write the tallies back."""
    ideas = load_ideas(settings.ideas_path)
    if not ideas:
        log.info("No ideas to sync to GitHub Issues.")
        return 0, 0
    created, synced = sync_ideas(ideas, repo, token, create_missing=create_missing)
    write_json(settings.ideas_path, ideas)
    log.info("GitHub Issues sync on %s: +%d created, %d refreshed", repo, created, synced)
    return created, synced
