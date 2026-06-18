"""Detect upstream changes and fetch the upstream README.

A single cheap GitHub API call gets the latest commit SHA for the upstream
branch; the pipeline runs the expensive stages only when it differs from the
SHA stored in data/state.json. The README is fetched from raw.githubusercontent
pinned to the resolved SHA so a run is reproducible even if upstream changes
mid-run.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import requests

from .logging_setup import get_logger
from .models import State
from .util import read_json, write_json

log = get_logger(__name__)

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"
USER_AGENT = "exodia/0.1 (+https://github.com/ju2ez/exodia)"


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    h = {"User-Agent": USER_AGENT, "Accept": accept}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get_latest_sha(repo: str, branch: str, timeout: int = 30) -> str:
    """Return the latest commit SHA of repo@branch via the GitHub REST API."""
    url = f"{GITHUB_API}/repos/{repo}/commits/{branch}"
    r = requests.get(url, headers=_headers("application/vnd.github.sha"), timeout=timeout)
    r.raise_for_status()
    return r.text.strip()


def fetch_readme(repo: str, ref: str, timeout: int = 30) -> str:
    """Fetch README.md from raw.githubusercontent at the given ref (sha/branch)."""
    url = f"{RAW_BASE}/{repo}/{ref}/README.md"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text


_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def is_sha(token: str | None) -> bool:
    return bool(token and _SHA_RE.match(token))


def content_token(md: str) -> str:
    """A change token derived from README content (API-independent fallback)."""
    return "content:" + hashlib.sha256(md.encode("utf-8")).hexdigest()[:16]


def resolve_change_token(repo: str, branch: str) -> tuple[str, str | None]:
    """Return (token, readme_or_None).

    Prefer the commit SHA via the GitHub API (cheap, exact). If the API is
    unavailable (rate-limited / blocked), fall back to fetching the README and
    hashing its content — still a reliable change signal, no API needed.
    """
    try:
        return get_latest_sha(repo, branch), None
    except Exception as ex:
        log.warning("GitHub API unavailable (%s); using README content hash", ex)
        md = fetch_readme(repo, branch)
        return content_token(md), md


def fetch_upstream(repo: str, branch: str) -> tuple[str, str]:
    """Fetch the README and a change token in as few requests as possible.

    Returns (readme_markdown, token). token is a commit SHA when the API is
    reachable, else a 'content:'-prefixed hash.
    """
    try:
        sha = get_latest_sha(repo, branch)
        return fetch_readme(repo, sha), sha
    except Exception as ex:
        log.warning("GitHub API unavailable (%s); fetching README by branch", ex)
        md = fetch_readme(repo, branch)
        return md, content_token(md)


def load_state(path: str | Path, repo: str) -> State:
    data = read_json(path, default=None)
    if not data:
        return State(upstream_repo=repo)
    return State.from_dict(data)


def save_state(state: State, path: str | Path) -> None:
    write_json(path, state.to_dict())


def is_changed(state: State, latest_sha: str) -> bool:
    """True on the first run (no stored sha) or when the sha differs."""
    return state.upstream_sha != latest_sha


def compare_url(repo: str, from_sha: str | None, to_sha: str | None) -> str | None:
    # Only build links for real commit SHAs (content-hash tokens are not linkable).
    if not is_sha(to_sha):
        return None
    if is_sha(from_sha) and from_sha != to_sha:
        return f"https://github.com/{repo}/compare/{from_sha}...{to_sha}"
    return f"https://github.com/{repo}/commit/{to_sha}"
