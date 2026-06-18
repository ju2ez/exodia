"""Centralized filesystem paths.

The package lives at ``<repo>/src/exodia``; templates, assets, config, and the
committed ``data/`` directory live at the repo root. We resolve the repo root
relative to this file so the pipeline works whether invoked from the repo root,
from an editable install, or from CI.
"""

from __future__ import annotations

from pathlib import Path

# <repo>/src/exodia/paths.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = REPO_ROOT / "src"
TEMPLATES_DIR = REPO_ROOT / "templates"
ASSETS_DIR = REPO_ROOT / "assets"
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Default data/site dirs; config.yaml may override (see config.Settings).
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SITE_DIR = REPO_ROOT / "site"


def resolve(path_like: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)
