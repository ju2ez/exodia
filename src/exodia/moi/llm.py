"""A thin, cache-backed LLM client for the MOI generator.

The main pipeline shells out to AI-Scientist-v2 (subprocess, minutes per idea) —
far too heavy for a quality-diversity loop that makes hundreds of short calls. So
this is a minimal OpenAI-compatible chat client built on ``requests`` (the same
stack as :mod:`enrich`/:mod:`citations`), reaching Gemini via its
OpenAI-compatible endpoint exactly as the rest of the project does.

Two design points make the backtest reproducible and testable:
- **Disk cache**: every completion is content-addressed (model + prompts + temp +
  seed) under ``data/moi_cache/`` (git-ignored). Reruns are nearly free.
- **Injectable transport**: tests pass a deterministic ``transport`` callable, so
  the whole suite runs offline with no API key and no network.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..logging_setup import get_logger
from ..util import stable_id

log = get_logger(__name__)

# OpenAI-compatible endpoints. Gemini exposes one; the project already relies on it.
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENAI_BASE = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    base_url: str = GEMINI_BASE
    api_key_env: str = "GEMINI_API_KEY"
    knowledge_cutoff: str = ""  # "YYYY-MM" (documented training cutoff), best-effort


Transport = Callable[[ModelSpec, dict], str]


def _http_transport(spec: ModelSpec, payload: dict) -> str:
    import requests

    key = os.environ.get(spec.api_key_env)
    if not key:
        raise RuntimeError(
            f"{spec.api_key_env} not set — needed to call {spec.id}. "
            "Use --mock for an offline run."
        )
    resp = requests.post(
        f"{spec.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


class LLMClient:
    """Cache-first chat client. ``transport`` is injectable for offline tests."""

    def __init__(self, spec: ModelSpec, cache_dir: Path, *, transport: Transport | None = None,
                 seed: int = 0, temperature: float = 0.7):
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.transport = transport or _http_transport
        self.seed = seed
        self.temperature = temperature
        self.n_calls = 0
        self.n_cache_hits = 0

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def complete(self, system: str, user: str, *, max_tokens: int = 1200,
                 temperature: float | None = None) -> str:
        temp = self.temperature if temperature is None else temperature
        key = stable_id(self.spec.id, system, user, f"{temp:.2f}", str(self.seed), str(max_tokens))
        cp = self._cache_path(key)
        if cp.exists():
            self.n_cache_hits += 1
            return json.loads(cp.read_text(encoding="utf-8"))["content"]

        payload = {
            "model": self.spec.id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        content = self.transport(self.spec, payload)
        self.n_calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"content": content}), encoding="utf-8")
        return content
