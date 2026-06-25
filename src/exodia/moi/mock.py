"""A deterministic offline LLM transport for tests and ``--mock`` smoke runs.

Returns a parseable JSON array of plausible ideas (mentioning real curated
concepts so the archive/concept logic is exercised), seeded by the prompt so runs
are reproducible and require no API key or network.
"""

from __future__ import annotations

import json
import re

from ..util import stable_id

_TOPICS = [
    ("LLM agents", "autonomous llm agents that design their own curricula"),
    ("recursive self-improvement", "agents that recursively self-improve their own code"),
    ("quality-diversity", "quality-diversity search over open-ended environments"),
    ("unsupervised environment design", "regret-based unsupervised environment design"),
    ("open-ended learning", "open-ended learning toward generally capable agents"),
    ("automated scientific discovery", "ai scientist automating open-ended discovery"),
    ("cultural accumulation", "cultural accumulation across generations of agents"),
    ("world models", "learned world models for open-ended exploration"),
    ("novelty search", "novelty search with intrinsic motivation and curiosity"),
    ("foundation model self-play", "open-ended self-play between foundation models"),
]


def mock_transport(spec, payload: dict) -> str:
    user = next((m["content"] for m in payload["messages"] if m["role"] == "user"), "")
    seed = int(stable_id(spec.id, user, length=8), 16)
    m = re.search(r"[Pp]ropose (\d+)", user)
    n = 1 if "single object" in user else (int(m.group(1)) if m else 3)
    ideas = []
    for i in range(n):
        topic, blurb = _TOPICS[(seed + i) % len(_TOPICS)]
        ideas.append({
            "name": f"{topic.replace(' ', '-')}-{(seed + i) % 97}",
            "title": f"Toward {topic}: {blurb}",
            "short_hypothesis": f"We hypothesize that {blurb} yields more open-ended progress.",
            "abstract": (f"This direction explores {blurb}. It builds on prior open-endedness "
                         f"work and proposes a concrete research agenda around {topic}."),
        })
    return json.dumps(ideas)
