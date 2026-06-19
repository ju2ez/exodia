"""Curated concept gazetteer for the open-endedness field.

Pure TF-IDF over full papers surfaces generic words ("opportunity", "future",
"performance") that aren't really *concepts*. This module instead detects a
hand-curated set of real research concepts (LLMs, open-ended learning, recursive
self-improvement, quality-diversity, novelty search, …) by matching their
aliases against each entry's text (:mod:`corpus`), so what we track and plot is
always sensible.

Each concept maps to a list of alias phrases matched as whole words/phrases
(case-insensitive, word-boundary), so "llm" also catches "llms" but "albert"
never matches "bert".
"""

from __future__ import annotations

import re
from collections import Counter

from .config import Settings
from .corpus import corpus_text
from .models import Entry

# Canonical concept -> alias phrases (lowercase). Order is roughly by specificity.
CONCEPTS: dict[str, list[str]] = {
    "Open-ended learning": ["open-ended learning", "open ended learning", "open-endedness",
                            "open endedness", "open-ended"],
    "Large language models (LLMs)": ["large language model", "llm", "foundation model",
                                     "language model", "gpt", "transformer"],
    "Recursive self-improvement": ["recursive self-improvement", "recursively self-improv",
                                   "self-improvement", "self-improving", "self improvement"],
    "Quality-diversity": ["quality-diversity", "quality diversity", "illumination algorithm",
                          "qd algorithm"],
    "MAP-Elites": ["map-elites", "map elites"],
    "Novelty search": ["novelty search", "novelty-search"],
    "Intrinsic motivation & curiosity": ["intrinsic motivation", "intrinsic reward", "curiosity",
                                         "curiosity-driven"],
    "Auto-curricula": ["auto-curricul", "autocurricul", "automatic curricul", "curriculum learning",
                       "curriculum generation"],
    "Self-play": ["self-play", "self play"],
    "Coevolution": ["coevolution", "co-evolution", "competitive evolution"],
    "POET / paired open-ended": ["poet", "paired open-ended", "minimal criterion coevolution"],
    "Evolutionary computation": ["evolutionary algorithm", "evolutionary computation",
                                 "genetic algorithm", "neuroevolution", "evolution strateg"],
    "Reinforcement learning": ["reinforcement learning", "policy gradient", "q-learning",
                               "deep rl", "actor-critic"],
    "World models": ["world model"],
    "Generative models": ["generative model", "diffusion model", "generative adversarial",
                          "variational autoencoder"],
    "In-context learning": ["in-context learning", "in context learning", "few-shot"],
    "Multi-agent systems": ["multi-agent", "multi agent", "multiagent"],
    "Embodied agents": ["embodied", "embodiment"],
    "Autonomous / LLM agents": ["autonomous agent", "llm agent", "agentic", "language agent"],
    "Exploration": ["exploration", "explorative", "explore novel"],
    "Procedural content generation": ["procedural content generation", "procedural generation",
                                      "pcg"],
    "Emergent abilities": ["emergent abilit", "emergent capabilit", "emergent behavior",
                           "emergence of"],
    "Meta-learning": ["meta-learning", "meta learning", "learning to learn"],
    "Lifelong & continual learning": ["lifelong learning", "continual learning",
                                      "never-ending learning", "lifelong"],
    "Program synthesis & code generation": ["program synthesis", "code generation",
                                            "code synthesis", "programming by"],
    "Autotelic / goal generation": ["autotelic", "goal generation", "self-generated goal",
                                    "goal-conditioned"],
    "AI-generating algorithms (AI-GAs)": ["ai-generating algorithm", "ai generating algorithm",
                                          "ai-ga"],
    "Artificial life": ["artificial life", "alife", "a-life"],
    "Self-organization": ["self-organization", "self-organizing", "self-organisation"],
    "Diversity-driven search": ["divergent search", "behavioral diversity", "behavioural diversity",
                                "diversity-driven", "diversity of solutions"],
    "Compositionality": ["compositional", "compositionality"],
    "Automated scientific discovery": ["scientific discovery", "automated science", "ai scientist",
                                       "automated research", "hypothesis generation"],
}


def concept_matchers() -> dict[str, re.Pattern]:
    """Compiled whole-word/phrase matchers, one per curated concept."""
    return {
        label: re.compile("|".join(r"\b" + re.escape(a) for a in aliases), re.IGNORECASE)
        for label, aliases in CONCEPTS.items()
    }


def detect_concepts(entries: list[Entry], settings: Settings) -> list[dict]:
    """Count entries mentioning each curated concept (with citation weighting).

    Returns a list of ``{"concept", "docs", "share", "citations"}`` sorted by doc
    count desc, keeping only concepts that actually appear.
    """
    matchers = concept_matchers()
    total = len(entries) or 1
    docs: Counter[str] = Counter()
    cites: Counter[str] = Counter()
    for e in entries:
        text = corpus_text(e, settings)
        c = e.citation_count or 0
        for label, rx in matchers.items():
            if rx.search(text):
                docs[label] += 1
                cites[label] += c
    out = [
        {"concept": label, "docs": docs[label], "share": round(docs[label] / total, 4),
         "citations": cites[label]}
        for label in CONCEPTS
        if docs[label] > 0
    ]
    out.sort(key=lambda d: (d["docs"], d["citations"]), reverse=True)
    return out
