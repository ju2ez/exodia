"""Year-Y-constrained idea generators: the HONEST and ORACLE arms.

- **HONEST**: a model whose genuine knowledge cutoff is ≤ Y, given only a "you are
  a researcher in year Y" role prompt — its only knowledge of the field is
  parametric. The cleanest (if weaker) horizon.
- **ORACLE**: a modern strong model fed a RAG context built *only* from ≤Y papers
  and told to role-play year Y. A leakage *upper bound* — it cannot truly forget
  the future, so its lead over HONEST measures contamination, not foresight.

Both emit a strict JSON array of ideas; parsing is tolerant.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..concepts import concept_matchers
from ..config import Settings
from ..logging_setup import get_logger
from ..models import Entry
from ..util import stable_id
from .features import NoveltySpace
from .llm import GEMINI_BASE, OPENAI_BASE, LLMClient, ModelSpec
from .schema import GenIdea

log = get_logger(__name__)


def spec_for_model_id(model_id: str, cutoff: str = "") -> ModelSpec:
    """Infer the OpenAI-compatible endpoint + key env from a model id."""
    low = model_id.lower()
    if low.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "davinci", "text-")):
        return ModelSpec(model_id, OPENAI_BASE, "OPENAI_API_KEY", cutoff)
    return ModelSpec(model_id, GEMINI_BASE, "GEMINI_API_KEY", cutoff)


def pick_honest_model(year: int, settings: Settings) -> ModelSpec:
    """Strongest configured model whose horizon is ≤ ``year`` (from moi_honest_models)."""
    registry: dict = settings.moi_honest_models or {}
    # keys may be ints or strings; pick the entry for the largest year <= cutoff.
    eligible = sorted((int(y), m) for y, m in registry.items() if int(y) <= year)
    if eligible:
        _, model_id = eligible[-1]
        return spec_for_model_id(model_id, cutoff=f"{year}-12")
    # Fallback: use the oracle model id but flag the missing horizon.
    log.warning("MOI: no honest model configured for year<=%d; falling back to oracle id", year)
    return spec_for_model_id(settings.moi_oracle_model, cutoff="")


def oracle_spec(settings: Settings) -> ModelSpec:
    return spec_for_model_id(settings.moi_oracle_model, cutoff="modern")


def build_landscape(train_entries: list[Entry], year: int, *, limit: int = 25) -> str:
    """A compact ≤Y landscape brief (title + venue/year + abstract snippet)."""
    chosen = sorted(train_entries, key=lambda e: (1 if e.abstract else 0, e.year or 0),
                    reverse=True)[:limit]
    lines = [f"Open-endedness research as of year {year} (selected works):"]
    for e in chosen:
        meta = ", ".join(filter(None, [e.venue or None, str(e.year) if e.year else None]))
        snip = (e.abstract or "")[:280]
        lines.append(f"- {e.title}" + (f" ({meta})" if meta else "") + (f" — {snip}" if snip else ""))
    return "\n".join(lines)


def build_rag_context(train_entries: list[Entry], query: str, space: NoveltySpace,
                      k: int = 12) -> str:
    """Top-k ≤Y papers most similar to ``query`` (title + abstract + year only)."""
    from sklearn.metrics.pairwise import cosine_similarity

    vq = space.vectorizer.transform([query or "open-endedness"])
    sims = cosine_similarity(vq, space.matrix)[0]
    order = sims.argsort()[::-1][:k]
    out = []
    for i in order:
        e = train_entries[i]
        out.append(f"- ({e.year}) {e.title} — {(e.abstract or '')[:300]}")
    return "\n".join(out)


_SYSTEM = (
    "You are an open-endedness AI researcher writing in the year {year}. You only "
    "know work published on or before December {year}. Do NOT use any knowledge of "
    "developments after {year}. Propose concrete, novel research directions that "
    "you believe will emerge and matter in the years AFTER {year}.\n"
    "Respond with ONLY a JSON array (no markdown, no commentary). Each element is an "
    'object with keys: "name" (short slug), "title", "short_hypothesis" (1 sentence), '
    '"abstract" (2 sentences). Keep each idea short so the JSON is complete and valid.'
)


def _coerce_str(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(_coerce_str(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_coerce_str(x) for x in v.values())
    return "" if v is None else str(v)


def _extract_json_array(text: str) -> list[dict]:
    text = (text or "").strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("ideas") or obj.get("directions") or [obj]
        return [o for o in obj if isinstance(o, dict)]
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return [o for o in json.loads(m.group(0)) if isinstance(o, dict)]
        except Exception:
            pass
    return _salvage_objects(text)


def _salvage_objects(text: str) -> list[dict]:
    """Recover complete ``{...}`` objects from a truncated/invalid JSON array.

    Thinking-heavy models sometimes cut the array off mid-element; we still want
    the ideas that *did* complete. Scan for balanced top-level braces and parse
    each block independently.
    """
    out: list[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    pass
                start = -1
    return out


def parse_gen_ideas(raw: str, *, generation: int = 0, parents: list[str] | None = None) -> list[GenIdea]:
    matchers = concept_matchers()
    ideas: list[GenIdea] = []
    for o in _extract_json_array(raw):
        name = _coerce_str(o.get("name") or o.get("title"))[:80]
        title = _coerce_str(o.get("title") or o.get("name"))[:200]
        hyp = _coerce_str(o.get("short_hypothesis") or o.get("hypothesis"))
        abstract = _coerce_str(o.get("abstract"))
        if not (title or hyp):
            continue
        text = " ".join([title, hyp, abstract])
        concepts = [label for label, rx in matchers.items() if rx.search(text)]
        ideas.append(GenIdea(
            idea_id=stable_id(name, title), name=name, title=title,
            short_hypothesis=hyp, abstract=abstract, concepts=concepts,
            generation=generation, parents=parents or [],
        ))
    return ideas


@dataclass
class Generator:
    """Wraps an LLM client + the year-Y context for one arm."""

    client: LLMClient
    year: int
    arm: str  # "honest" | "oracle"
    landscape: str
    rag: str = ""

    def _system(self) -> str:
        return _SYSTEM.format(year=self.year)

    def _context_block(self) -> str:
        # HONEST arm gets no corpus (parametric horizon only); ORACLE gets ≤Y RAG.
        if self.arm == "oracle" and self.rag:
            return f"\nPapers you have read (all published on or before {self.year}):\n{self.rag}\n"
        return ""

    def propose(self, n: int) -> list[GenIdea]:
        user = (
            f"It is {self.year}. Brief on the field so far:\n{self.landscape}\n"
            f"{self._context_block()}\n"
            f"Propose {n} distinct, concrete research directions likely to emerge after {self.year}."
        )
        # Generous budget: Gemini "thinking" tokens count against max_tokens, so a
        # small cap truncates the JSON mid-array. 8k leaves room for thinking + output.
        raw = self.client.complete(self._system(), user, max_tokens=8000)
        return parse_gen_ideas(raw, generation=0)

    def mutate(self, parent: GenIdea, cell_hint: str, generation: int) -> list[GenIdea]:
        user = (
            f"It is {self.year}. A promising direction:\n"
            f"Title: {parent.title}\nHypothesis: {parent.short_hypothesis}\n{self._context_block()}\n"
            f"Produce 1 bolder, more specific variant that pushes toward: {cell_hint}. "
            "Return a JSON array with a single object."
        )
        raw = self.client.complete(self._system(), user, max_tokens=2048)
        return parse_gen_ideas(raw, generation=generation, parents=[parent.idea_id])

    def crossover(self, a: GenIdea, b: GenIdea, generation: int) -> list[GenIdea]:
        user = (
            f"It is {self.year}. Combine the key insights of these two directions into one "
            f"novel direction:\nA: {a.title} — {a.short_hypothesis}\n"
            f"B: {b.title} — {b.short_hypothesis}\n{self._context_block()}\n"
            "Return a JSON array with a single object."
        )
        raw = self.client.complete(self._system(), user, max_tokens=2048)
        return parse_gen_ideas(raw, generation=generation, parents=[a.idea_id, b.idea_id])
