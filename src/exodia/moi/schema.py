"""Dataclasses for the MOI backtest, driving ``data/moi_backtest.json``.

Serialization mirrors the rest of the repo: ``asdict`` for writing, a
``_from_dict``-style loader that ignores unknown keys for forward/backward
compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

SCHEMA_VERSION = 1

T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[call-arg]


@dataclass
class GenIdea:
    """One generated research direction (a candidate prediction)."""

    idea_id: str
    name: str
    title: str
    short_hypothesis: str
    abstract: str
    concepts: list[str] = field(default_factory=list)  # curated concepts the idea names
    bd: tuple[int, int] = (0, 0)  # MAP-Elites cell coords (concept-family, novelty-bin)
    fitness: float = 0.0  # MOI reward
    generation: int = 0
    parents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bd"] = list(self.bd)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenIdea:
        obj = _from_dict(cls, data)
        if isinstance(obj.bd, list):
            obj.bd = tuple(obj.bd)  # type: ignore[assignment]
        return obj


@dataclass
class GenerationResult:
    """The population produced by one (cutoff, arm, mode) run."""

    arm: str  # "honest" | "oracle"
    mode: str  # "steered" | "baseline"
    model: str
    ideas: list[GenIdea] = field(default_factory=list)
    n_llm_calls: int = 0
    n_cache_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "mode": self.mode, "model": self.model,
            "ideas": [i.to_dict() for i in self.ideas],
            "n_llm_calls": self.n_llm_calls, "n_cache_hits": self.n_cache_hits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationResult:
        obj = _from_dict(cls, {k: v for k, v in data.items() if k != "ideas"})
        obj.ideas = [GenIdea.from_dict(d) for d in data.get("ideas", [])]
        return obj


@dataclass
class CutoffResult:
    cutoff_year: int
    n_train_le_y: int
    n_test_gt_y: int
    runs: list[GenerationResult] = field(default_factory=list)
    metrics: dict[str, dict] = field(default_factory=dict)  # "<arm>.<mode>" -> metric dict
    examples: list[dict] = field(default_factory=list)  # correct-prediction examples
    concept_table: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cutoff_year": self.cutoff_year,
            "n_train_le_y": self.n_train_le_y,
            "n_test_gt_y": self.n_test_gt_y,
            "runs": [r.to_dict() for r in self.runs],
            "metrics": self.metrics,
            "examples": self.examples,
            "concept_table": self.concept_table,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CutoffResult:
        obj = _from_dict(cls, {k: v for k, v in data.items() if k != "runs"})
        obj.runs = [GenerationResult.from_dict(d) for d in data.get("runs", [])]
        return obj


@dataclass
class MoiBacktest:
    schema_version: int = SCHEMA_VERSION
    generated_utc: str = ""
    config: dict = field(default_factory=dict)
    cutoffs: list[CutoffResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_utc": self.generated_utc,
            "config": self.config,
            "cutoffs": [c.to_dict() for c in self.cutoffs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoiBacktest:
        obj = _from_dict(cls, {k: v for k, v in data.items() if k != "cutoffs"})
        obj.cutoffs = [CutoffResult.from_dict(d) for d in data.get("cutoffs", [])]
        return obj
