"""MAP-Elites archive for the steering loop.

Quality-diversity "illumination": keep the best idea (by MOI fitness) in each cell
of a 2-D behaviour grid — concept-family × novelty-bin — so the agent explores a
*diverse* set of predicted directions instead of collapsing onto one. This is the
OMNI / MAP-Elites pattern: novelty defines the niches, the MOI defines quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import GenIdea

# Collapse the curated concept gazetteer into ~6 super-families (the first axis).
FAMILIES: list[tuple[str, set[str]]] = [
    ("Quality-diversity & evolution", {
        "Quality-diversity", "MAP-Elites", "Novelty search", "Evolutionary computation",
        "Coevolution", "Diversity-driven search", "Self-play"}),
    ("RL, agents & curricula", {
        "Reinforcement learning", "Multi-agent systems", "Embodied agents", "Exploration",
        "Auto-curricula", "Unsupervised environment design"}),
    ("LLMs & foundation models", {
        "Large language models (LLMs)", "In-context learning", "Autonomous / LLM agents",
        "Program synthesis & code generation", "Generative models"}),
    ("Open-ended learning & innovation", {
        "Open-ended learning", "POET / paired open-ended", "AI-generating algorithms (AI-GAs)",
        "Emergent abilities", "Autotelic / goal generation", "Intrinsic motivation & curiosity"}),
    ("World models & automated science", {
        "World models", "Automated scientific discovery", "Recursive self-improvement",
        "Meta-learning", "Lifelong & continual learning"}),
    ("Artificial life & other", {
        "Artificial life", "Cellular automata & self-replication", "Self-organization",
        "Cultural accumulation", "Compositionality", "Reward & objective design",
        "Red-teaming & adversarial generation", "Procedural content generation"}),
]
OTHER_FAMILY = len(FAMILIES) - 1  # unmatched concepts land in the last family

FAMILY_NAMES = [name for name, _ in FAMILIES]


def family_of(concepts: list[str]) -> int:
    for label in concepts:
        for idx, (_, members) in enumerate(FAMILIES):
            if label in members:
                return idx
    return OTHER_FAMILY


def novelty_bin(novelty: float, n_bins: int = 3) -> int:
    """derivative (0) / incremental (1) / novel (2) from novelty = 1 - nn_sim."""
    if novelty < 0.4:
        return 0
    if novelty < 0.7:
        return 1
    return min(n_bins - 1, 2)


def behavior_descriptor(concepts: list[str], novelty: float, n_bins: int = 3) -> tuple[int, int]:
    return (family_of(concepts), novelty_bin(novelty, n_bins))


@dataclass
class Archive:
    n_families: int = len(FAMILIES)
    n_bins: int = 3
    cells: dict[tuple[int, int], GenIdea] = field(default_factory=dict)

    def add(self, idea: GenIdea) -> bool:
        """Insert if its cell is empty or it beats the incumbent. Returns True if placed."""
        cell = idea.bd
        cur = self.cells.get(cell)
        if cur is None or idea.fitness > cur.fitness:
            self.cells[cell] = idea
            return True
        return False

    def elites(self) -> list[GenIdea]:
        return list(self.cells.values())

    def coverage(self) -> float:
        return len(self.cells) / float(self.n_families * self.n_bins)

    def qd_score(self) -> float:
        return sum(i.fitness for i in self.cells.values())

    def max_fitness(self) -> float:
        return max((i.fitness for i in self.cells.values()), default=0.0)

    def sample_parents(self, rng, n: int) -> list[GenIdea]:
        elites = self.elites()
        if not elites:
            return []
        idx = rng.integers(0, len(elites), size=n)
        return [elites[i] for i in idx]

    def stats(self) -> dict:
        return {"coverage": round(self.coverage(), 3), "qd_score": round(self.qd_score(), 3),
                "max_fitness": round(self.max_fitness(), 3), "filled": len(self.cells)}
