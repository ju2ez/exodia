"""Load run configuration from config.yaml (with sensible fallbacks)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal ``.env`` loader: ``KEY=VALUE`` lines into ``os.environ``.

    Dependency-free and non-overriding (an already-exported variable wins), so
    secrets like ``GEMINI_API_KEY`` / ``OPENAI_API_KEY`` / ``GITHUB_TOKEN`` placed
    in a local ``.env`` reach both this process and the AI-Scientist-v2 subprocess.
    """
    p = Path(path) if path else (paths.REPO_ROOT / ".env")
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


@dataclass
class Settings:
    upstream_repo: str = "jennyzzt/awesome-open-ended"
    upstream_branch: str = "main"

    site_title: str = "Exodia"
    site_base_url: str = ""
    github_repo: str = ""  # owner/repo hosting idea issues (votes/feedback backend)

    ideation_model: str = "gpt-4o-2024-05-13"
    ideation_max_num_generations: int = 3
    ideation_num_reflections: int = 3
    ideation_timeout_seconds: int = 1800

    enrich_max_new_fetches: int = 250
    enrich_request_delay_seconds: float = 3.0

    pdf_fetch: bool = False
    pdf_max_new_downloads: int = 250
    pdf_request_delay_seconds: float = 3.0

    citations_fetch: bool = True
    citations_max_new_fetches: int = 500
    citations_request_delay_seconds: float = 1.0

    # Mine the locally-cached full-text PDFs into the analysis corpus.
    fulltext_analyze: bool = True
    # Per-paper cap on extracted text. Set generously so the *whole* paper —
    # including the conclusion / future-work sections that live at the very end —
    # is captured (a typical paper is ~80-150k chars; 40k cut off ~95% of them
    # before their conclusion). The clustering view still uses title+abstract
    # only, so a large cap helps concept/entity recall without skewing themes.
    fulltext_max_chars: int = 400000

    # Fetch video transcripts (YouTube captions) into the analysis corpus.
    transcripts_fetch: bool = True
    transcripts_max_new_fetches: int = 100
    transcripts_request_delay_seconds: float = 1.0

    analysis_method: str = "tfidf_kmeans"
    analysis_use_llm: bool = False
    analysis_k_range: tuple[int, int] = (5, 9)

    # Idea<->paper realization matching and idea novelty dedup.
    match_threshold: float = 0.13
    novelty_threshold: float = 0.82

    # Model of Interestingness / forecasting backtest (manual, offline; never in CI).
    moi_oracle_model: str = "gemini-2.5-flash"  # modern model for the leakage-upper-bound arm
    moi_honest_models: dict = field(default_factory=lambda: {})  # {year: model_id} genuine ≤Y cutoffs
    moi_generations: int = 6
    moi_init_pop: int = 16
    moi_batch: int = 8
    moi_archive_shape: tuple[int, int] = (6, 3)  # concept-family × novelty-bin
    moi_hit_threshold: float = 0.18  # cosine ≥ this = a predicted idea "hit" a real paper
    moi_fitness_lambda: float = 0.5  # penalty on reward-ensemble disagreement (anti-hack)
    moi_reward_ensemble: int = 4
    moi_reward_pairs: int = 400
    moi_temperature: float = 0.7

    data_dir: Path = field(default_factory=lambda: paths.DEFAULT_DATA_DIR)
    site_dir: Path = field(default_factory=lambda: paths.DEFAULT_SITE_DIR)

    # --- Derived file locations -------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def kb_path(self) -> Path:
        return self.data_dir / "knowledge_base.json"

    @property
    def ideas_path(self) -> Path:
        return self.data_dir / "ideas.json"

    @property
    def themes_path(self) -> Path:
        return self.data_dir / "themes.json"

    @property
    def changelog_path(self) -> Path:
        return self.data_dir / "changelog.json"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def pdfs_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def moi_backtest_path(self) -> Path:
        return self.data_dir / "moi_backtest.json"

    @property
    def moi_cache_dir(self) -> Path:
        return self.data_dir / "moi_cache"

    @property
    def moi_model_path(self) -> Path:
        return self.data_dir / "moi_model.json"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def plots_dir(self) -> Path:
        return self.site_dir / "assets" / "plots"

    @property
    def upstream_url(self) -> str:
        return f"https://github.com/{self.upstream_repo}"


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load Settings from a YAML file, falling back to dataclass defaults."""
    path = Path(config_path) if config_path else paths.CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    s = Settings()
    s.upstream_repo = _get(raw, "upstream", "repo", default=s.upstream_repo)
    s.upstream_branch = _get(raw, "upstream", "branch", default=s.upstream_branch)

    s.site_title = _get(raw, "site", "title", default=s.site_title)
    s.site_base_url = _get(raw, "site", "base_url", default=s.site_base_url) or ""
    s.github_repo = _get(raw, "site", "repo", default=s.github_repo) or ""

    s.ideation_model = _get(raw, "ideation", "model", default=s.ideation_model)
    s.ideation_max_num_generations = int(
        _get(raw, "ideation", "max_num_generations", default=s.ideation_max_num_generations)
    )
    s.ideation_num_reflections = int(
        _get(raw, "ideation", "num_reflections", default=s.ideation_num_reflections)
    )
    s.ideation_timeout_seconds = int(
        _get(raw, "ideation", "timeout_seconds", default=s.ideation_timeout_seconds)
    )

    s.enrich_max_new_fetches = int(
        _get(raw, "enrich", "max_new_fetches", default=s.enrich_max_new_fetches)
    )
    s.enrich_request_delay_seconds = float(
        _get(raw, "enrich", "request_delay_seconds", default=s.enrich_request_delay_seconds)
    )

    s.pdf_fetch = bool(_get(raw, "pdfs", "fetch", default=s.pdf_fetch))
    s.pdf_max_new_downloads = int(
        _get(raw, "pdfs", "max_new_downloads", default=s.pdf_max_new_downloads)
    )
    s.pdf_request_delay_seconds = float(
        _get(raw, "pdfs", "request_delay_seconds", default=s.pdf_request_delay_seconds)
    )

    s.citations_fetch = bool(_get(raw, "citations", "fetch", default=s.citations_fetch))
    s.citations_max_new_fetches = int(
        _get(raw, "citations", "max_new_fetches", default=s.citations_max_new_fetches)
    )
    s.citations_request_delay_seconds = float(
        _get(raw, "citations", "request_delay_seconds", default=s.citations_request_delay_seconds)
    )

    s.fulltext_analyze = bool(_get(raw, "fulltext", "analyze", default=s.fulltext_analyze))
    s.fulltext_max_chars = int(_get(raw, "fulltext", "max_chars", default=s.fulltext_max_chars))

    s.transcripts_fetch = bool(_get(raw, "transcripts", "fetch", default=s.transcripts_fetch))
    s.transcripts_max_new_fetches = int(
        _get(raw, "transcripts", "max_new_fetches", default=s.transcripts_max_new_fetches)
    )
    s.transcripts_request_delay_seconds = float(
        _get(raw, "transcripts", "request_delay_seconds",
             default=s.transcripts_request_delay_seconds)
    )

    s.analysis_method = _get(raw, "analysis", "method", default=s.analysis_method)
    s.analysis_use_llm = bool(_get(raw, "analysis", "use_llm", default=s.analysis_use_llm))
    k_range = _get(raw, "analysis", "k_range", default=list(s.analysis_k_range))
    s.analysis_k_range = (int(k_range[0]), int(k_range[1]))

    s.match_threshold = float(_get(raw, "ideas", "match_threshold", default=s.match_threshold))
    s.novelty_threshold = float(_get(raw, "ideas", "novelty_threshold", default=s.novelty_threshold))

    s.moi_oracle_model = _get(raw, "moi", "oracle_model", default=s.moi_oracle_model)
    honest = _get(raw, "moi", "honest_models", default=None)
    if isinstance(honest, dict):
        s.moi_honest_models = {int(y): str(m) for y, m in honest.items()}
    s.moi_generations = int(_get(raw, "moi", "generations", default=s.moi_generations))
    s.moi_init_pop = int(_get(raw, "moi", "init_pop", default=s.moi_init_pop))
    s.moi_batch = int(_get(raw, "moi", "batch", default=s.moi_batch))
    s.moi_hit_threshold = float(_get(raw, "moi", "hit_threshold", default=s.moi_hit_threshold))
    s.moi_fitness_lambda = float(_get(raw, "moi", "fitness_lambda", default=s.moi_fitness_lambda))
    s.moi_reward_ensemble = int(_get(raw, "moi", "reward_ensemble", default=s.moi_reward_ensemble))
    s.moi_reward_pairs = int(_get(raw, "moi", "reward_pairs", default=s.moi_reward_pairs))
    s.moi_temperature = float(_get(raw, "moi", "temperature", default=s.moi_temperature))

    s.data_dir = paths.resolve(_get(raw, "paths", "data_dir", default="data"))
    s.site_dir = paths.resolve(_get(raw, "paths", "site_dir", default="site"))
    return s
