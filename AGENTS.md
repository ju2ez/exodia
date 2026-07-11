# AGENTS.md

Guidance for coding agents working in this repository.

## Project

**exodia** — "Distilling SotA open-endedness through open-endedness."

A self-updating pipeline + website that watches Jenny Zhang's curated
[`awesome-open-ended`](https://github.com/jennyzzt/awesome-open-ended) list,
distills it into a structured knowledge base, generates/summarizes research ideas
with AI-Scientist-v2, and publishes the result. See `README.md` for the full loop.

## Tooling

- **Python ≥ 3.11**, packaged with `pyproject.toml` (setuptools, `src/` layout).
- **Package manager: `uv`.** Setup: `uv venv && uv pip install -e ".[dev]"`.
- **Tests: pytest.** All: `.venv/bin/python -m pytest`. Single:
  `.venv/bin/python -m pytest tests/test_webapp.py::test_vote_best_and_me`.
- **Lint: ruff.** `.venv/bin/ruff check src/exodia tests` (line-length 100; rules E,F,I,UP,B).
- Network-touching code (arXiv enrich/PDFs) is exercised via monkeypatched/fixture
  tests, so the suite runs offline in a few seconds.

## Architecture

CLI entry: `python -m exodia <cmd>` (`cli.py`). Commands: `gate`, `run-all`,
`render`, `serve`, `sync-issues`, `moi-backtest`, `moi-train`. `cli.main` calls
`config.load_dotenv()` first, so a local `.env` (git-ignored; see `.env.example`)
supplies `GEMINI_API_KEY` / `OPENAI_API_KEY` / `GITHUB_TOKEN` /
`EXODIA_AISCIENTIST_DIR` to the process and the AI-Scientist-v2 subprocess. The full
loop (`run_all` in `pipeline.py`):
parse → merge (`store.py`) → enrich abstracts+journal_ref (`enrich.py`) → download
PDFs (`pdfs.py`) → resolve real venue (`venues.py`) → analyze themes (`analysis.py`)
→ ideate (`ideation.py`, content-based ids + novelty filter) → star-match new papers
to ideas (`matching.py`) → changelog (`diffing.py`) → render.

Idea generation defaults to **Google Gemini** (`ideation.model` in `config.yaml`;
needs `GEMINI_API_KEY`). AI-Scientist-v2 reaches Gemini via its OpenAI-compatible
endpoint, so only the key + a `gemini-*` model id are needed; other provider model
ids work with their respective keys.

Data: committed JSON under `data/` (`knowledge_base.json`, `ideas.json`,
`themes.json`, `changelog.json`, `state.json`). **Not committed / git-ignored:**
`data/pdfs/` (full-text PDFs, analysis-only) and `data/community.db` (FastAPI votes).

**Voting/feedback = GitHub Issues** (`issues.py`, `sync-issues`): one issue per
idea, 👍/👎 reactions are votes, comments are feedback. The pipeline syncs ideas →
issues (idempotent via the `issue_number` stored on each idea + a `<!-- exodia-idea:
<id> -->` body marker) and reads tallies back into `ideas.json`, so the **static**
Pages site needs no server. Two ways to serve the same data + Jinja templates
(`templates/`), sharing `render.site_context()`:
- **Static** (`render.py` → `site/`): shows vote tallies + a ranked best-idea list,
  links out to each issue to vote/discuss. This is what the Action deploys to Pages.
- **Dynamic** (`webapp.py`, FastAPI + `db.py` SQLite): optional *local* mode with
  in-page voting/feedback. The ideas template branches on `dynamic`.

Plots are interactive **Plotly** HTML embedded inline (`plotting.py`) — no PNG files;
plotly.js loads once from the CDN. Config lives in `config.yaml` (`config.py`
`Settings`).

## Conventions worth keeping

- **Idea ids are content-based** (`stable_id(name, title)`) so re-generated ideas
  dedupe across runs and their votes persist — do not reintroduce run-id into the id.
- **arXiv is a preprint server, not a venue.** Use `Entry.venue_display` /
  `venues.resolve_venue`; never count raw "arXiv" as a publication venue.
- New `Entry`/`Idea` fields must have defaults (`models._from_dict` ignores unknown
  keys and fills missing ones — keeps committed JSON forward/backward compatible).
- Don't republish upstream prose or re-host PDFs; link back. See `NOTICE`.
