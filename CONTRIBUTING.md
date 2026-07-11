# Contributing to Exodia

Exodia grows slowly and deliberately — quality over quantity. The most valuable contribution is **judgment**, not code.

## The most useful thing you can do: vote and critique ideas

Every generated research idea gets one GitHub Issue (created by the pipeline, labeled accordingly). On the [Ideas page](https://julianhatzky.me/exodia/ideas.html), each idea links to its issue:

- **👍 / 👎 reactions on the issue = votes.** Tallies are read back by the pipeline and baked into the site on the next run (daily) — a ranked "best ideas" list comes from exactly these votes.
- **Comments = feedback.** Why an idea is derivative, what prior work it misses, how you'd sharpen it — critical comments are worth more than praise. If an idea was already realized by a published paper, say so with a link; the pipeline stars ideas that reality catches up with.

## Bugs and improvements

Open a [bug report](https://github.com/ju2ez/exodia/issues/new/choose) for pipeline or site problems. Small, focused PRs are welcome:

```bash
uv venv && uv pip install -e ".[dev]"
python -m exodia run-all --dry-run   # full pipeline, no API cost (fixture ideation)
ruff check src tests && pytest       # both must pass (CI enforces)
```

State lives as committed JSON under `data/` — PRs should generally not touch it; the pipeline regenerates it.

## Suggesting papers

The knowledge base deliberately follows one curated source: [`awesome-open-ended`](https://github.com/jennyzzt/awesome-open-ended) by Jenny Zhang. Suggest new papers **upstream** there — exodia picks them up automatically on the next poll.

## What NOT to expect

- Feature-scope creep: exodia stays a distillation loop; ideas that turn it into a general paper-manager will be declined kindly.
- Instant vote tallies: the site is static; tallies refresh with the daily pipeline run.
