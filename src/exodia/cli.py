"""Command-line interface: ``exodia <subcommand>`` / ``python -m exodia ...``."""

from __future__ import annotations

import argparse
import os

from . import pipeline
from .config import load_dotenv, load_settings
from .logging_setup import get_logger, setup_logging

log = get_logger(__name__)


def _emit_github_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="exodia", description="Exodia pipeline.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="Check upstream SHA; emit changed=true/false.")
    g.add_argument("--force", default="false", help="Force changed=true.")

    r = sub.add_parser("run-all", help="Run the full pipeline.")
    r.add_argument("--force", action="store_true", help="Run even if upstream is unchanged.")
    r.add_argument("--dry-run", action="store_true", help="Use the idea fixture; no LLM call.")
    r.add_argument("--no-enrich", action="store_true", help="Skip arXiv abstract enrichment.")
    r.add_argument("--pdfs", action="store_true", help="Download full-text PDFs (override config).")
    r.add_argument("--no-pdfs", action="store_true", help="Skip full-text PDF download (override config).")
    r.add_argument("--max-pdfs", type=int, default=None, help="Override the per-run PDF download cap.")
    r.add_argument("--readme-file", default=None, help="Use a local README instead of fetching.")
    r.add_argument("--ais-dir", default=None, help="Path to an AI-Scientist-v2 clone.")
    r.add_argument("--max-generations", type=int, default=None, help="Override idea count.")

    sub.add_parser("render", help="Rebuild the site from committed data.")

    sv = sub.add_parser("serve", help="Run the dynamic site (FastAPI: voting + feedback).")
    sv.add_argument("--host", default="127.0.0.1", help="Bind host.")
    sv.add_argument("--port", type=int, default=8000, help="Bind port.")

    si = sub.add_parser("sync-issues", help="Create/refresh a GitHub Issue per idea; pull vote+comment tallies.")
    si.add_argument("--repo", default=None, help="owner/repo hosting the idea issues.")
    si.add_argument("--no-create", action="store_true", help="Only refresh tallies; don't open new issues.")

    mb = sub.add_parser("moi-backtest",
                        help="Walk-forward Model-of-Interestingness forecasting backtest (manual; LLM cost).")
    mb.add_argument("--cutoffs", default="2021,2022,2023", help="Comma-separated cutoff years Y.")
    mb.add_argument("--arms", default="honest,oracle", help="Comma-separated: honest,oracle.")
    mb.add_argument("--modes", default="steered,baseline", help="Comma-separated: steered,baseline.")
    mb.add_argument("--generations", type=int, default=None, help="Override QD generations.")
    mb.add_argument("--init-pop", type=int, default=None, help="Override seed population size.")
    mb.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    mb.add_argument("--mock", action="store_true", help="Use the offline mock LLM (no API calls).")
    mb.add_argument("--out", default=None, help="Output path (default data/moi_backtest.json).")

    mt = sub.add_parser("moi-train",
                        help="Train the hindsight realization model from the backtest's own "
                             "outcomes (offline; no LLM cost). Writes data/moi_model.json.")
    mt.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    load_dotenv()  # pick up .env: GEMINI_API_KEY / OPENAI_API_KEY / GITHUB_TOKEN / EXODIA_AISCIENTIST_DIR
    settings = load_settings(args.config)

    if args.cmd == "gate":
        changed, latest, state = pipeline.gate(settings, force=_truthy(args.force))
        print(f"changed={changed} latest={latest} stored={state.upstream_sha}")
        _emit_github_output("changed", "true" if changed else "false")
        _emit_github_output("latest_sha", latest or "")
        return 0

    if args.cmd == "run-all":
        if args.max_generations is not None:
            settings.ideation_max_num_generations = args.max_generations
        if args.max_pdfs is not None:
            settings.pdf_max_new_downloads = args.max_pdfs
        if args.pdfs:
            settings.pdf_fetch = True
        if args.no_pdfs:
            settings.pdf_fetch = False
        result = pipeline.run_all(
            settings,
            force=args.force,
            dry_run=args.dry_run,
            readme_path=args.readme_file,
            ais_dir=args.ais_dir,
            no_enrich=args.no_enrich,
        )
        print(result)
        _emit_github_output("changed", "true" if result.get("changed") else "false")
        return 0

    if args.cmd == "render":
        from .render import render_site

        render_site(settings)
        return 0

    if args.cmd == "serve":
        import uvicorn

        from .webapp import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0

    if args.cmd == "sync-issues":
        from .issues import sync

        repo = args.repo or settings.github_repo or os.environ.get("GITHUB_REPOSITORY")
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not repo or not token:
            log.error(
                "sync-issues needs a repo (--repo / site.repo / $GITHUB_REPOSITORY) "
                "and a token in $GITHUB_TOKEN."
            )
            return 2
        created, synced = sync(settings, repo, token, create_missing=not args.no_create)
        print(f"issues: +{created} created, {synced} refreshed (repo={repo})")
        return 0

    if args.cmd == "moi-backtest":
        from .moi import run_backtest

        if args.generations is not None:
            settings.moi_generations = args.generations
        if args.init_pop is not None:
            settings.moi_init_pop = args.init_pop
        cutoffs = [int(y) for y in args.cutoffs.split(",") if y.strip()]
        out = run_backtest(
            settings,
            cutoffs=cutoffs,
            arms=[a.strip() for a in args.arms.split(",") if a.strip()],
            modes=[m.strip() for m in args.modes.split(",") if m.strip()],
            mock=args.mock,
            seed=args.seed,
            out=args.out,
        )
        print(f"moi-backtest -> {out}")
        return 0

    if args.cmd == "moi-train":
        from .moi.learner import train_and_save

        model = train_and_save(settings, seed=args.seed)
        if model is None:
            print("moi-train: not enough backtest data to learn from "
                  "(run `exodia moi-backtest` with >= 2 cutoffs first)")
            return 2
        print(f"moi-train -> {settings.moi_model_path} "
              f"({model.n_rows} rows, {model.n_pos} realized)")
        for wf in model.walk_forward:
            print(f"  cutoff {wf['cutoff']}: AUC {wf['auc']} · P@10 {wf['precision_at_10']} "
                  f"· base rate {wf['base_rate']} (train n={wf['n_train']}, test n={wf['n_test']})")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
