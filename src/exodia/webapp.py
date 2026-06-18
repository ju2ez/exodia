"""FastAPI app: serves the Exodia site dynamically and collects votes + feedback.

Read-only content (ideas, knowledge base, themes, interactive plots) is built
exactly as the static renderer builds it, via :func:`render.site_context`. The
Ideas page is augmented with live vote tallies, a ranked "best ideas" list, and
per-idea feedback from the SQLite store (:mod:`db`). Run it with::

    python -m exodia serve            # http://127.0.0.1:8000

Static pages are rendered once and cached; the Ideas page is rendered per request
so vote counts stay live. ``POST /api/rebuild`` reloads content after a pipeline
run without restarting the server.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, paths
from .config import Settings, load_settings
from .logging_setup import get_logger
from .render import PAGES, site_context

log = get_logger(__name__)

_STATIC_PAGES = [p for p in PAGES if p != "ideas.html"]


class VoteIn(BaseModel):
    voter: str = Field(min_length=1, max_length=64)
    value: int = Field(ge=-1, le=1)


class FeedbackIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    author: str | None = Field(default=None, max_length=80)


class _State:
    """Cached, rebuildable view of the rendered site + the idea index."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = settings.data_dir / "community.db"
        self.build()

    def build(self) -> None:
        env, global_ctx, page_ctx = site_context(self.settings)
        global_ctx = {**global_ctx, "dynamic": True}
        self.env = env
        self.global_ctx = global_ctx
        self.page_ctx = page_ctx
        self.ideas = page_ctx["ideas.html"]["ideas"]
        self.idea_ids = {d["idea_id"] for d in self.ideas}
        # Static pages never change between pipeline runs — render them once.
        self.static_html = {
            name: env.get_template(name).render(**global_ctx, **page_ctx[name])
            for name in _STATIC_PAGES
        }
        log.info("Web app built: %d ideas, %d static pages", len(self.ideas), len(self.static_html))

    def best_ideas(self, tallies: dict, limit: int = 10) -> list[dict]:
        scored = [{**d, **tallies.get(d["idea_id"], {"up": 0, "down": 0, "score": 0})} for d in self.ideas]
        scored.sort(
            key=lambda d: (d["score"], d["up"], bool(d.get("realized")), d.get("run_utc", "")),
            reverse=True,
        )
        return scored[:limit]

    def render_ideas(self) -> str:
        tallies = db.tallies(self.db_path, list(self.idea_ids))
        feedback = db.feedback_by_idea(self.db_path)
        ideas_live = [{**d, **tallies.get(d["idea_id"], {"up": 0, "down": 0, "score": 0})} for d in self.ideas]
        ctx = {
            **self.global_ctx,
            "ideas": ideas_live,
            "n_ideas": len(ideas_live),
            "best_ideas": self.best_ideas(tallies),
            "feedback": feedback,
        }
        return self.env.get_template("ideas.html").render(**ctx)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    db.init_db(settings.data_dir / "community.db")
    state = _State(settings)

    app = FastAPI(title="Exodia", docs_url="/api/docs")
    app.mount("/assets", StaticFiles(directory=str(paths.ASSETS_DIR)), name="assets")

    def _require_idea(idea_id: str) -> None:
        if idea_id not in state.idea_ids:
            raise HTTPException(status_code=404, detail="unknown idea_id")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(state.static_html["index.html"])

    @app.get("/ideas.html", response_class=HTMLResponse)
    @app.get("/ideas", response_class=HTMLResponse)
    def ideas_page() -> HTMLResponse:
        return HTMLResponse(state.render_ideas())

    @app.get("/{page}.html", response_class=HTMLResponse)
    def static_page(page: str) -> HTMLResponse:
        name = f"{page}.html"
        if name not in state.static_html:
            raise HTTPException(status_code=404, detail="no such page")
        return HTMLResponse(state.static_html[name])

    @app.post("/api/ideas/{idea_id}/vote")
    def vote(idea_id: str, body: VoteIn) -> JSONResponse:
        _require_idea(idea_id)
        tally = db.cast_vote(state.db_path, idea_id, body.voter, body.value)
        return JSONResponse(tally)

    @app.get("/api/me")
    def my_votes(voter: str) -> JSONResponse:
        return JSONResponse(db.voter_votes(state.db_path, voter))

    @app.get("/api/best")
    def best(limit: int = 10) -> JSONResponse:
        tallies = db.tallies(state.db_path, list(state.idea_ids))
        items = [
            {"idea_id": d["idea_id"], "title": d["title"], "score": d["score"],
             "up": d["up"], "down": d["down"], "realized": bool(d.get("realized"))}
            for d in state.best_ideas(tallies, limit)
        ]
        return JSONResponse(items)

    @app.get("/api/ideas/{idea_id}/feedback")
    def get_feedback(idea_id: str) -> JSONResponse:
        _require_idea(idea_id)
        return JSONResponse(db.list_feedback(state.db_path, idea_id))

    @app.post("/api/ideas/{idea_id}/feedback")
    def post_feedback(idea_id: str, body: FeedbackIn) -> JSONResponse:
        _require_idea(idea_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="empty feedback")
        author = (body.author or "").strip() or None
        return JSONResponse(db.add_feedback(state.db_path, idea_id, text, author), status_code=201)

    @app.post("/api/rebuild")
    def rebuild() -> JSONResponse:
        state.build()
        return JSONResponse({"ok": True, "ideas": len(state.ideas)})

    return app
