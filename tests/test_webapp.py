import json

from starlette.testclient import TestClient

from exodia.config import Settings
from exodia.webapp import create_app


def _idea(idea_id, title, run_utc, **kw):
    base = dict(
        idea_id=idea_id, name=idea_id, title=title, short_hypothesis="h",
        related_work="r", abstract="ab", experiments="e",
        risk_factors_and_limitations="l", run_id="r1", run_utc=run_utc, model="m",
    )
    base.update(kw)
    return base


def _client(tmp_path) -> TestClient:
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.site_dir = tmp_path / "site"
    s.data_dir.mkdir(parents=True)
    ideas = [
        _idea("i1", "Alpha idea", "2026-01-01T00:00:00Z", realized=True,
              realized_paper_title="Paper X", realized_paper_url="http://x", realized_score=0.5),
        _idea("i2", "Beta idea", "2026-01-02T00:00:00Z"),
    ]
    (s.data_dir / "ideas.json").write_text(json.dumps(ideas))
    return TestClient(create_app(s))


def test_pages_render(tmp_path):
    c = _client(tmp_path)
    assert c.get("/").status_code == 200
    r = c.get("/ideas.html")
    assert r.status_code == 200
    assert "Alpha idea" in r.text
    assert "vote-btn" in r.text          # voting widget present (dynamic mode)
    assert "Best ideas" in r.text        # best-idea list present
    assert "★" in r.text                 # realized-paper star
    assert c.get("/concepts.html").status_code == 200
    assert c.get("/nope.html").status_code == 404


def test_vote_best_and_me(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/ideas/i2/vote", json={"voter": "v1", "value": 1})
    assert r.status_code == 200 and r.json()["score"] == 1
    assert c.post("/api/ideas/unknown/vote", json={"voter": "v1", "value": 1}).status_code == 404
    assert c.post("/api/ideas/i1/vote", json={"voter": "v", "value": 5}).status_code == 422

    best = c.get("/api/best").json()
    assert best[0]["idea_id"] == "i2"    # highest score floats to the top
    assert c.get("/api/me", params={"voter": "v1"}).json() == {"i2": 1}

    # retract returns to zero
    assert c.post("/api/ideas/i2/vote", json={"voter": "v1", "value": 0}).json()["score"] == 0


def test_feedback_api(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/ideas/i1/feedback", json={"text": "nice", "author": "alice"})
    assert r.status_code == 201 and r.json()["text"] == "nice"
    assert len(c.get("/api/ideas/i1/feedback").json()) == 1
    assert c.post("/api/ideas/i1/feedback", json={"text": "   "}).status_code == 422
    assert c.post("/api/ideas/unknown/feedback", json={"text": "hi"}).status_code == 404
