from exodia import issues


class _Resp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._p


def _idea(idea_id="a", **kw):
    base = {"idea_id": idea_id, "name": idea_id, "title": f"Title {idea_id}",
            "short_hypothesis": "h", "abstract": "ab"}
    base.update(kw)
    return base


def test_tally_from_issue_parses_reactions_and_comments():
    d = {"reactions": {"+1": 5, "-1": 2}, "comments": 3, "html_url": "http://x/1"}
    t = issues.tally_from_issue(d)
    assert t == {"votes_up": 5, "votes_down": 2, "score": 3, "feedback_count": 3, "issue_url": "http://x/1"}


def test_sync_creates_missing_then_tallies(monkeypatch):
    posts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        posts.append(json)
        return _Resp({"number": 11, "html_url": "http://x/11"})

    def fake_get(url, headers=None, timeout=None):
        return _Resp({"html_url": "http://x/11", "reactions": {"+1": 4, "-1": 1}, "comments": 2})

    monkeypatch.setattr("exodia.issues.requests.post", fake_post)
    monkeypatch.setattr("exodia.issues.requests.get", fake_get)

    ideas = [_idea("a")]
    created, synced = issues.sync_ideas(ideas, "owner/repo", "tok")
    assert (created, synced) == (1, 1)
    assert ideas[0]["issue_number"] == 11
    assert ideas[0]["issue_url"] == "http://x/11"
    assert ideas[0]["votes_up"] == 4 and ideas[0]["score"] == 3
    # the created issue body carries the idea-id marker
    assert "exodia-idea: a" in posts[0]["body"]


def test_sync_does_not_recreate_existing_issue(monkeypatch):
    def boom_post(*a, **k):
        raise AssertionError("should not create an issue when issue_number is set")

    def fake_get(url, headers=None, timeout=None):
        return _Resp({"html_url": "http://x/7", "reactions": {"+1": 1, "-1": 0}, "comments": 0})

    monkeypatch.setattr("exodia.issues.requests.post", boom_post)
    monkeypatch.setattr("exodia.issues.requests.get", fake_get)

    ideas = [_idea("a", issue_number=7)]
    created, synced = issues.sync_ideas(ideas, "owner/repo", "tok")
    assert (created, synced) == (0, 1)
    assert ideas[0]["score"] == 1


def test_no_create_flag_skips_issueless_ideas(monkeypatch):
    monkeypatch.setattr("exodia.issues.requests.post",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no post")))
    monkeypatch.setattr("exodia.issues.requests.get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no get")))
    ideas = [_idea("a")]
    created, synced = issues.sync_ideas(ideas, "owner/repo", "tok", create_missing=False)
    assert (created, synced) == (0, 0)
