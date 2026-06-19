from exodia.citations import fetch_citation_batch, fetch_citations
from exodia.config import Settings
from exodia.models import Entry


def _e(eid, arxiv_id, cites=None, infl=None):
    return Entry(eid, f"Paper {eid}", [], "", None, 2023, "papers", {},
                 arxiv_id=arxiv_id, citation_count=cites, influential_citation_count=infl)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _settings():
    s = Settings()
    s.citations_request_delay_seconds = 0.0  # no sleeping in tests
    return s


def test_fetch_citation_batch_aligns_and_skips_null(monkeypatch):
    captured = {}

    def fake_post(url, params=None, json=None, headers=None, timeout=None):
        captured["json"] = json
        # Semantic Scholar returns a list aligned with input ids, null for unknown.
        return _FakeResp([{"citationCount": 10, "influentialCitationCount": 2}, None])

    monkeypatch.setattr("exodia.citations.requests.post", fake_post)
    out = fetch_citation_batch(["2305.16291", "9999.99999"])
    assert captured["json"] == {"ids": ["ARXIV:2305.16291", "ARXIV:9999.99999"]}
    assert out == {"2305.16291": (10, 2)}  # the null entry is dropped


def test_fetch_citations_fills_counts_and_strips_version(monkeypatch):
    es = [_e("a", "2305.16291v3"), _e("b", "1901.01753")]

    def fake_post(url, params=None, json=None, headers=None, timeout=None):
        # Version suffix must be stripped before querying.
        assert json == {"ids": ["ARXIV:2305.16291", "ARXIV:1901.01753"]}
        return _FakeResp([
            {"citationCount": 100, "influentialCitationCount": 9},
            {"citationCount": 50, "influentialCitationCount": 4},
        ])

    monkeypatch.setattr("exodia.citations.requests.post", fake_post)
    n = fetch_citations(es, _settings())
    assert n == 2
    assert (es[0].citation_count, es[0].influential_citation_count) == (100, 9)
    assert (es[1].citation_count, es[1].influential_citation_count) == (50, 4)


def test_fetch_citations_is_cache_first(monkeypatch):
    es = [_e("a", "2305.16291", cites=42, infl=3)]

    def boom(*a, **k):
        raise AssertionError("should not hit the network when cached")

    monkeypatch.setattr("exodia.citations.requests.post", boom)
    assert fetch_citations(es, _settings()) == 0
    assert es[0].citation_count == 42  # untouched


def test_fetch_citations_respects_cap(monkeypatch):
    es = [_e(str(i), f"2300.0000{i}") for i in range(5)]
    s = _settings()
    s.citations_max_new_fetches = 2

    seen_ids = []

    def fake_post(url, params=None, json=None, headers=None, timeout=None):
        seen_ids.extend(json["ids"])
        return _FakeResp([{"citationCount": 1, "influentialCitationCount": 0}
                          for _ in json["ids"]])

    monkeypatch.setattr("exodia.citations.requests.post", fake_post)
    fetch_citations(es, s)
    assert len(seen_ids) == 2  # only the first two were requested


def test_fetch_citations_survives_batch_error(monkeypatch):
    es = [_e("a", "2305.16291")]

    def fake_post(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("exodia.citations.requests.post", fake_post)
    assert fetch_citations(es, _settings()) == 0  # error swallowed, no crash
    assert es[0].citation_count is None
