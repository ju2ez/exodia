from exodia.config import Settings
from exodia.models import Entry
from exodia.transcripts import (
    cached_transcript,
    fetch_transcripts,
    transcript_path,
    youtube_id,
)


def _settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.transcripts_request_delay_seconds = 0.0
    return s


def _video(eid, url):
    return Entry(eid, f"Talk {eid}", [], "", None, 2022, "videos", {"video": url})


def test_youtube_id_parses_url_forms():
    assert youtube_id("https://youtu.be/gIHAVTj9fjo") == "gIHAVTj9fjo"
    assert youtube_id("https://www.youtube.com/watch?v=5jL5wRGrCvk") == "5jL5wRGrCvk"
    assert youtube_id("https://youtube.com/embed/abc123XYZ_-") == "abc123XYZ_-"
    assert youtube_id("https://iclr.cc/virtual/2025/invited-talk/36780") is None
    assert youtube_id("") is None


def test_fetch_transcripts_writes_cache_and_is_cache_first(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    calls = []
    monkeypatch.setattr(
        "exodia.transcripts._fetch_segments",
        lambda vid: calls.append(vid) or f"transcript words for {vid}",
    )
    es = [
        _video("1", "https://youtu.be/gIHAVTj9fjo"),
        _video("2", "https://iclr.cc/virtual/x"),   # not YouTube -> skipped
        Entry("3", "a paper", [], "", None, 2022, "papers", {}),  # not a video -> skipped
    ]
    n = fetch_transcripts(es, s)
    assert n == 1 and calls == ["gIHAVTj9fjo"]
    assert transcript_path(s, "gIHAVTj9fjo").read_text().startswith("transcript words")

    # Second run: cached on disk -> no fetch.
    calls.clear()
    assert fetch_transcripts(es, s) == 0 and calls == []


def test_fetch_transcripts_respects_cap(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.transcripts_max_new_fetches = 1
    seen = []
    monkeypatch.setattr("exodia.transcripts._fetch_segments", lambda vid: seen.append(vid) or "x")
    es = [_video("1", "https://youtu.be/aaaaaaaaaaa"), _video("2", "https://youtu.be/bbbbbbbbbbb")]
    fetch_transcripts(es, s)
    assert len(seen) == 1  # capped


def test_fetch_transcripts_survives_errors(tmp_path, monkeypatch):
    s = _settings(tmp_path)

    def boom(vid):
        raise RuntimeError("no captions")

    monkeypatch.setattr("exodia.transcripts._fetch_segments", boom)
    assert fetch_transcripts([_video("1", "https://youtu.be/gIHAVTj9fjo")], s) == 0


def test_cached_transcript_reader(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    e = _video("1", "https://youtu.be/gIHAVTj9fjo")
    assert cached_transcript(e, s) == ""  # nothing cached yet
    p = transcript_path(s, "gIHAVTj9fjo")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("hello world", encoding="utf-8")
    assert cached_transcript(e, s) == "hello world"
