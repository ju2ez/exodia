from exodia.config import Settings
from exodia.corpus import corpus_text
from exodia.fulltext import text_path_for
from exodia.models import Entry
from exodia.transcripts import transcript_path


def _settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    return s


def test_corpus_text_falls_back_to_title_abstract(tmp_path):
    s = _settings(tmp_path)
    e = Entry("1", "A Title", [], "", None, 2021, "papers", {}, abstract="the abstract")
    assert corpus_text(e, s) == "A Title the abstract"


def test_corpus_text_includes_transcript_and_pdf_text(tmp_path):
    s = _settings(tmp_path)
    e = Entry("1", "Talk Title", [], "", None, 2022, "videos",
              {"video": "https://youtu.be/gIHAVTj9fjo"}, abstract="abs", arxiv_id="2305.16291")
    tp = transcript_path(s, "gIHAVTj9fjo")
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text("spoken transcript", encoding="utf-8")
    pt = text_path_for(s, "2305.16291")
    pt.parent.mkdir(parents=True, exist_ok=True)
    pt.write_text("paper body", encoding="utf-8")

    text = corpus_text(e, s)
    assert "Talk Title" in text and "abs" in text
    assert "spoken transcript" in text and "paper body" in text


def test_corpus_text_respects_disable_flags(tmp_path):
    s = _settings(tmp_path)
    s.transcripts_fetch = False
    s.fulltext_analyze = False
    e = Entry("1", "T", [], "", None, 2022, "videos",
              {"video": "https://youtu.be/gIHAVTj9fjo"}, arxiv_id="2305.16291")
    transcript_path(s, "gIHAVTj9fjo").parent.mkdir(parents=True, exist_ok=True)
    transcript_path(s, "gIHAVTj9fjo").write_text("nope", encoding="utf-8")
    pt = text_path_for(s, "2305.16291")
    pt.parent.mkdir(parents=True, exist_ok=True)
    pt.write_text("nope", encoding="utf-8")
    assert corpus_text(e, s) == "T"  # caches ignored when disabled
