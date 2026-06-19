from exodia.analysis import analyze
from exodia.config import Settings
from exodia.models import Entry
from exodia.plotting import abstract_coverage, make_plots


def test_analyze_produces_keyphrases(entries_v1):
    report = analyze(entries_v1, Settings())
    assert report.n_docs == 6
    assert report.method in ("tfidf_kmeans", "keyword_counts")
    assert report.top_keyphrases
    phrases = {k["phrase"] for k in report.top_keyphrases}
    assert "open" not in phrases  # domain stopwords filtered
    assert "ended" not in phrases


def test_analyze_deterministic(entries_v1):
    r1 = analyze(entries_v1, Settings())
    r2 = analyze(entries_v1, Settings())
    assert [k["phrase"] for k in r1.top_keyphrases] == [k["phrase"] for k in r2.top_keyphrases]
    assert [c["label"] for c in r1.clusters] == [c["label"] for c in r2.clusters]


def test_analyze_small_corpus_no_clusters():
    es = [
        Entry("1", "open-ended learning agents", [], "", None, 2020, "papers", {}),
        Entry("2", "quality diversity search", [], "", None, 2021, "papers", {}),
    ]
    report = analyze(es, Settings())
    assert report.n_docs == 2
    assert report.clusters == []


def test_abstract_coverage_excludes_blogs_and_videos():
    es = [
        Entry("1", "paper a", [], "", None, 2021, "papers", {}, abstract="x"),
        Entry("2", "paper b", [], "", None, 2021, "papers", {}),       # missing abstract
        Entry("3", "a blog", [], "", None, 2021, "blogs", {}),         # excluded
        Entry("4", "a talk", [], "", None, 2021, "videos", {}),        # excluded
    ]
    card = abstract_coverage(es, Settings())
    assert card and card["id"] == "abstract_coverage"
    # Only the two papers count: 1 with, 1 without (blogs/videos are not in the denominator).
    assert list(card["fig"].data[0].values) == [1, 1]


def test_abstract_coverage_none_when_only_blogs_videos():
    es = [Entry("1", "blog", [], "", None, 2021, "blogs", {}),
          Entry("2", "vid", [], "", None, 2021, "videos", {})]
    assert abstract_coverage(es, Settings()) is None


def test_make_plots_returns_interactive_html(entries_v1):
    s = Settings()
    report = analyze(entries_v1, s)
    cards = make_plots(entries_v1, report, s)
    assert len(cards) >= 4
    for c in cards:
        assert {"id", "title", "caption", "html"} <= set(c)
        assert "plotly-graph-div" in c["html"]  # interactive, embedded inline
    # Plotly.js is pulled from the CDN exactly once (only the first chart on the page).
    assert sum("cdn.plot.ly" in c["html"] for c in cards) == 1
