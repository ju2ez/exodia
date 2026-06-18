def _find(entries, category, substr):
    for e in entries:
        if e.category == category and substr.lower() in e.title.lower():
            return e
    return None


def test_counts(entries_v1):
    cats = [e.category for e in entries_v1]
    assert cats.count("papers") == 3
    assert cats.count("safety") == 1
    assert cats.count("blogs") == 1
    assert cats.count("videos") == 1
    assert len(entries_v1) == 6


def test_poet_paper_fields(entries_v1):
    e = _find(entries_v1, "papers", "Paired Open-Ended Trailblazer")
    assert e is not None
    assert e.venue == "GECCO"
    assert e.year == 2019
    assert {"paper", "code", "website"}.issubset(e.links.keys())
    assert e.arxiv_id == "1901.01753"
    assert "Rui Wang" in e.authors
    assert "Kenneth O. Stanley" in e.authors


def test_blog_fields(entries_v1):
    e = _find(entries_v1, "blogs", "Interactive poetry breeding")
    assert e is not None
    assert e.year == 2024
    assert e.venue is None
    assert e.authors == ["Joel Lehman"]
    assert "blog" in e.links


def test_video_fields(entries_v1):
    e = _find(entries_v1, "videos", "Paired Open-Ended Trailblazer")
    assert e is not None
    assert e.year == 2019
    assert "video" in e.links
    assert e.authors == ["Jeff Clune"]


def test_same_title_different_category_distinct_ids(entries_v1):
    paper = _find(entries_v1, "papers", "Paired Open-Ended Trailblazer")
    video = _find(entries_v1, "videos", "Paired Open-Ended Trailblazer")
    assert paper is not None and video is not None
    assert paper.entry_id != video.entry_id


def test_toc_links_not_parsed(entries_v1):
    titles = {e.title.lower() for e in entries_v1}
    assert "papers" not in titles
    assert "videos" not in titles
    assert "open-ended ai safety" not in titles
