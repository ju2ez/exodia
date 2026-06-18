from exodia.analysis import analyze, save_themes
from exodia.config import Settings
from exodia.diffing import append_changelog, build_changelog_entry
from exodia.ideation import ideate, store_ideas
from exodia.models import State
from exodia.render import PAGES, render_site
from exodia.store import merge, save_kb
from exodia.upstream import save_state


def _setup(tmp_path, entries):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.site_dir = tmp_path / "site"
    merged, diff = merge([], entries)
    save_kb(s, merged)
    report = analyze(merged, s)
    save_themes(s, report)
    ideas = ideate(merged, report, s, "run1", "abc123def456", dry_run=True, workdir=tmp_path / "work")
    store_ideas(s, ideas)
    ce = build_changelog_entry(
        "run1", diff, from_sha=None, to_sha="abc123def456", compare_url=None,
        total_entries=len(merged), new_ideas=len(ideas), note="Initial build.",
    )
    append_changelog(s.changelog_path, ce)
    save_state(
        State(s.upstream_repo, upstream_sha="abc123def456", last_run_utc="2026-06-18T00:00:00Z", run_count=1),
        s.state_path,
    )
    return s


def test_render_creates_all_pages_and_assets(tmp_path, entries_v1):
    s = _setup(tmp_path, entries_v1)
    render_site(s)
    for name in PAGES:
        assert (s.site_dir / name).exists()
    assert (s.site_dir / ".nojekyll").exists()
    assert (s.site_dir / "assets" / "css" / "style.css").exists()


def test_index_has_mermaid_before_plots_and_last_updated(tmp_path, entries_v1):
    s = _setup(tmp_path, entries_v1)
    render_site(s)
    html = (s.site_dir / "index.html").read_text(encoding="utf-8")
    assert 'class="mermaid"' in html
    assert "Last updated" in html
    assert html.index('class="mermaid"') < html.index("plot-grid")


def test_jenny_zhang_credit_on_every_page(tmp_path, entries_v1):
    s = _setup(tmp_path, entries_v1)
    render_site(s)
    for name in PAGES:
        html = (s.site_dir / name).read_text(encoding="utf-8")
        assert "Jenny Zhang" in html, f"missing credit on {name}"


def test_ideas_page_has_ai_disclaimer(tmp_path, entries_v1):
    s = _setup(tmp_path, entries_v1)
    render_site(s)
    html = (s.site_dir / "ideas.html").read_text(encoding="utf-8")
    assert "AI-generated content" in html
    assert "AI-Scientist-v2" in html
    assert "Open-Ended Curricula from Self-Generated Goals" in html


def test_papers_page_has_arxiv_attribution(tmp_path, entries_v1):
    for e in entries_v1:
        if e.arxiv_id:
            e.abstract = "Test abstract."
            e.abstract_url = f"https://arxiv.org/abs/{e.arxiv_id}"
            e.abstract_source = "arXiv"
    s = _setup(tmp_path, entries_v1)
    render_site(s)
    html = (s.site_dir / "papers.html").read_text(encoding="utf-8")
    assert "Abstract via" in html
    assert "arXiv" in html
