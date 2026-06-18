from exodia.enrich import parse_arxiv_atom
from exodia.models import Entry
from exodia.venues import PREPRINT_LABEL, resolve_venue, resolve_venues

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2305.16291v3</id>
    <published>2023-05-25T00:00:00Z</published>
    <title>Voyager</title>
    <summary>An open-ended embodied agent.</summary>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:journal_ref>Transactions on Machine Learning Research (2024)</arxiv:journal_ref>
    <arxiv:comment>Accepted at TMLR</arxiv:comment>
    <arxiv:doi>10.1000/xyz123</arxiv:doi>
  </entry>
</feed>"""


def _entry(**kw) -> Entry:
    base = dict(
        entry_id="e", title="t", authors=[], authors_raw="", venue=None, year=2023,
        category="papers", links={},
    )
    base.update(kw)
    return Entry(**base)


def test_parse_atom_extracts_journal_ref_comment_doi():
    meta = parse_arxiv_atom(ATOM)
    m = meta["2305.16291"]
    assert m.journal_ref == "Transactions on Machine Learning Research (2024)"
    assert m.comment == "Accepted at TMLR"
    assert m.doi == "10.1000/xyz123"


def test_journal_ref_wins_and_is_recognized():
    e = _entry(arxiv_id="2305.16291", venue="arXiv",
               journal_ref="Proc. of NeurIPS 2023")
    assert resolve_venue(e) == "NeurIPS"


def test_comment_acceptance_detected_when_no_journal_ref():
    e = _entry(arxiv_id="1", venue="arXiv", comment="Accepted at ICLR 2024 (spotlight)")
    assert resolve_venue(e) == "ICLR"


def test_arxiv_only_is_labeled_preprint_not_a_venue():
    e = _entry(arxiv_id="1", venue="arXiv")
    assert resolve_venue(e) == PREPRINT_LABEL


def test_real_curated_venue_passes_through():
    e = _entry(venue="Nature Machine Intelligence")
    assert resolve_venue(e) == "Nature Machine Intelligence"


def test_non_arxiv_without_venue_is_unpublished():
    e = _entry(category="blogs")
    assert resolve_venue(e) == "Unpublished"


def test_resolve_venues_populates_display_field():
    es = [_entry(arxiv_id="1", venue="arXiv"), _entry(venue="ICML")]
    resolve_venues(es)
    assert es[0].venue_display == PREPRINT_LABEL
    assert es[1].venue_display == "ICML"
