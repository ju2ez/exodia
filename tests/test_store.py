from exodia.config import Settings
from exodia.store import load_entries, merge, save_kb


def test_merge_all_added_on_first_run(entries_v1):
    merged, diff = merge([], entries_v1)
    assert len(diff["added"]) == 6
    assert len(diff["removed"]) == 0
    assert len(diff["changed"]) == 0
    assert all(e.first_seen_utc for e in merged)
    assert all(e.last_seen_utc for e in merged)


def test_merge_v1_to_v2(entries_v1, entries_v2):
    merged, diff = merge(entries_v1, entries_v2)
    assert len(diff["added"]) == 1
    assert len(diff["removed"]) == 1
    assert len(diff["changed"]) == 1
    assert any("OMNI" in e.title for e in diff["added"])
    assert any("poetry breeding" in e.title.lower() for e in diff["removed"])
    old, new = diff["changed"][0]
    assert "Voyager" in new.title
    assert old.venue != new.venue
    assert old.year != new.year


def test_carry_forward_abstract(entries_v1, entries_v2):
    for e in entries_v1:
        if "Voyager" in e.title:
            e.abstract = "An abstract."
            e.abstract_source = "arXiv"
            e.first_seen_utc = "2020-01-01T00:00:00Z"
    merged, _ = merge(entries_v1, entries_v2)
    voyager = next(e for e in merged if "Voyager" in e.title)
    assert voyager.abstract == "An abstract."
    assert voyager.abstract_source == "arXiv"
    assert voyager.first_seen_utc == "2020-01-01T00:00:00Z"


def test_save_load_roundtrip(tmp_path, entries_v1):
    s = Settings()
    s.data_dir = tmp_path
    save_kb(s, entries_v1)
    loaded = load_entries(s.kb_path)
    assert len(loaded) == 6
    assert {e.entry_id for e in entries_v1} == {e.entry_id for e in loaded}
