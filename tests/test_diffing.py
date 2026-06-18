import copy

from exodia.diffing import (
    append_changelog,
    build_changelog_entry,
    has_changes,
    load_changelog,
)
from exodia.store import merge


def test_counts_and_changed_fields(entries_v1, entries_v2):
    merged, diff = merge(entries_v1, entries_v2)
    assert has_changes(diff)
    ce = build_changelog_entry(
        "run1", diff, from_sha="aaa", to_sha="bbb",
        compare_url="http://x", total_entries=len(merged),
    )
    assert ce.counts == {"added": 1, "removed": 1, "changed": 1, "total_entries": len(merged)}
    fields = ce.changed[0]["fields"]
    assert "venue" in fields and "year" in fields


def test_no_changes_on_identical_input(entries_v1):
    _, diff = merge(entries_v1, copy.deepcopy(entries_v1))
    assert not has_changes(diff)


def test_append_changelog_roundtrip(tmp_path, entries_v1, entries_v2):
    merged, diff = merge(entries_v1, entries_v2)
    p = tmp_path / "changelog.json"
    append_changelog(
        p, build_changelog_entry("run1", diff, from_sha=None, to_sha="bbb",
                                 compare_url=None, total_entries=len(merged))
    )
    assert len(load_changelog(p)) == 1
    append_changelog(
        p, build_changelog_entry("run2", diff, from_sha="bbb", to_sha="ccc",
                                 compare_url=None, total_entries=len(merged))
    )
    log = load_changelog(p)
    assert len(log) == 2
    assert [e["run_id"] for e in log] == ["run1", "run2"]
