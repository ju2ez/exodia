import json
from pathlib import Path

from exodia.analysis import analyze
from exodia.config import Settings
from exodia.ideation import (
    _allow_model_in_ais,
    _make_ideation_robust,
    ideate,
    parse_ideas,
    run_ideation,
    store_ideas,
    synthesize_topic_md,
)

FIX = Path(__file__).parent / "fixtures"


def test_parse_ideas_flattens_list_fields(tmp_path):
    # Real AI-Scientist output: Experiments = list of dicts, Risks = list of strings.
    p = tmp_path / "ideas.json"
    p.write_text(json.dumps([{
        "Name": "x", "Title": "T", "Short Hypothesis": "H", "Related Work": "R",
        "Abstract": "A",
        "Experiments": [{"name": "exp1", "description": "do a thing"}, {"name": "exp2",
                        "description": "do another"}],
        "Risk Factors and Limitations": ["risk one", "risk two"],
    }]))
    ideas = parse_ideas(p, "run1", "gemini-2.5-pro", None)
    assert len(ideas) == 1
    i = ideas[0]
    assert "exp1 — do a thing" in i.experiments and "exp2 — do another" in i.experiments
    assert "risk one" in i.risk_factors_and_limitations and "risk two" in i.risk_factors_and_limitations


def test_allow_model_and_robustness_patches(tmp_path):
    ais = tmp_path / "ais"
    (ais / "ai_scientist").mkdir(parents=True)
    (ais / "ai_scientist" / "llm.py").write_text("AVAILABLE_LLMS = [\n    'gpt-4o',\n]\n")
    (ais / "ai_scientist" / "perform_ideation_temp_free.py").write_text(
        '        idea = arguments_json.get("idea")\n'
    )
    _allow_model_in_ais(ais, "gemini-2.5-pro")
    _make_ideation_robust(ais)
    llm = (ais / "ai_scientist" / "llm.py").read_text()
    script = (ais / "ai_scientist" / "perform_ideation_temp_free.py").read_text()
    assert '"gemini-2.5-pro"' in llm
    assert 'arguments_json.get("idea") or arguments_json' in script
    # Idempotent: running again doesn't double-patch.
    _allow_model_in_ais(ais, "gemini-2.5-pro")
    assert (ais / "ai_scientist" / "llm.py").read_text().count('"gemini-2.5-pro"') == 1


def test_parse_ideas_maps_seven_keys():
    ideas = parse_ideas(FIX / "aiscientist_ideas.json", "run1", "gpt-4o", "deadbeef")
    assert len(ideas) == 2
    i = ideas[0]
    assert i.name and i.title and i.short_hypothesis and i.related_work
    assert i.abstract and i.experiments and i.risk_factors_and_limitations
    assert i.run_id == "run1" and i.model == "gpt-4o" and i.source_sha == "deadbeef"
    assert i.generated is True
    # idea_id is deterministic for the same run + name
    again = parse_ideas(FIX / "aiscientist_ideas.json", "run1", "gpt-4o", "deadbeef")
    assert [x.idea_id for x in ideas] == [x.idea_id for x in again]


def test_run_ideation_invokes_script_with_expected_args(tmp_path, monkeypatch):
    ais = tmp_path / "ais"
    (ais / "ai_scientist").mkdir(parents=True)
    (ais / "ai_scientist" / "perform_ideation_temp_free.py").write_text("# fake")
    topic = tmp_path / "exodia_topic.md"
    topic.write_text("# topic")

    captured = {}

    def fake_run(cmd, cwd=None, env=None, check=None, timeout=None, **kw):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        # Simulate the script writing <stem>.json next to the topic file.
        topic.with_suffix(".json").write_text(
            json.dumps([{
                "Name": "x", "Title": "T", "Short Hypothesis": "H", "Related Work": "R",
                "Abstract": "A", "Experiments": "E", "Risk Factors and Limitations": "L",
            }])
        )
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr("exodia.ideation.subprocess.run", fake_run)
    out = run_ideation(topic, Settings(), ais)

    assert out == topic.with_suffix(".json")
    assert captured["cwd"] == str(ais)
    assert "--workshop-file" in captured["cmd"]
    assert str(topic.resolve()) in captured["cmd"]
    assert "--model" in captured["cmd"]
    assert "--max-num-generations" in captured["cmd"]
    assert "--num-reflections" in captured["cmd"]


def test_ideate_dry_run_uses_fixture_and_synthesizes_topic(tmp_path, entries_v1):
    settings = Settings()
    report = analyze(entries_v1, settings)
    ideas = ideate(
        entries_v1, report, settings, run_id="r1", source_sha=None,
        dry_run=True, workdir=tmp_path,
    )
    assert len(ideas) == 2
    assert "(dry-run)" in ideas[0].model
    topic = (tmp_path / "exodia_topic.md")
    assert topic.exists()
    assert "Open-Endedness" in topic.read_text(encoding="utf-8")


def test_store_ideas_dedupes(tmp_path, entries_v1):
    settings = Settings()
    settings.data_dir = tmp_path
    report = analyze(entries_v1, settings)
    ideas = ideate(entries_v1, report, settings, "r1", None, dry_run=True, workdir=tmp_path)
    store_ideas(settings, ideas)
    store_ideas(settings, ideas)  # second time should not duplicate
    stored = json.loads(settings.ideas_path.read_text(encoding="utf-8"))
    assert len(stored) == 2


def test_synthesize_topic_includes_themes(tmp_path, entries_v1):
    settings = Settings()
    report = analyze(entries_v1, settings)
    out = synthesize_topic_md(entries_v1, report, settings, tmp_path / "topic.md")
    text = out.read_text(encoding="utf-8")
    assert "## Keywords" in text
    assert "## Representative work" in text
