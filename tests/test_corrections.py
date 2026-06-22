from exodia.corrections import corrected_arxiv_id
from exodia.parser import parse_entry


def test_corrected_arxiv_id_fixes_known_bad_id():
    # AI Scientist-v2 is mislinked upstream to v1's id (2408.06292); real v2 is 2504.08066.
    assert corrected_arxiv_id(
        "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search",
        "2408.06292",
    ) == "2504.08066"


def test_corrected_arxiv_id_passes_through_unknown_titles():
    assert corrected_arxiv_id("Some Other Paper", "1234.56789") == "1234.56789"
    assert corrected_arxiv_id("Some Other Paper", None) is None


def test_parser_applies_correction_to_id_and_link():
    block = [
        "* **The AI Scientist-v2: Workshop-Level Automated Scientific Discovery "
        "via Agentic Tree Search** <br>",
        "Authors. <br>",
        "arXiv, 2025. [[Paper]](https://arxiv.org/abs/2408.06292) "
        "[[Code]](https://github.com/SakanaAI/AI-Scientist-v2)",
    ]
    e = parse_entry(block, "papers", "jennyzzt/awesome-open-ended")
    assert e is not None
    assert e.arxiv_id == "2504.08066"  # corrected away from v1's id
    assert e.links["paper"] == "https://arxiv.org/abs/2504.08066"  # link rewritten too
    assert e.links["code"] == "https://github.com/SakanaAI/AI-Scientist-v2"  # untouched
