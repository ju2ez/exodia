from pathlib import Path

import pytest

from exodia.parser import parse_readme

FIXTURES = Path(__file__).parent / "fixtures"
UPSTREAM_REPO = "jennyzzt/awesome-open-ended"


@pytest.fixture
def readme_v1() -> str:
    return (FIXTURES / "upstream_readme_v1.md").read_text(encoding="utf-8")


@pytest.fixture
def readme_v2() -> str:
    return (FIXTURES / "upstream_readme_v2.md").read_text(encoding="utf-8")


@pytest.fixture
def entries_v1(readme_v1):
    entries, _ = parse_readme(readme_v1, UPSTREAM_REPO)
    return entries


@pytest.fixture
def entries_v2(readme_v2):
    entries, _ = parse_readme(readme_v2, UPSTREAM_REPO)
    return entries
