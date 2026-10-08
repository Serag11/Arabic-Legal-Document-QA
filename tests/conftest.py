"""Load the generated corpus once for validation against the source contract."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def articles():
    path = Path(__file__).resolve().parents[1] / "data/egyptian_civil_code_articles.jsonl"
    assert path.is_file(), "Corpus missing: run the DVC extraction stage first."
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="session")
def by_number(articles):
    return {article["article_number"]: article for article in articles}
