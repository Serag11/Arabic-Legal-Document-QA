"""Incremental re-indexing with a clearly fictional second document (tests/fixtures/test_statute.jsonl)."""

import json
from pathlib import Path
import shutil

import pytest

from src.rag import prepare_documents
from src.reindex import reindex
from src.retrieval import Retriever, build_index, config_hash, load_config, sha256
from tests.test_retrieval import CharacterTokenizer, FakeEncoder, prepared

FIXTURE = Path(__file__).parent / "fixtures/test_statute.jsonl"
CIVIL = "egyptian-civil-code"


class CountingEncoder(FakeEncoder):
    def __init__(self):
        self.encoded = 0

    def encode(self, texts, *, query=False):
        self.encoded += len(texts)
        return super().encode(texts, query=query)


class ForbiddenEncoder:
    def encode(self, *args, **kwargs):
        raise AssertionError("an unchanged corpus must not be embedded")


@pytest.fixture
def base_index(tmp_path, by_number):
    config = load_config()
    chunks = prepared(tmp_path, prepare_documents([by_number[1], by_number[147], by_number[54]]), config)
    index = tmp_path / "index"
    build_index(chunks, index, config, FakeEncoder())
    return index, config


def run(index, config, corpora=None, **kwargs):
    kwargs.setdefault("encoder", CountingEncoder())
    kwargs.setdefault("tokenizer", CharacterTokenizer())
    return reindex(index, {"test-fixture": FIXTURE} if corpora is None else corpora, config, **kwargs)


def manifest(index):
    return json.loads((index / "manifest.json").read_text())


def test_new_corpus_is_added_beside_existing_documents(base_index):
    index, config = base_index
    before = manifest(index)
    encoder = CountingEncoder()
    report = run(index, config, encoder=encoder)
    # 3 bilingual articles = 6 chunks, plus one English repeal notice.
    assert report["corpora"]["test-fixture"]["added"] == 7 and report["embedded_chunks"] == encoder.encoded == 7
    after = manifest(index)
    assert after["indexed_count"] == before["indexed_count"] + 7 == report["indexed_count"]
    assert after["article_count"] == 3 + 4  # articles 1, 54, 147 plus the four fixture articles
    assert after["corpus_document_counts"] == {CIVIL: before["indexed_count"], "test-fixture": 7}
    assert after["corpora"]["test-fixture"]["corpus_sha256"] == sha256(FIXTURE)
    assert after["corpus_sha256"] == before["corpus_sha256"]  # the original build's provenance is untouched

    retriever = Retriever(index, config, FakeEncoder())
    fixture = retriever.search("library", filters={"corpus_id": "test-fixture", "language": "en"}, k=10)
    assert {r["citation"] for r in fixture} == {f"Test Fixture Statute, Article {n}" for n in (1, 2, 3, 4)}
    assert {r["corpus_id"] for r in fixture} == {"test-fixture"}
    original = retriever.search("contract", filters={"corpus_id": CIVIL}, k=10)
    assert {r["corpus_id"] for r in original} == {CIVIL}


def test_article_numbers_that_exist_in_both_corpora_stay_separate_results(base_index):
    index, config = base_index
    run(index, config)
    retriever = Retriever(index, config, FakeEncoder())
    lookup = retriever.search("Article 1")
    assert sorted((r["corpus_id"], r["citation"]) for r in lookup) == [
        (CIVIL, "Egyptian Civil Code, Article 1"), ("test-fixture", "Test Fixture Statute, Article 1")]
    repealed = retriever.search("Article 4", filters={"corpus_id": "test-fixture"})
    assert repealed[0]["passages"][0]["source_field"] == "repeal_note" and repealed[0]["passages"][0]["metadata"]["is_repealed"]


def test_rerun_with_unchanged_input_embeds_nothing_and_keeps_the_manifest(base_index):
    index, config = base_index
    run(index, config)
    first = (index / "manifest.json").read_bytes()
    report = run(index, config, encoder=ForbiddenEncoder())
    assert report["embedded_chunks"] == 0 and report["corpora"]["test-fixture"]["unchanged"] == 7
    assert (index / "manifest.json").read_bytes() == first


def test_only_changed_and_new_chunks_are_embedded_and_deleted_articles_disappear(base_index, tmp_path):
    index, config = base_index
    run(index, config)
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    rows[0]["text_en"] = "Library books may be borrowed for twenty-one days."
    rows = rows[:2] + rows[3:]  # drop Article 3 (and renumber the repealed record to keep numbers consecutive)
    rows[2] = dict(rows[2], article_number=3, citation="Test Fixture Statute, Article 3")
    edited = tmp_path / "edited.jsonl"
    edited.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    encoder = CountingEncoder()
    report = run(index, config, {"test-fixture": edited}, encoder=encoder)
    plan = report["corpora"]["test-fixture"]
    # Article 1 (English) changed; Article 3 is now the repeal notice (new content under an id that held
    # English text), and the old Article 3 Arabic chunk is gone.
    assert plan["changed"] >= 2 and plan["removed"] >= 1 and plan["unchanged"] >= 3
    assert encoder.encoded == plan["added"] + plan["changed"] < 7
    retriever = Retriever(index, config, FakeEncoder())
    article1 = retriever.search("Article 1", filters={"corpus_id": "test-fixture", "language": "en"})
    assert "twenty-one days" in article1[0]["passages"][0]["source_text"]
    assert manifest(index)["indexed_count"] == report["indexed_count"]
    assert manifest(index)["reindex_history"][-1]["corpora"]["test-fixture"]["removed"] >= 1


def test_dry_run_reports_the_plan_without_touching_the_index(base_index):
    index, config = base_index
    files = {str(p.relative_to(index)): sha256(p) for p in index.rglob("*") if p.is_file()}
    report = run(index, config, dry_run=True, encoder=ForbiddenEncoder())
    assert report["dry_run"] and report["corpora"]["test-fixture"]["added"] == 7
    assert {str(p.relative_to(index)): sha256(p) for p in index.rglob("*") if p.is_file()} == files


def test_removing_a_corpus_leaves_the_others_intact(base_index):
    index, config = base_index
    original = manifest(index)["indexed_count"]
    run(index, config)
    report = reindex(index, {}, config, remove=["test-fixture"])
    assert report["removed_corpora"] == {"test-fixture": 7} and report["indexed_count"] == original
    after = manifest(index)
    assert "test-fixture" not in after["corpora"] and after["corpus_document_counts"] == {CIVIL: original}
    assert Retriever(index, config, FakeEncoder()).search("library", filters={"corpus_id": "test-fixture"}) == []


def test_refuses_unsafe_requests(base_index, tmp_path):
    index, config = base_index
    with pytest.raises(ValueError, match="configuration differs"):
        run(index, dict(config, batch_size=8))
    with pytest.raises(ValueError, match="Invalid corpus ID"):
        run(index, config, {"Bad ID!": FIXTURE})
    with pytest.raises(ValueError, match="both updated and removed"):
        run(index, config, remove=["test-fixture"])
    with pytest.raises(ValueError, match="manifest.json missing"):
        run(tmp_path / "nothing", config)
    broken = tmp_path / "broken.jsonl"
    broken.write_text(FIXTURE.read_text(encoding="utf-8").replace("Test Fixture Statute, Article 2", "Other, Article 2"),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="citation does not match"):
        run(index, config, {"test-fixture": broken})
    assert config_hash(config) == manifest(index)["config_sha256"]


def test_service_answers_only_from_its_configured_corpus(base_index):
    from src.generation import AnsweringService, load_generation_config
    from tests.test_generation import FakeGenerator, supported

    index, config = base_index
    run(index, config)
    seen = []

    class SpyRetriever(Retriever):
        def search(self, question, k=5, filters=None, **kwargs):
            seen.append(dict(filters or {}))
            return super().search(question, k, filters, **kwargs)

    retriever = SpyRetriever(index, config, FakeEncoder())
    service = AnsweringService(retriever, FakeGenerator(supported()), load_generation_config())
    service.ask("Article 147", filters={"language": "ar"})
    assert seen[0] == {"language": "ar", "corpus_id": CIVIL}
    shutil.rmtree(index, ignore_errors=True)
