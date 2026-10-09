"""Offline tests for token safety and the real persistent Chroma boundary."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from src.rag import prepare_documents
from src.retrieval import (
    E5Encoder, Retriever, build_index, config_hash, explicit_article_number, load_config, make_filter, sha256,
    split_document, write_json,
)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [0] * (len(text) + 2)


class FakeEncoder:
    device = "test"
    precision = "float32"

    def encode(self, texts, *, query=False):
        vectors = np.zeros((len(texts), 768), dtype="float32")
        for i, text in enumerate(texts):
            vectors[i, 0] = 1
            vectors[i, 1] = len(text) / 1000
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def prepared(tmp_path, documents, config):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("".join(json.dumps(asdict(d), ensure_ascii=False) + "\n" for d in documents))
    write_json(chunks.with_suffix(".manifest.json"), {
        "config": config, "config_sha256": config_hash(config),
        "corpus_sha256": "test-corpus", "chunks_sha256": sha256(chunks),
        "document_count": len(documents),
    })
    return chunks


def test_provisions_split_without_losing_source_or_article_identity(by_number):
    doc = prepare_documents([by_number[147]])[-1]
    text = "(1) " + "a" * 280 + " (2) " + "b" * 280 + " (3) " + "c" * 280
    doc = replace(doc, source_text=text, embedding_text=text, source_end=len(text))
    chunks = split_document(doc, CharacterTokenizer(), load_config())
    assert len(chunks) == 3
    assert "".join(c.source_text for c in chunks) == text
    for i, chunk in enumerate(chunks):
        assert chunk.metadata == doc.metadata
        assert chunk.chunk_index == i
        assert chunk.source_text == text[chunk.source_start:chunk.source_end]
        assert len("passage: " + chunk.embedding_text) + 2 <= 512


def test_indivisible_provision_is_rejected_instead_of_truncated(by_number):
    doc = prepare_documents([by_number[147]])[-1]
    text = "(1) " + "a" * 600
    doc = replace(doc, source_text=text, embedding_text=text, source_end=len(text))
    with pytest.raises(ValueError, match="indivisible provision"):
        split_document(doc, CharacterTokenizer(), load_config())


def test_embedding_guard_counts_prefix_and_special_tokens():
    encoder = E5Encoder.__new__(E5Encoder)
    encoder.config = load_config()
    encoder.tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="truncation is forbidden"):
        encoder.encode(["a" * 505])


def test_unnumbered_source_provision_anchors_keep_article_1143_intact(by_number):
    class WordTokenizer:
        def encode(self, text, **kwargs):
            return [0] * (len(text.split()) * 2 + 2)

    doc = prepare_documents([by_number[1143]])[-1]
    chunks = split_document(doc, WordTokenizer(), load_config())
    assert len(chunks) > 1
    assert "".join(chunk.source_text for chunk in chunks) == doc.source_text
    starts = load_config()["source_provision_starts"][doc.document_id]
    assert all(any(chunk.source_text.startswith(start) for start in starts) for chunk in chunks)


def test_exact_lookup_handles_digits_and_does_not_hide_multiple_articles():
    assert explicit_article_number("١٠٢٢") == 1022
    assert explicit_article_number("Article 147") == 147
    assert explicit_article_number("المادة ٥٤") == 54
    assert explicit_article_number("Compare Article 147 with Article 148") is None


def test_repeal_missing_language_filters_and_semantic_source_mapping(tmp_path, by_number):
    config = load_config()
    docs = prepare_documents([by_number[54], by_number[147], by_number[1022]])
    chunks = prepared(tmp_path, docs, config)
    index = tmp_path / "index"
    build_index(chunks, index, config, FakeEncoder())
    original_files = {str(path.relative_to(index)): sha256(path) for path in index.rglob("*") if path.is_file()}
    # A second instance reloads persisted state without model/network access.
    retriever = Retriever(index, config, FakeEncoder())
    repeal, = retriever.search("ما حالة المادة ٥٤؟")
    assert repeal["score"] is None and repeal["method"] == "article_lookup"
    assert repeal["passages"][0]["metadata"]["is_repealed"]
    assert repeal["passages"][0]["source_field"] == "repeal_note"
    missing, = retriever.search("Article 1022")
    assert len(missing["passages"]) == 1
    assert missing["passages"][0]["metadata"]["language"] == "en"
    assert retriever.search("المادة ١٠٢٢", filters={"language": "ar"}) == []
    assert retriever.search("Article 5000") == []
    assert retriever.search("Article 147", filters={"is_repealed": True}) == []
    results = retriever.search("contract rights", filters={"article_number": 147, "language": "ar"})
    assert [r["article_number"] for r in results] == [147]
    passage = results[0]["passages"][0]
    assert passage["source_text"] == by_number[147]["ar_text"]
    assert passage["metadata"]["citation"] == by_number[147]["citation"]
    hierarchy = by_number[147]["book"]
    assert all(p["metadata"]["book"] == hierarchy
               for r in retriever.search("contracts", filters={"book": hierarchy}) for p in r["passages"])
    assert {str(path.relative_to(index)): sha256(path) for path in index.rglob("*") if path.is_file()} == original_files


def test_rebuild_removes_stale_ids_and_rejects_configuration_drift(tmp_path, by_number):
    config = load_config()
    docs = prepare_documents([by_number[54], by_number[147]])
    chunks = prepared(tmp_path, docs, config)
    index = tmp_path / "index"
    first = build_index(chunks, index, config, FakeEncoder())
    second = build_index(chunks, index, config, FakeEncoder())
    assert first["indexed_count"] == second["indexed_count"] == 3
    chunks = prepared(tmp_path, docs[:1], config)
    assert build_index(chunks, index, config, FakeEncoder())["indexed_count"] == 1
    assert Retriever(index, config, FakeEncoder()).search("Article 147") == []
    with pytest.raises(ValueError, match="configuration differs"):
        Retriever(index, dict(config, batch_size=8), FakeEncoder())
    with pytest.raises(ValueError, match="do not match"):
        build_index(chunks, index, dict(config, batch_size=8), FakeEncoder())


@pytest.mark.parametrize("filters", [
    {"article_number": "147"}, {"is_repealed": "false"}, {"language": "fr"},
    {"unknown": "value"}, {"topic": None},
])
def test_invalid_filters_are_rejected(filters):
    with pytest.raises(ValueError):
        make_filter(filters)
