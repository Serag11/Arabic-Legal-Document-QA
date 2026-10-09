"""Verify provenance and explicit handling of source/repeal exceptions."""

from copy import deepcopy
import json

import pytest

from src.rag import load_articles, prepare_corpus, prepare_documents


def test_every_document_maps_to_exact_source_passage(articles, by_number):
    documents = prepare_documents(articles)
    assert len(documents) == 2241
    assert len({doc.document_id for doc in documents}) == len(documents)
    assert {doc.metadata.article_number for doc in documents} == set(by_number)
    for doc in documents:
        article = by_number[doc.metadata.article_number]
        assert doc.source_text == article[doc.source_field][doc.source_start:doc.source_end]
        assert doc.metadata.citation == article["citation"]
        for field in ("book", "chapter", "section", "topic", "source_book", "source_page"):
            assert getattr(doc.metadata, field) == article[field]
        assert doc.metadata.qa_flags == tuple(article["qa_flags"])
        expected = article["ar_text_normalized"] if doc.metadata.language == "ar" else doc.source_text
        assert doc.embedding_text == expected


def test_source_exception_is_english_only(by_number):
    documents = prepare_documents([by_number[1022]])
    assert len(documents) == 1
    assert documents[0].metadata.language == "en"
    assert "missing_ar_text_in_source" in documents[0].metadata.qa_flags


def test_repealed_article_indexes_notice_not_original_provision(by_number):
    document, = prepare_documents([by_number[54]])
    assert document.metadata.is_repealed
    assert document.metadata.content_kind == "repeal_notice"
    assert document.source_field == "repeal_note"
    assert document.source_text == by_number[54]["repeal_note"]
    assert document.metadata.article_number == 54


def test_unexplained_missing_arabic_is_rejected(by_number):
    article = deepcopy(by_number[147])
    article["ar_text"] = ""
    with pytest.raises(ValueError, match="unexplained missing Arabic"):
        prepare_documents([article])


def test_preparation_is_repeatable_and_preserves_unicode(tmp_path, articles):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in articles), encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    manifest = prepare_corpus(corpus, output)
    first = output.read_bytes()
    assert prepare_corpus(corpus, output) == manifest
    assert output.read_bytes() == first
    assert manifest["language_counts"] == {"ar": 1092, "en": 1149}
    assert manifest["repeal_notice_count"] == 56
    assert "العقد شريعة المتعاقدين" in output.read_text(encoding="utf-8")


def test_loader_rejects_duplicate_numbers(tmp_path, by_number):
    corpus = tmp_path / "invalid.jsonl"
    corpus.write_text("\n".join(json.dumps(by_number[1]) for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="consecutive integers"):
        load_articles(corpus)
