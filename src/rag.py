"""Article-first ingestion contracts for the retrieval baseline.

Prepare source-traceable chunks without selecting an embedding provider yet.
Source text is returned verbatim; normalized text is only embedding input.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path("data/egyptian_civil_code_articles.jsonl")
CORPUS_ID = "egyptian-civil-code"


@dataclass(frozen=True)
class ArticleMetadata:
    corpus_id: str
    article_number: int
    citation: str
    book: str | None
    chapter: str | None
    section: str | None
    topic: str | None
    source_book: str | None
    source_page: int
    language: str
    is_repealed: bool
    repeal_note: str | None
    qa_flags: tuple[str, ...]
    content_kind: str


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    chunk_index: int
    source_field: str
    source_start: int
    source_end: int
    source_text: str
    embedding_text: str
    metadata: ArticleMetadata


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked passage; scores are backend-specific, not confidence values."""

    document: RetrievalDocument
    rank: int
    score: float


def load_articles(path: Path) -> list[dict[str, Any]]:
    """Read structured JSONL; reject broken numbering and source metadata."""
    articles = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not articles:
        raise ValueError("Corpus is empty")
    for expected_number, article in enumerate(articles, start=1):
        number = article["article_number"]
        if type(number) is not int or number != expected_number:
            raise ValueError("Article numbers must be consecutive integers starting at 1")
        if article["citation"] != f"Egyptian Civil Code, Article {number}":
            raise ValueError(f"Article {number}: citation does not match article number")
        if type(article["is_repealed"]) is not bool:
            raise ValueError(f"Article {number}: invalid repeal status")
        if type(article["source_page"]) is not int or article["source_page"] <= 0:
            raise ValueError(f"Article {number}: invalid source page")
    return articles


def prepare_documents(articles: list[dict[str, Any]]) -> list[RetrievalDocument]:
    """Create one whole-article chunk per available language, with zero overlap.

    Repealed articles produce one English source-notice chunk. Empty Arabic is
    accepted only with the extractor's explicit missing-language source flag.
    Full corpus validation is a separate prerequisite for an eventual index build.
    """
    documents = []
    for article in articles:
        number = article["article_number"]
        if article["is_repealed"]:
            if not article["repeal_note"] or not article["repeal_note"].strip():
                raise ValueError(f"Article {number}: missing repeal notice")
            if article["ar_text"] or article["text_en"]:
                raise ValueError(f"Article {number}: repealed record has operative text")
            texts = [("en", "repeal_note", article["repeal_note"], article["repeal_note"])]
        else:
            if not article["text_en"].strip():
                raise ValueError(f"Article {number}: missing English text")
            if not article["ar_text"].strip():
                if "missing_ar_text_in_source" not in article["qa_flags"]:
                    raise ValueError(f"Article {number}: unexplained missing Arabic")
                texts = []
            else:
                if not article["ar_text_normalized"].strip():
                    raise ValueError(f"Article {number}: missing normalized Arabic")
                texts = [("ar", "ar_text", article["ar_text"], article["ar_text_normalized"])]
            texts.append(("en", "text_en", article["text_en"], article["text_en"]))

        for language, source_field, source_text, embedding_text in texts:
            metadata = ArticleMetadata(
                corpus_id=CORPUS_ID,
                article_number=number,
                citation=article["citation"],
                book=article["book"],
                chapter=article["chapter"],
                section=article["section"],
                topic=article["topic"],
                source_book=article["source_book"],
                source_page=article["source_page"],
                language=language,
                is_repealed=article["is_repealed"],
                repeal_note=article["repeal_note"],
                qa_flags=tuple(article["qa_flags"]),
                content_kind="repeal_notice" if article["is_repealed"] else "article_text",
            )
            documents.append(RetrievalDocument(
                document_id=f"{CORPUS_ID}:{number}:{language}:0",
                chunk_index=0,
                source_field=source_field,
                source_start=0,
                source_end=len(source_text),
                source_text=source_text,
                embedding_text=embedding_text,
                metadata=metadata,
            ))
    return documents


def prepare_corpus(path: Path, output: Path) -> dict[str, Any]:
    """Write deterministic JSONL chunks and a preparation manifest."""
    documents = prepare_documents(load_articles(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(asdict(doc), ensure_ascii=False) + "\n" for doc in documents),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "corpus_id": CORPUS_ID,
        "corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chunk_strategy": "whole_article_per_language",
        "chunk_size": None,
        "chunk_size_units": "whole_article",
        "overlap": 0,
        "embedding_model": None,
        "document_count": len(documents),
        "article_count": len({doc.metadata.article_number for doc in documents}),
        "language_counts": {
            lang: sum(doc.metadata.language == lang for doc in documents)
            for lang in ("ar", "en")
        },
        "repeal_notice_count": sum(doc.metadata.is_repealed for doc in documents),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_corpus(args.corpus, args.out), indent=2))


if __name__ == "__main__":
    main()
