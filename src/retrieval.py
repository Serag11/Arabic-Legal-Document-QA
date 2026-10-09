"""Pinned E5 embeddings, persistent Chroma retrieval, and bilingual evaluation.

Network access is confined to the explicit `download` command. All preparation,
indexing, and query commands require the model to be cached locally.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import shutil
import tempfile
import time

if __package__:
    from .rag import ArticleMetadata, DEFAULT_CORPUS, RetrievalDocument, load_articles, prepare_documents
    from .extract_egyptian_civil_code import normalize_arabic_for_search
else:
    from rag import ArticleMetadata, DEFAULT_CORPUS, RetrievalDocument, load_articles, prepare_documents
    from extract_egyptian_civil_code import normalize_arabic_for_search


DEFAULT_CONFIG = Path("config/retrieval.json")
DEFAULT_CHUNKS = Path("data/retrieval/chunks.jsonl")
DEFAULT_INDEX = Path("data/retrieval/chroma")
PROVISION_MARKER = re.compile(r"(?<!\w)\(\s*[0-9٠-٩۰-۹]+\s*\)")
ARTICLE_QUERY = re.compile(r"(?:\barticle\s*|المادة\s*|مادة\s*)[(:\s]*([0-9٠-٩۰-۹]+)", re.I)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("Model revision must be an immutable commit SHA")
    if config["model_id"] != "intfloat/multilingual-e5-base":
        raise ValueError("This baseline adapter supports multilingual-e5-base only")
    if config["dimensions"] != 768 or config["max_tokens"] != 512:
        raise ValueError("E5-base requires 768 dimensions and a 512-token input budget")
    if config["query_prefix"] != "query: " or config["passage_prefix"] != "passage: ":
        raise ValueError("E5 requires literal query/passage prefixes")
    if config["overlap"] != 0 or not config["normalize_embeddings"] or config["distance"] != "cosine":
        raise ValueError("Baseline requires zero overlap and normalized cosine embeddings")
    if config["batch_size"] < 1:
        raise ValueError("Batch size must be positive")
    return config


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def download_model(config):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=config["model_id"], revision=config["model_revision"],
        allow_patterns=[
            "config.json", "model.safetensors", "modules.json", "1_Pooling/config.json",
            "sentence_bert_config.json", "sentencepiece.bpe.model", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json",
        ],
    )


def load_tokenizer(config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], local_files_only=True,
    )


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))


def split_document(doc, tokenizer, config):
    """Greedily pack whole source provisions; retain exact contiguous source slices.

    A provision exceeding the budget is an error, never silently truncated. An
    introductory passage stays attached to the first numbered provision.
    """
    prefix, limit = config["passage_prefix"], config["max_tokens"]
    if token_count(tokenizer, prefix + doc.embedding_text) <= limit:
        return [doc]
    markers = [m.start() for m in PROVISION_MARKER.finditer(doc.source_text)]
    starts = config.get("source_provision_starts", {}).get(doc.document_id)
    if starts:
        if any(doc.source_text.count(text) != 1 for text in starts):
            raise ValueError(f"{doc.document_id}: source provision anchors are not unique")
        markers = [doc.source_text.index(text) for text in starts]
        if markers != sorted(markers) or markers[0] != 0:
            raise ValueError(f"{doc.document_id}: source provision anchors are out of order")
    boundaries = [0] + [n for n in markers[1:] if n > 0] + [len(doc.source_text)]
    ranges = []
    start = boundaries[0]
    end = start
    for next_end in boundaries[1:]:
        candidate = doc.source_text[start:next_end]
        embedding = normalize_arabic_for_search(candidate) if doc.metadata.language == "ar" else candidate
        if token_count(tokenizer, prefix + embedding) <= limit:
            end = next_end
            continue
        if end == start:
            raise ValueError(f"{doc.document_id}: indivisible provision exceeds {limit} tokens")
        ranges.append((start, end))
        start, end = end, next_end
        piece = doc.source_text[start:end]
        embedding = normalize_arabic_for_search(piece) if doc.metadata.language == "ar" else piece
        if token_count(tokenizer, prefix + embedding) > limit:
            raise ValueError(f"{doc.document_id}: indivisible provision exceeds {limit} tokens")
    ranges.append((start, end))
    chunks = []
    for i, (start, end) in enumerate(ranges):
        source = doc.source_text[start:end]
        chunks.append(replace(
            doc, document_id=doc.document_id.rsplit(":", 1)[0] + f":{i}", chunk_index=i,
            source_start=start, source_end=end, source_text=source,
            embedding_text=normalize_arabic_for_search(source) if doc.metadata.language == "ar" else source,
        ))
    return chunks


def prepare(corpus, output, config):
    tokenizer = load_tokenizer(config)
    whole = prepare_documents(load_articles(corpus))
    whole_lengths = [token_count(tokenizer, config["passage_prefix"] + d.embedding_text) for d in whole]
    documents = [chunk for doc in whole for chunk in split_document(doc, tokenizer, config)]
    lengths = [token_count(tokenizer, config["passage_prefix"] + d.embedding_text) for d in documents]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(asdict(d), ensure_ascii=False) + "\n" for d in documents), encoding="utf-8")
    manifest = {
        "schema_version": 1, "corpus_sha256": sha256(corpus), "chunks_sha256": sha256(output),
        "normalization_code_sha256": sha256(Path(__file__).with_name("extract_egyptian_civil_code.py")),
        "config": config, "config_sha256": config_hash(config),
        "whole_document_count": len(whole), "document_count": len(documents),
        "article_count": len({d.metadata.article_number for d in documents}),
        "language_counts": {lang: sum(d.metadata.language == lang for d in documents) for lang in ("ar", "en")},
        "repeal_notice_count": sum(d.metadata.is_repealed for d in documents),
        "max_whole_tokens": max(whole_lengths), "max_chunk_tokens": max(lengths),
        "split_document_ids": [d.document_id for d, n in zip(whole, whole_lengths) if n > config["max_tokens"]],
        "token_counts_include_prefix_and_special_tokens": True,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def document_from_dict(value):
    value = dict(value)
    meta = dict(value.pop("metadata"))
    meta["qa_flags"] = tuple(meta["qa_flags"])
    return RetrievalDocument(metadata=ArticleMetadata(**meta), **value)


class E5Encoder:
    def __init__(self, config):
        import torch
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = config["device"]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = config["gpu_precision"] if self.device.startswith("cuda") else config["cpu_precision"]
        if self.device == "cpu":
            torch.set_num_threads(config["cpu_threads"])
        self.model = SentenceTransformer(
            config["model_id"], revision=config["model_revision"], device=self.device,
            local_files_only=True, model_kwargs={"torch_dtype": getattr(torch, self.precision), "use_safetensors": True},
        )
        self.model.max_seq_length = config["max_tokens"]
        self.tokenizer = self.model.tokenizer

    def encode(self, texts, *, query=False):
        prefix = self.config["query_prefix"] if query else self.config["passage_prefix"]
        inputs = [prefix + text for text in texts]
        for text in inputs:
            if token_count(self.tokenizer, text) > self.config["max_tokens"]:
                raise ValueError("Input exceeds embedding budget; truncation is forbidden")
        vectors = self.model.encode(
            inputs, batch_size=self.config["batch_size"], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=not query,
        )
        if vectors.shape != (len(inputs), self.config["dimensions"]):
            raise ValueError("Unexpected embedding dimensions")
        return vectors.astype("float32")


def chroma_client(path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def stored_metadata(doc):
    """Scalar fields are filterable; full payload preserves optional fields/lists."""
    meta = {key: value for key, value in asdict(doc.metadata).items()
            if value is not None and key != "qa_flags"}
    meta["payload"] = json.dumps(asdict(doc), ensure_ascii=False)
    return meta


def build_index(chunks, index, config, encoder=None):
    chunks, index = Path(chunks), Path(index)
    manifest = json.loads(chunks.with_suffix(".manifest.json").read_text())
    if manifest["config_sha256"] != config_hash(config) or manifest["chunks_sha256"] != sha256(chunks):
        raise ValueError("Prepared chunks/config do not match their manifest; prepare again")
    documents = [document_from_dict(json.loads(line)) for line in chunks.read_text().splitlines()]
    if len(documents) != manifest["document_count"] or len({d.document_id for d in documents}) != len(documents):
        raise ValueError("Prepared document count or unique IDs are invalid")
    if index.exists() and any(index.iterdir()) and not (index / "manifest.json").exists():
        raise ValueError("Refusing to rebuild an unrecognized index directory")
    load_start = time.perf_counter()
    encoder = encoder or E5Encoder(config)
    load_seconds = time.perf_counter() - load_start
    gpu = getattr(encoder, "device", "").startswith("cuda")
    if gpu:
        import torch
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    vectors = encoder.encode([d.embedding_text for d in documents])
    client = chroma_client(index)
    names = [c.name for c in client.list_collections()]
    if config["collection"] in names:
        client.delete_collection(config["collection"])
    collection = client.create_collection(
        config["collection"], embedding_function=None,
        configuration={"hnsw": {"space": "cosine", "num_threads": 1}},
        metadata={"config_sha256": config_hash(config), "chunks_sha256": sha256(chunks)},
    )
    for start_i in range(0, len(documents), 128):
        batch = documents[start_i:start_i + 128]
        collection.add(
            ids=[d.document_id for d in batch], documents=[d.source_text for d in batch],
            metadatas=[stored_metadata(d) for d in batch], embeddings=vectors[start_i:start_i + 128].tolist(),
        )
    result = dict(manifest, indexed_count=collection.count(), index_seconds=time.perf_counter() - start,
                  model_load_seconds=load_seconds,
                  peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2 if gpu else None,
                  peak_gpu_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2 if gpu else None,
                  runtime_device=getattr(encoder, "device", "test"), runtime_precision=getattr(encoder, "precision", "test"),
                  packages={name: importlib.metadata.version(name) for name in ("torch", "sentence-transformers", "transformers", "chromadb")})
    write_json(index / "manifest.json", result)
    return result


def make_filter(filters):
    allowed = {"language", "book", "chapter", "section", "topic", "source_book", "article_number", "is_repealed", "content_kind"}
    if set(filters) - allowed:
        raise ValueError("Unknown metadata filter")
    if "article_number" in filters and type(filters["article_number"]) is not int:
        raise ValueError("article_number filter must be an integer")
    if "is_repealed" in filters and type(filters["is_repealed"]) is not bool:
        raise ValueError("is_repealed filter must be a boolean")
    if "language" in filters and filters["language"] not in ("ar", "en"):
        raise ValueError("language must be ar or en")
    if any(value is None for value in filters.values()):
        raise ValueError("Filter values cannot be null")
    clauses = [{key: {"$eq": value}} for key, value in filters.items()]
    return {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)


def explicit_article_number(question):
    if re.fullmatch(r"\s*[0-9٠-٩۰-۹]+\s*", question):
        return int(question.strip())
    numbers = {int(match.group(1)) for match in ARTICLE_QUERY.finditer(question)}
    return next(iter(numbers)) if len(numbers) == 1 else None


def group_results(documents, distances, k, method):
    groups = {}
    for doc, distance in zip(documents, distances):
        number = doc.metadata.article_number
        if number not in groups:
            groups[number] = {"article_number": number, "citation": doc.metadata.citation,
                              "score": None if distance is None else 1.0 - distance,
                              "method": method, "passages": []}
        groups[number]["passages"].append(asdict(doc))
    result = list(groups.values())[:k]
    for rank, item in enumerate(result, 1):
        item["rank"] = rank
    return result


class Retriever:
    def __init__(self, index=DEFAULT_INDEX, config=None, encoder=None):
        index = Path(index)
        self.manifest = json.loads((index / "manifest.json").read_text())
        self.config = config or self.manifest["config"]
        if config_hash(self.config) != self.manifest["config_sha256"]:
            raise ValueError("Query configuration differs from the index configuration")
        # Chroma updates SQLite housekeeping even when only reading. Query a
        # runtime copy so loading/evaluation cannot mutate DVC-tracked artifacts.
        self._runtime = tempfile.TemporaryDirectory(prefix="legal-qa-chroma-")
        runtime_index = Path(self._runtime.name) / "chroma"
        shutil.copytree(index, runtime_index)
        self.client = chroma_client(runtime_index)
        self.collection = self.client.get_collection(self.config["collection"], embedding_function=None)
        if self.collection.count() != self.manifest["indexed_count"]:
            raise ValueError("Index document count differs from its manifest")
        if self.collection.metadata["config_sha256"] != config_hash(self.config):
            raise ValueError("Collection configuration differs from its manifest")
        self.encoder = encoder

    def search(self, question, k=5, filters=None, direct_lookup=True):
        if not question.strip() or k < 1:
            raise ValueError("Question must be nonempty and k must be positive")
        filters = dict(filters or {})
        where = make_filter(filters)
        number = explicit_article_number(question) if direct_lookup else None
        if number is not None:
            if "article_number" in filters and filters["article_number"] != number:
                return []
            filters["article_number"] = number
            found = self.collection.get(where=make_filter(filters), include=["metadatas"])
            docs = [document_from_dict(json.loads(meta["payload"])) for meta in found["metadatas"]]
            docs.sort(key=lambda d: (d.metadata.language, d.chunk_index))
            return group_results(docs, [None] * len(docs), k, "article_lookup")
        self.encoder = self.encoder or E5Encoder(self.config)
        vector = self.encoder.encode([normalize_arabic_for_search(question)], query=True)[0].tolist()
        count = self.collection.count()
        n = min(count, max(k * 4, k))
        while True:
            found = self.collection.query(query_embeddings=[vector], n_results=n, where=where, include=["metadatas", "distances"])
            docs = [document_from_dict(json.loads(meta["payload"])) for meta in found["metadatas"][0]]
            result = group_results(docs, found["distances"][0], k, "semantic")
            if len(result) >= k or n >= count or len(docs) < n:
                return result
            n = min(count, n * 2)


def evaluate(retriever, questions, output, k=5):
    pairs = [json.loads(line) for line in Path(questions).read_text().splitlines()]
    records = []
    for pair in pairs:
        for language in ("ar", "en"):
            start = time.perf_counter()
            results = retriever.search(pair[f"question_{language}"], k=k)
            numbers = [r["article_number"] for r in results]
            expected = pair["expected_article_numbers"]
            ranks = [numbers.index(n) + 1 for n in expected if n in numbers]
            records.append({
                "id": pair["id"], "language": language, "category": pair["category"],
                "question": pair[f"question_{language}"], "expected_article_numbers": expected,
                "retrieved_article_numbers": numbers, "results": results,
                "seconds": time.perf_counter() - start,
                "hit_at_k": int(bool(ranks)) if expected else None,
                "recall_at_k": len(ranks) / len(expected) if expected else None,
                "reciprocal_rank": 1 / min(ranks) if ranks else (0 if expected else None),
                "method": results[0]["method"] if results else ("article_lookup" if explicit_article_number(pair[f"question_{language}"]) is not None else "semantic"),
            })
    metrics = {}
    for subset in ("all_supported", "semantic_only", "article_lookup"):
        metrics[subset] = {}
        for language in ("ar", "en"):
            group = [r for r in records if r["language"] == language and r["expected_article_numbers"]
                     and (subset == "all_supported" or r["method"] == ("semantic" if subset == "semantic_only" else "article_lookup"))]
            metrics[subset][language] = {"count": len(group), **{
                name: sum(r[name] for r in group) / len(group) if group else None
                for name in ("hit_at_k", "recall_at_k", "reciprocal_rank")}}
    supported = [p for p in pairs if p["expected_article_numbers"]]
    matched = {r["id"]: {} for r in records}
    for r in records:
        matched[r["id"]][r["language"]] = r
    agreement = sum(matched[p["id"]]["ar"]["retrieved_article_numbers"][:1] == matched[p["id"]]["en"]["retrieved_article_numbers"][:1] for p in supported) / len(supported)
    summary = {"draft_ground_truth": True, "k": k, "pair_count": len(pairs), "metrics": metrics,
               "bilingual_top1_agreement_supported": agreement, "config": retriever.config,
               "corpus_sha256": retriever.manifest["corpus_sha256"],
               "unsupported_probes_excluded_from_metrics": sum(not p["expected_article_numbers"] for p in pairs),
               "questions_sha256": sha256(questions), "records": records}
    write_json(output, summary)
    write_json(Path(output).with_suffix(".metrics.json"), {
        "k": k, "pair_count": len(pairs), "metrics": metrics,
        "bilingual_top1_agreement_supported": agreement,
    })
    return {key: value for key, value in summary.items() if key != "records"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    prep = commands.add_parser("prepare")
    prep.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    prep.add_argument("--out", type=Path, default=DEFAULT_CHUNKS)
    index = commands.add_parser("index")
    index.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    index.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    query = commands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    query.add_argument("--k", type=int, default=5)
    query.add_argument("--filters", type=json.loads, default={})
    query.add_argument("--semantic-only", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    evaluation.add_argument("--questions", type=Path, default=Path("evaluation/retrieval_questions.jsonl"))
    evaluation.add_argument("--out", type=Path, default=Path("reports/retrieval-results.json"))
    evaluation.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "download":
        result = {"model_cache": download_model(config)}
    elif args.command == "prepare":
        result = prepare(args.corpus, args.out, config)
    elif args.command == "index":
        result = build_index(args.chunks, args.index, config)
    elif args.command == "query":
        result = Retriever(args.index, config).search(args.question, args.k, args.filters, not args.semantic_only)
    else:
        result = evaluate(Retriever(args.index, config), args.questions, args.out, args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
