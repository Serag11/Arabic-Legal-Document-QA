"""Incremental batch re-indexing: add, update or remove whole corpora in an existing Chroma index.

Each corpus is prepared exactly like the main build (same normalization, token-safe
chunking, stable `<corpus_id>:<article>:<lang>:<n>` IDs). The job then compares content
hashes with what the index holds for that corpus and embeds only new or changed chunks.
Running it again with unchanged input embeds nothing and does not load the embedding model.
Write the result to an index you name explicitly; the DVC-tracked index is not a default.
"""

import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import re

if __package__:
    from .rag import prepare_documents, load_articles
    from .retrieval import (
        E5Encoder, chroma_client, config_hash, load_config, load_tokenizer, sha256, split_document, stored_metadata,
        write_json,
    )
else:
    from rag import prepare_documents, load_articles
    from retrieval import (
        E5Encoder, chroma_client, config_hash, load_config, load_tokenizer, sha256, split_document, stored_metadata,
        write_json,
    )

CORPUS_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
BATCH = 128
HISTORY_LIMIT = 50


def payload_hash(document):
    return hashlib.sha256(json.dumps(asdict(document), ensure_ascii=False).encode("utf-8")).hexdigest()


def citation_prefix(path):
    """'<Title>' from the first record's '<Title>, Article 1'; load_articles then checks every record."""
    first = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
    prefix, separator, number = first["citation"].rpartition(", Article ")
    if not separator or number != "1" or not prefix:
        raise ValueError(f"{path}: the first record's citation must look like '<Title>, Article 1'")
    return prefix


def prepare_corpus_chunks(path, corpus_id, config, tokenizer):
    articles = load_articles(Path(path), citation_prefix(path))
    whole = prepare_documents(articles, corpus_id)
    chunks = [chunk for document in whole for chunk in split_document(document, tokenizer, config)]
    if len({chunk.document_id for chunk in chunks}) != len(chunks):
        raise ValueError(f"{corpus_id}: duplicate chunk IDs")
    return chunks


def indexed_hashes(collection, corpus_id):
    found = collection.get(where={"corpus_id": {"$eq": corpus_id}}, include=["metadatas"])
    return {
        identifier: hashlib.sha256(meta["payload"].encode("utf-8")).hexdigest()
        for identifier, meta in zip(found["ids"], found["metadatas"])
    }


def diff(current, existing):
    """current: id -> document; existing: id -> payload hash."""
    return {
        "added": sorted(set(current) - set(existing)),
        "changed": sorted(i for i in set(current) & set(existing) if payload_hash(current[i]) != existing[i]),
        "removed": sorted(set(existing) - set(current)),
        "unchanged": sum(1 for i in set(current) & set(existing) if payload_hash(current[i]) == existing[i]),
    }


def corpus_statistics(collection):
    metas = collection.get(include=["metadatas"])["metadatas"]
    articles = {(m["corpus_id"], m["article_number"]) for m in metas}
    per_corpus = {}
    for meta in metas:
        per_corpus[meta["corpus_id"]] = per_corpus.get(meta["corpus_id"], 0) + 1
    return len(metas), len(articles), per_corpus


def reindex(index, corpora, config, *, remove=(), encoder=None, tokenizer=None, dry_run=False, now=None):
    """Update `index` in place. corpora: {corpus_id: path to a corpus JSONL}. Returns a change report."""
    index = Path(index)
    manifest_path = index / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Not an existing index (manifest.json missing); build one first with the `index` command")
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != config_hash(config):
        raise ValueError("Embedding/chunking configuration differs from the index; a full rebuild is required")
    for corpus_id in [*corpora, *remove]:
        if not CORPUS_ID.fullmatch(corpus_id):
            raise ValueError(f"Invalid corpus ID {corpus_id!r}: use lowercase letters, digits and hyphens")
    if set(corpora) & set(remove):
        raise ValueError("A corpus cannot be both updated and removed")

    client = chroma_client(index)
    collection = client.get_collection(config["collection"], embedding_function=None)
    tokenizer = tokenizer or (load_tokenizer(config) if corpora else None)

    plans, pending = {}, []
    for corpus_id, path in corpora.items():
        chunks = {c.document_id: c for c in prepare_corpus_chunks(path, corpus_id, config, tokenizer)}
        plan = diff(chunks, indexed_hashes(collection, corpus_id))
        plan["corpus_sha256"] = sha256(path)
        plan["document_count"] = len(chunks)
        plans[corpus_id] = (plan, chunks)
        pending += [chunks[i] for i in plan["added"] + plan["changed"]]
    removals = {corpus_id: sorted(indexed_hashes(collection, corpus_id)) for corpus_id in remove}
    report = {
        "dry_run": dry_run,
        "corpora": {
            corpus_id: {key: (len(value) if isinstance(value, list) else value) for key, value in plan.items()}
            for corpus_id, (plan, _) in plans.items()
        },
        "removed_corpora": {corpus_id: len(ids) for corpus_id, ids in removals.items()},
    }
    changed = bool(pending) or any(removals.values()) or any(p["removed"] for p, _ in plans.values())
    report["embedded_chunks"] = len(pending)
    if dry_run or not changed:
        report["indexed_count"] = collection.count()
        return report

    # Embed first, write second: a failure while embedding leaves the index untouched, and a
    # failure part-way through writing is repaired by simply running the same job again.
    vectors = None
    if pending:
        encoder = encoder or E5Encoder(config)
        vectors = encoder.encode([d.embedding_text for d in pending])
    for start in range(0, len(pending), BATCH):
        batch = pending[start:start + BATCH]
        collection.upsert(
            ids=[d.document_id for d in batch], documents=[d.source_text for d in batch],
            metadatas=[stored_metadata(d) for d in batch], embeddings=vectors[start:start + BATCH].tolist(),
        )
    stale = [i for plan, _ in plans.values() for i in plan["removed"]] + [i for ids in removals.values() for i in ids]
    for start in range(0, len(stale), BATCH):
        collection.delete(ids=stale[start:start + BATCH])

    count, article_count, per_corpus = corpus_statistics(collection)
    timestamp = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat(timespec="seconds")
    corpora_state = dict(manifest.get("corpora", {}))
    for corpus_id in remove:
        corpora_state.pop(corpus_id, None)
    for corpus_id, (plan, _) in plans.items():
        corpora_state[corpus_id] = {"corpus_sha256": plan["corpus_sha256"], "document_count": plan["document_count"],
                                    "updated_at": timestamp}
    history = list(manifest.get("reindex_history", []))
    history.append({"at": timestamp, **{k: v for k, v in report.items() if k != "dry_run"}})
    manifest.update(indexed_count=count, article_count=article_count, corpus_document_counts=per_corpus,
                    corpora=corpora_state, reindex_history=history[-HISTORY_LIMIT:])
    write_json(manifest_path, manifest)
    report["indexed_count"] = count
    return report


def run_cli(args, config):
    def pairs(values, what):
        result = {}
        for value in values:
            key, sep, rest = value.partition("=")
            if not sep or not key or not rest:
                raise SystemExit(f"{what} must look like ID=VALUE, got {value!r}")
            result[key] = rest
        return result

    corpora = {key: Path(path) for key, path in pairs(args.corpus, "--corpus").items()}
    if not corpora and not args.remove:
        raise SystemExit("Nothing to do: pass --corpus ID=PATH and/or --remove ID")
    return reindex(args.index, corpora, config, remove=args.remove, dry_run=args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--remove", action="append", default=[], metavar="ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    print(json.dumps(run_cli(args, config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
