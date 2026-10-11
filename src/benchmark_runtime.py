"""Offline ONNX retrieval comparison against the current 60-question evaluation."""

import argparse
import json
from pathlib import Path
import time

from .deployment import OnnxEncoder
from .deployment_gate import context_hash, runtime_fingerprint
from .generation import build_context, load_generation_config
from .retrieval import E5Encoder, Retriever, load_config, write_json
from .runtime_artifacts import directory_sha256, file_sha256


def benchmark(model, output):
    config = load_config()
    generation = load_generation_config()
    baseline = json.loads(Path("reports/experiments/selected-full-validation.json").read_text())
    reference = E5Encoder(config)
    sample = ["What is a lease?", "ما تعريف عقد الإيجار؟", "Who bears the costs of a servitude?"]
    expected = reference.encode(sample, query=True)
    del reference
    encoder = OnnxEncoder(config, model)
    actual = encoder.encode(sample, query=True)
    retriever = Retriever(encoder=encoder)
    records = []
    for row in baseline["records"]:
        start = time.perf_counter()
        found = retriever.search(row["question"], k=generation["retrieval_top_k"])
        numbers = [hit["article_number"] for hit in found]
        contexts = build_context(found, generation["max_context_characters"])
        ranks = [numbers.index(n) + 1 for n in row["expected_article_numbers"] if n in numbers]
        records.append({"id": row["id"], "language": row["language"],
                        "expected_article_numbers": row["expected_article_numbers"],
                        "retrieved_article_numbers": numbers,
                        "baseline_article_numbers": row["retrieved_article_numbers"],
                        "context_identical": json.loads(json.dumps(contexts)) == row["retrieved_contexts"],
                        "context_sha256": context_hash(contexts),
                        "hit": bool(ranks), "rr": 1 / min(ranks) if ranks else 0,
                        "seconds": time.perf_counter() - start})
    supported = [r for r in records if r["expected_article_numbers"]]
    report = {"model": encoder.metadata, "embedding_sample_cosine": (actual * expected).sum(axis=1).tolist(),
              "runtime_fingerprint": runtime_fingerprint(Path.cwd(), encoder.metadata),
              "baseline_report_sha256": file_sha256("reports/experiments/selected-full-validation.json"),
              "model_directory_sha256": directory_sha256(model),
              "index_directory_sha256": directory_sha256("data/retrieval/chroma"),
              "count": len(records), "supported_count": len(supported),
              "expected_article_hits": sum(r["hit"] for r in supported),
              "context_identical_count": sum(r["context_identical"] for r in records),
              "changed_context_ids": [r["id"] for r in records if not r["context_identical"]],
              "mean_query_seconds": sum(r["seconds"] for r in records) / len(records),
              "language_hits": {lang: sum(r["hit"] for r in supported if r["language"] == lang)
                                for lang in ("ar", "en")}, "records": records}
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.model, args.output)


if __name__ == "__main__":
    main()
