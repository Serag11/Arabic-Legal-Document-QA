"""Offline CI gate for measured RAGAS scores and exact pipeline provenance.

No Google calls or model downloads. A changed pipeline requires fresh evaluation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FINGERPRINT_FILES = (
    "src/rag.py", "src/retrieval.py", "src/generation.py", "src/experiments.py",
    "src/quality_gate.py", "src/registry.py",
    "config/retrieval.json", "config/generation.json", "config/experiments.json",
    "prompts/answer_v1.txt", "prompts/faithfulness_claims_v1.txt", "uv.lock",
    "evaluation/answer_questions.jsonl", "evaluation/review.json",
)


def fingerprint(root=ROOT):
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FINGERPRINT_FILES}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def check_report(report, expected_fingerprint, expected_ids, threshold=0.75, minimum=20):
    if report.get("pipeline_fingerprint") != expected_fingerprint:
        raise ValueError("Evaluation is stale: pipeline/configuration/dataset changed")
    records = report.get("records", [])
    if len(records) != len(expected_ids) or {r["id"] for r in records} != set(expected_ids):
        raise ValueError("Full evaluation must cover every fixed question exactly once")
    if any("error" in r for r in records):
        raise ValueError("Evaluation contains operational/scoring errors")
    supported = [r for r in records if r["expected_article_numbers"]]
    scores = [r["faithfulness"] for r in supported if r.get("faithfulness") is not None]
    if len(scores) < minimum or not supported:
        raise ValueError(f"Need at least {minimum} scored supported questions")
    if any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Invalid/non-finite faithfulness scores")
    # Abstentions are not assigned a raw RAGAS faithfulness score. They contribute
    # zero to this coverage-adjusted gate so refusing everything cannot pass.
    adjusted = sum(scores) / len(supported)
    if adjusted < threshold:
        raise ValueError(f"Coverage-adjusted faithfulness {adjusted:.4f} is below {threshold:.2f}")
    return {"coverage_adjusted_faithfulness": adjusted, "scored_questions": len(scores),
            "supported_questions": len(supported), "threshold": threshold}


def check_current(root=ROOT):
    summary_path = root / "reports/experiments/summary.json"
    if not summary_path.exists():
        raise ValueError("No completed live experiment summary; run the Stage 5 evaluation")
    summary = json.loads(summary_path.read_text())
    if not summary.get("selected") or not summary.get("validation_file"):
        raise ValueError("No selected configuration with full validation")
    path = (root / summary["validation_file"]).resolve()
    path.relative_to(root.resolve())
    report = json.loads(path.read_text())
    rows = [json.loads(line) for line in (root / "evaluation/answer_questions.jsonl").read_text().splitlines()]
    if len(rows) < 50:
        raise ValueError("Final evaluation needs at least 50 language-specific questions")
    return check_report(report, fingerprint(root), [r["id"] for r in rows])


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        result = check_current()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"Quality gate failed: {exc}") from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
