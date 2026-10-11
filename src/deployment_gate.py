"""Offline promotion gate: pinned ONNX model, current scores, identical full contexts.

Reuse of measured generation scores is allowed only when every generator input
in the complete fixed evaluation remains unchanged. This is not a new judge run.
"""

import hashlib
import json
from pathlib import Path

from .quality_gate import ROOT, check_current, fingerprint


def context_hash(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def runtime_fingerprint(root, model_metadata):
    root = Path(root)
    files = ("src/api.py", "src/deployment.py", "src/deployment_gate.py", "src/benchmark_runtime.py",
             "src/runtime_artifacts.py", "requirements/runtime.lock", "requirements/runtime.in")
    value = {"pipeline": fingerprint(root), "model": model_metadata,
             "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_equivalence(report, baseline, expected_fingerprint, model):
    if report.get("runtime_fingerprint") != expected_fingerprint or report.get("model") != model:
        raise ValueError("Deployment validation is stale or refers to another model")
    originals = {r["id"]: r for r in baseline["records"]}
    records = report.get("records", [])
    if len(records) != len(originals) or {r["id"] for r in records} != set(originals):
        raise ValueError("Runtime validation must cover every fixed question exactly once")
    for row in records:
        original = originals[row["id"]]
        if (row.get("context_identical") is not True
                or row.get("context_sha256") != context_hash(original["retrieved_contexts"])
                or row.get("retrieved_article_numbers") != original["retrieved_article_numbers"]):
            raise ValueError(f"Generator input changed for {row['id']}; fresh answer evaluation is required")
    return len(records)


def check_deployment(root=ROOT):
    root = Path(root)
    quality = check_current(root)
    descriptor = json.loads((root / "config/runtime-artifacts.json").read_text())
    report = json.loads((root / "reports/deployment/compact-retrieval.json").read_text())
    summary = json.loads((root / "reports/experiments/summary.json").read_text())
    path = root / summary["validation_file"]
    if report.get("baseline_report_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Runtime comparison does not match the current full evaluation")
    baseline = json.loads(path.read_text())
    for key in ("model_directory_sha256", "index_directory_sha256"):
        if descriptor[key] != report.get(key):
            raise ValueError("Release files differ from the validated runtime artifacts")
    for key in ("corpus_sha256", "chunks_sha256"):
        if any(descriptor[key] != row["result"]["configuration"][key] for row in baseline["records"]):
            raise ValueError("Release artifacts do not match the evaluated corpus/index")
    count = check_equivalence(report, baseline, runtime_fingerprint(root, descriptor["model"]), descriptor["model"])
    return {**quality, "identical_generator_contexts": count,
            "precision": descriptor["model"]["precision"],
            "validation": "full-context equivalence; existing generation scores reused"}


def main():
    try:
        print(json.dumps(check_deployment(), indent=2))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"Deployment gate failed: {exc}") from None


if __name__ == "__main__":
    main()
