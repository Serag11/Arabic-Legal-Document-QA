"""Query-embedding drift: cosine distance of live queries to the reference evaluation-question centroid."""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
ROOT = Path(os.environ.get("LEGAL_QA_ROOT", Path(__file__).resolve().parent.parent)).resolve()
BASELINE_PATH = ROOT / "config/drift_baseline.json"
QUESTIONS_PATH = ROOT / "evaluation/answer_questions.jsonl"


class DriftBaseline:
    def __init__(self, data):
        self.data = data
        self.centroid = np.asarray(data["centroid"], dtype=np.float64)
        self.centroid /= np.linalg.norm(self.centroid)
        self.baseline_mean = data["cosine_mean"]
        self.baseline_std = data["cosine_std"]

    def cosine(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        return float(vector @ self.centroid / np.linalg.norm(vector))

    def matches(self, retrieval_config):
        return all(self.data[key] == retrieval_config[key] for key in ("model_id", "model_revision", "dimensions"))


def load_baseline(path=BASELINE_PATH, retrieval_config=None):
    """Returns the baseline, or None when absent/incompatible (drift metrics are then disabled)."""
    try:
        baseline = DriftBaseline(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        LOGGER.warning("No usable drift baseline at %s; drift metrics disabled", path)
        return None
    if retrieval_config is not None and not baseline.matches(retrieval_config):
        LOGGER.warning("Drift baseline was built with a different embedding model; drift metrics disabled")
        return None
    return baseline


def build_baseline(encoder, config, questions, normalize):
    vectors = encoder.encode([normalize(question) for question in questions], query=True)
    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    cosines = vectors @ centroid
    return {
        "schema_version": 1, "model_id": config["model_id"], "model_revision": config["model_revision"],
        "dimensions": config["dimensions"], "question_count": len(questions),
        "cosine_mean": round(float(cosines.mean()), 6), "cosine_std": round(float(cosines.std()), 6),
        "cosine_min": round(float(cosines.min()), 6),
        "centroid": [round(float(value), 7) for value in centroid],
    }


def main():
    from .retrieval import E5Encoder, load_config, normalize_arabic_for_search, write_json

    parser = argparse.ArgumentParser(description="Build the reference query centroid from the fixed evaluation questions")
    parser.add_argument("--questions", default=QUESTIONS_PATH, type=Path)
    parser.add_argument("--output", default=BASELINE_PATH, type=Path)
    args = parser.parse_args()
    config = load_config()
    rows = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines()]
    baseline = build_baseline(E5Encoder(config), config, [row["question"] for row in rows], normalize_arabic_for_search)
    write_json(args.output, baseline)
    print(f"Wrote {args.output}: mean cosine {baseline['cosine_mean']}, std {baseline['cosine_std']} over {len(rows)} questions")


if __name__ == "__main__":
    main()
