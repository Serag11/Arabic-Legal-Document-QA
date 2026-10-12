"""Prometheus metrics: outcomes, stage latency, tokens/cost, query drift and sampled faithfulness."""

import logging
import os
from pathlib import Path

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LOGGER = logging.getLogger(__name__)
ROOT = Path(os.environ.get("LEGAL_QA_ROOT", Path(__file__).resolve().parent.parent)).resolve()
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80)
SIMILARITY_BUCKETS = tuple(round(0.05 * step, 2) for step in range(1, 21))
OUTCOMES = {
    "no_retrieved_evidence": "no_evidence",
    "model_reported_insufficient_evidence": "model_refusal",
    "invalid_source_grounding": "invalid_grounding",
}


def estimate_cost(usage, pricing):
    """Estimated USD for one call; thinking tokens are billed as output. None if unpriced."""
    if not pricing or not usage:
        return None
    prompt = usage.get("prompt_token_count", 0)
    output = usage.get("candidates_token_count", 0) + usage.get("thoughts_token_count", 0)
    return (prompt * pricing["input"] + output * pricing["output"]) / 1_000_000


class Metrics:
    def __init__(self, registry=None, pricing=None, drift=None):
        self.registry = registry or CollectorRegistry()
        self.pricing = pricing
        self.drift = drift
        r = self.registry
        self.answers = Counter("legal_qa_answers_total", "Answers by outcome (grounded, model_refusal, invalid_grounding, "
                               "no_evidence, provider_error)", ["outcome"], registry=r)
        self.provider_errors = Counter("legal_qa_provider_errors_total", "Generation provider failures by HTTP status",
                                       ["status"], registry=r)
        self.stage_seconds = Histogram("legal_qa_stage_seconds", "Latency of pipeline stages", ["stage"],
                                       buckets=LATENCY_BUCKETS, registry=r)
        self.tokens = Counter("legal_qa_tokens_total", "Generation tokens by type", ["type"], registry=r)
        self.cost = Counter("legal_qa_estimated_cost_usd_total",
                            "Estimated generation cost in USD from configured prices", registry=r)
        self.top1_similarity = Histogram("legal_qa_top1_similarity", "Cosine similarity of the best retrieved article",
                                         buckets=SIMILARITY_BUCKETS, registry=r)
        self.centroid_cosine = Histogram("legal_qa_query_centroid_cosine",
                                         "Cosine of each query embedding to the baseline query centroid",
                                         buckets=SIMILARITY_BUCKETS, registry=r)
        self.centroid_baseline = Gauge("legal_qa_query_centroid_cosine_baseline_mean",
                                       "Mean centroid cosine of the reference evaluation questions", registry=r)
        self.centroid_baseline_std = Gauge("legal_qa_query_centroid_cosine_baseline_std",
                                           "Standard deviation of the reference centroid cosine", registry=r)
        self.faithfulness = Histogram("legal_qa_faithfulness", "Sampled online RAGAS faithfulness of grounded answers",
                                      buckets=SIMILARITY_BUCKETS, registry=r)
        self.faithfulness_failures = Counter("legal_qa_faithfulness_failures_total",
                                             "Sampled faithfulness evaluations that failed", registry=r)
        self.http_requests = Counter("legal_qa_http_requests_total", "HTTP requests", ["method", "path", "status"], registry=r)
        self.http_seconds = Histogram("legal_qa_http_request_seconds", "HTTP request duration", ["path"],
                                      buckets=LATENCY_BUCKETS, registry=r)
        if drift:
            self.centroid_baseline.set(drift.baseline_mean)
            self.centroid_baseline_std.set(drift.baseline_std)

    def observe_stage(self, stage, seconds):
        self.stage_seconds.labels(stage).observe(seconds)

    def observe_results(self, results):
        scores = [item["score"] for item in results[:1] if item.get("score") is not None]
        if scores:
            self.top1_similarity.observe(min(max(scores[0], 0.0), 1.0))

    def observe_query_vector(self, vector):
        if self.drift:
            self.centroid_cosine.observe(min(max(self.drift.cosine(vector), 0.0), 1.0))

    def observe_answer(self, result):
        reason = result.get("reason")
        self.answers.labels("grounded" if not result.get("insufficient_evidence") else OUTCOMES.get(reason, reason or "refused")).inc()
        usage = result.get("usage") or {}
        for kind, key in (("prompt", "prompt_token_count"), ("completion", "candidates_token_count"),
                          ("thinking", "thoughts_token_count")):
            if usage.get(key):
                self.tokens.labels(kind).inc(usage[key])
        cost = estimate_cost(usage, self.pricing)
        if cost:
            self.cost.inc(cost)

    def observe_provider_error(self, exc):
        self.answers.labels("provider_error").inc()
        self.provider_errors.labels(str(getattr(exc, "status_code", "unknown"))).inc()

    def render(self):
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


_default = None


def get_metrics():
    global _default
    if _default is None:
        _default = build_default_metrics()
    return _default


def build_default_metrics():
    import json

    pricing = None
    try:
        pricing = json.loads((ROOT / "config/generation.json").read_text(encoding="utf-8")).get("pricing_usd_per_million_tokens")
    except (OSError, ValueError):
        LOGGER.warning("Generation config unreadable; cost metrics disabled", exc_info=True)
    from .drift import load_baseline

    return Metrics(pricing=pricing, drift=load_baseline())
