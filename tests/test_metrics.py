"""Prometheus metrics, drift baseline and sampled online faithfulness (all offline)."""

import json
from threading import Event

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
import pytest

from src.api import create_app
from src.drift import DriftBaseline, build_baseline, load_baseline
from src.generation import AnsweringService, GenerationError, GenerationResponse, load_generation_config
from src.metrics import Metrics, estimate_cost
from src.observability import Tracer
from src.retrieval import load_config
from src.online_eval import FaithfulnessSampler
from tests.test_api import FakeService
from tests.test_generation import FakeGenerator, FakeRetriever, supported

PRICING = {"input": 0.25, "output": 1.5}
USAGE = {"prompt_token_count": 1000, "candidates_token_count": 100, "thoughts_token_count": 50, "total_token_count": 1150}


def sample(metrics, name, **labels):
    return metrics.registry.get_sample_value(name, labels) or 0


def make_service(by_number, answer, metrics, **kwargs):
    retriever = FakeRetriever([by_number[147]])
    generator = FakeGenerator(answer)
    original = generator.generate
    generator.generate = lambda *a, **k: GenerationResponse(original(*a, **k).answer, "test-model", dict(USAGE))
    return AnsweringService(retriever, generator, load_generation_config(), metrics=metrics, sampler=kwargs.pop("sampler", None), **kwargs)


def test_cost_estimate_bills_thinking_as_output_and_skips_unpriced():
    assert estimate_cost(USAGE, PRICING) == pytest.approx((1000 * 0.25 + 150 * 1.5) / 1e6)
    assert estimate_cost(USAGE, None) is None and estimate_cost({}, PRICING) is None


def test_grounded_answer_records_outcome_stages_tokens_and_cost(by_number):
    metrics = Metrics(CollectorRegistry(), pricing=PRICING)
    make_service(by_number, supported(), metrics).ask("ما القوة الملزمة للعقد؟")
    assert sample(metrics, "legal_qa_answers_total", outcome="grounded") == 1
    for stage in ("retrieval", "generation", "total"):
        assert sample(metrics, "legal_qa_stage_seconds_count", stage=stage) == 1
    assert sample(metrics, "legal_qa_tokens_total", type="prompt") == 1000
    assert sample(metrics, "legal_qa_tokens_total", type="completion") == 100
    assert sample(metrics, "legal_qa_tokens_total", type="thinking") == 50
    assert sample(metrics, "legal_qa_estimated_cost_usd_total") == pytest.approx(0.000475)


def test_rejected_grounding_and_provider_errors_are_counted(by_number):
    metrics = Metrics(CollectorRegistry(), pricing=PRICING)
    make_service(by_number, supported(quote="not in the article"), metrics).ask("ما القوة الملزمة للعقد؟")
    assert sample(metrics, "legal_qa_answers_total", outcome="invalid_grounding") == 1

    service = make_service(by_number, supported(), metrics)
    failure = GenerationError("Google generation failed (HTTP 429)")
    failure.status_code = 429
    service.generator.generate = lambda *a, **k: (_ for _ in ()).throw(failure)
    with pytest.raises(GenerationError):
        service.ask("ما القوة الملزمة للعقد؟")
    assert sample(metrics, "legal_qa_answers_total", outcome="provider_error") == 1
    assert sample(metrics, "legal_qa_provider_errors_total", status="429") == 1


def test_query_vectors_reach_drift_histogram_only_for_retrievers_that_expose_them(by_number):
    config = load_config()
    baseline = DriftBaseline({"centroid": [1.0, 0.0], "cosine_mean": 0.9, "cosine_std": 0.02,
                              **{key: config[key] for key in ("model_id", "model_revision", "dimensions")}})
    metrics = Metrics(CollectorRegistry(), drift=baseline)
    assert sample(metrics, "legal_qa_query_centroid_cosine_baseline_mean") == 0.9

    class VectorRetriever(FakeRetriever):
        supports_query_vector = True

        def search(self, question, k, filters, on_query_vector=None):
            on_query_vector([1.0, 1.0])
            return super().search(question, k, filters)

    service = make_service(by_number, supported(), metrics)
    service.ask("ما القوة الملزمة للعقد؟")
    assert sample(metrics, "legal_qa_query_centroid_cosine_count") == 0
    service.retriever = VectorRetriever([by_number[147]])
    service.ask("ما القوة الملزمة للعقد؟")
    assert sample(metrics, "legal_qa_query_centroid_cosine_count") == 1
    assert sample(metrics, "legal_qa_query_centroid_cosine_sum") == pytest.approx(2 ** -0.5)


def test_baseline_is_built_from_normalized_queries_and_rejects_other_models(tmp_path):
    class Encoder:
        def encode(self, texts, query):
            assert query
            return __import__("numpy").array([[1.0, 0.0], [0.0, 1.0]])[: len(texts)]

    config = {"model_id": "m", "model_revision": "r", "dimensions": 2}
    data = build_baseline(Encoder(), config, ["a", "b"], lambda text: text.upper())
    assert data["question_count"] == 2 and data["cosine_mean"] == pytest.approx(2 ** -0.5, abs=1e-5)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data))
    assert load_baseline(path, config) is not None
    assert load_baseline(path, dict(config, model_revision="other")) is None
    assert load_baseline(tmp_path / "missing.json") is None


def test_metrics_endpoint_and_http_counters_use_route_templates():
    metrics = Metrics(CollectorRegistry())
    with TestClient(create_app(FakeService, metrics=metrics)) as client:
        client.post("/ask", json={"question": "contract"})
        client.post("/ask", json={"question": "  "})
        body = client.get("/metrics")
    assert body.status_code == 200 and "legal_qa_http_requests_total" in body.text
    assert sample(metrics, "legal_qa_http_requests_total", method="POST", path="/ask", status="200") == 1
    assert sample(metrics, "legal_qa_http_requests_total", method="POST", path="/ask", status="422") == 1


class RecordingTracer(Tracer):
    def __init__(self):
        super().__init__()
        self.scores = []

    def score(self, trace_id, name, value):
        self.scores.append((trace_id, name, value))


class FakeJudge:
    def __init__(self, score=0.8, fail=False):
        self.value, self.fail, self.done = score, fail, Event()

    async def score(self, row, answer, context):
        self.done.set()
        if self.fail:
            raise RuntimeError("judge down")
        return {"faithfulness": self.value}


def grounded(trace_id="t1"):
    return {"answer": "A", "insufficient_evidence": False, "trace_id": trace_id}


def test_sampler_scores_grounded_answers_and_attaches_to_trace():
    metrics, tracer, judge = Metrics(CollectorRegistry()), RecordingTracer(), FakeJudge(0.8)
    sampler = FaithfulnessSampler(1.0, lambda: judge, metrics, tracer)
    assert sampler.maybe_submit("q", grounded(), [{"article_number": 1}])
    assert not sampler.maybe_submit("q", {"insufficient_evidence": True, "answer": "no"}, [])
    sampler.close()
    assert sample(metrics, "legal_qa_faithfulness_sum") == pytest.approx(0.8)
    assert tracer.scores == [("t1", "faithfulness", 0.8)]


def test_sampler_respects_rate_backlog_and_failures():
    metrics, tracer = Metrics(CollectorRegistry()), RecordingTracer()
    assert not FaithfulnessSampler(0.1, FakeJudge, metrics, tracer, rng=lambda: 0.5).maybe_submit("q", grounded(), [])
    assert FaithfulnessSampler.from_environment(metrics, tracer, env={}) is None
    with pytest.raises(ValueError):
        FaithfulnessSampler.from_environment(metrics, tracer, env={"ONLINE_FAITHFULNESS_SAMPLE_RATE": "2"})

    sampler = FaithfulnessSampler(1.0, lambda: FakeJudge(fail=True), metrics, tracer, max_pending=1)
    sampler.pending = 1
    assert not sampler.maybe_submit("q", grounded(), [])
    sampler.pending = 0
    assert sampler.maybe_submit("q", grounded(), [])
    sampler.close()
    assert sample(metrics, "legal_qa_faithfulness_failures_total") == 1 and sampler.pending == 0


def test_sampler_disables_itself_without_evaluation_dependencies():
    def missing():
        raise ImportError("ragas")

    sampler = FaithfulnessSampler(1.0, missing, Metrics(CollectorRegistry()), RecordingTracer())
    sampler.maybe_submit("q", grounded(), [])
    sampler.close()
    assert sampler.disabled and not sampler.maybe_submit("q", grounded(), [])
