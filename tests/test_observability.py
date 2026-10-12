"""Tracing records the rejected candidate and failing check, and never breaks answering."""

from contextlib import contextmanager

from src.observability import Tracer
from tests.test_generation import service, supported


class FakeObservation:
    trace_id = "trace-1"

    def __init__(self, record):
        self.record = record

    def update(self, **fields):
        self.record.update(fields)

    def score_trace(self, **fields):
        self.record.setdefault("scores", []).append(fields)


class FakeClient:
    def __init__(self, fail=False):
        self.records, self.flushed, self.fail = [], False, fail

    @contextmanager
    def start_as_current_observation(self, **fields):
        if self.fail:
            raise RuntimeError("backend down")
        record = dict(fields)
        self.records.append(record)
        yield FakeObservation(record)

    def flush(self):
        self.flushed = True


def traced(by_number, answer, client, **config):
    app, generator, _ = service(by_number, answer, **config)
    app.tracer = Tracer(client)
    return app, generator


def test_rejected_grounding_is_kept_with_candidate_and_error(by_number):
    client = FakeClient()
    app, _ = traced(by_number, supported(quote="not in the article"), client)
    result = app.ask("ما القوة الملزمة للعقد؟")
    assert result["reason"] == "invalid_source_grounding"
    assert result["reason_detail"] == "Evidence is not an exact source excerpt"
    assert result["trace_id"] == "trace-1"
    names = {r["name"]: r for r in client.records}
    assert list(names) == ["ask", "retrieval", "generation", "grounding"]
    assert names["generation"]["output"]["evidence"][0]["quote"] == "not in the article"
    assert names["generation"]["usage_details"] == {"total_token_count": 10}
    assert names["grounding"]["output"] == {"valid": False, "error": "Evidence is not an exact source excerpt"}
    assert names["ask"]["output"]["reason"] == "invalid_source_grounding"


def test_valid_answer_is_traced_without_reason_detail(by_number):
    client = FakeClient()
    app, _ = traced(by_number, supported(), client)
    result = app.ask("ما القوة الملزمة للعقد؟")
    assert "reason_detail" not in result
    assert {r["name"]: r for r in client.records}["grounding"]["output"] == {"valid": True}


def test_tracing_backend_failure_does_not_affect_answers(by_number):
    app, _ = traced(by_number, supported(), FakeClient(fail=True))
    assert app.ask("ما القوة الملزمة للعقد؟")["sources"] == ["Egyptian Civil Code, Article 147"]


def test_disabled_tracer_adds_no_trace_id(by_number):
    app, _, _ = service(by_number, supported())
    app.tracer = Tracer()
    assert "trace_id" not in app.ask("ما القوة الملزمة للعقد؟")


def test_tracer_needs_both_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    assert not Tracer.from_environment(tmp_path / ".env").enabled
