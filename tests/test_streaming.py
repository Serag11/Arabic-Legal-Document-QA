"""Incremental answer extraction and the /ask/stream server-sent-event contract."""

import json

from fastapi.testclient import TestClient
import pytest

from src.api import create_app
from src.generation import AnsweringService, GeneratedAnswer, GenerationError, GenerationResponse
from src.streaming import AnswerTextStream
from tests.test_api import FakeService
from tests.test_generation import FakeRetriever, supported
from src.generation import load_generation_config


def feed_all(payload, size):
    stream, out = AnswerTextStream(), []
    for start in range(0, len(payload), size):
        out.append(stream.feed(payload[start:start + size]))
    return out


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_answer_text_is_recovered_exactly_for_any_chunking(size):
    answer = 'He said "yes"\\no\nمرحبا 😀 العقد \u00e9'
    payload = json.dumps({"answer": answer, "insufficient_evidence": False, "citations": [1], "evidence": []})
    assert "".join(feed_all(payload, size)) == answer
    ascii_payload = json.dumps({"answer": answer}, ensure_ascii=True)
    assert "".join(feed_all(ascii_payload, size)) == answer


def test_nothing_is_emitted_before_the_answer_field_and_after_it_closes():
    stream = AnswerTextStream()
    assert stream.feed('{"insufficient_evidence": fal') == ""
    assert stream.feed('se, "answer": "ab') == "ab"
    assert stream.feed('c", "citations": [1], "extra": "zzz"}') == "c"


def parse_sse(body):
    events = []
    for frame in body.strip().split("\n\n"):
        name, data = frame.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


class StreamingGenerator:
    def __init__(self, answer, fail=False):
        self.answer, self.fail = GeneratedAnswer.model_validate(answer), fail

    def generate(self, question, context, language, prompt, on_text=None):
        if self.fail:
            raise GenerationError("Google generation failed (HTTP 429)")
        for word in self.answer.answer.split(" "):
            on_text(word + " ")
        return GenerationResponse(self.answer, "test-model", {"total_token_count": 5})


def stream_app(by_number, generator):
    retriever = FakeRetriever([by_number[147]])
    service = AnsweringService(retriever, generator, load_generation_config())
    return create_app(lambda: service)


def test_stream_sends_context_then_deltas_then_validated_final(by_number):
    answer = supported("The contract binds its parties under Article 147.", quote="العقد شريعة المتعاقدين")
    with TestClient(stream_app(by_number, StreamingGenerator(answer))) as client:
        response = client.post("/ask/stream", json={"question": "ما القوة الملزمة للعقد؟"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "context" and names[-1] == "final" and set(names[1:-1]) == {"delta"}
    assert events[0][1]["articles"][0]["article_number"] == 147
    assert "".join(data["text"] for name, data in events if name == "delta").strip() == answer["answer"]
    final = events[-1][1]
    assert final["sources"] == ["Egyptian Civil Code, Article 147"] and not final["insufficient_evidence"]


def test_ungrounded_stream_is_replaced_by_authoritative_final(by_number):
    bad = supported("It binds the parties under Article 147.", quote="not in the article")
    with TestClient(stream_app(by_number, StreamingGenerator(bad))) as client:
        events = parse_sse(client.post("/ask/stream", json={"question": "What binds the parties?"}).text)
    final = events[-1][1]
    assert any(name == "delta" for name, _ in events)
    assert final["insufficient_evidence"] and final["reason"] == "invalid_source_grounding" and final["sources"] == []


def test_provider_failure_is_reported_as_error_event(by_number):
    with TestClient(stream_app(by_number, StreamingGenerator(supported(), fail=True))) as client:
        events = parse_sse(client.post("/ask/stream", json={"question": "What binds the parties?"}).text)
    assert events[-1] == ("error", {"detail": "Google generation failed (HTTP 429)"})
    assert "final" not in [name for name, _ in events]


def test_stream_validates_requests_before_streaming():
    with TestClient(create_app(FakeService)) as client:
        assert client.post("/ask/stream", json={"question": "   "}).status_code == 422
