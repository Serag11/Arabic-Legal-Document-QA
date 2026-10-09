"""Service lifecycle, validation, cited responses, and safe error handling."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from src.api import create_app
from src.generation import GenerationError


class FakeService:
    def __init__(self):
        self.retriever = SimpleNamespace(manifest={"indexed_count": 3, "article_count": 2})
        self.calls = []

    def ask(self, question, **kwargs):
        self.calls.append((question, kwargs))
        if question == "provider failure":
            raise GenerationError("Google generation failed (HTTP 429)")
        return {"answer": "The contract binds its parties.",
                "sources": ["Egyptian Civil Code, Article 147"],
                "insufficient_evidence": False,
                "evidence": [{"citation": "Egyptian Civil Code, Article 147", "quote": "The contract makes the law of the parties.", "language": "en"}],
                "configuration": {}, "usage": {}, "model_version": "mock"}


def test_startup_loads_once_and_health_reports_index_counts():
    instances = []

    def factory():
        service = FakeService()
        instances.append(service)
        return service

    app = create_app(factory)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "indexed_documents": 3, "indexed_articles": 2}
        for _ in range(2):
            response = client.post("/ask", json={"question": "  contract  ", "filters": {"language": "en"}})
            assert response.status_code == 200
            assert response.json()["sources"] == ["Egyptian Civil Code, Article 147"]
            assert response.json()["evidence"][0]["language"] == "en"
        assert len(instances) == 1 and len(instances[0].calls) == 2
        assert instances[0].calls[0] == ("contract", {"language": None, "filters": {"language": "en"}})
    assert app.state.service is None


@pytest.mark.parametrize("body", [
    {"question": ""}, {"question": "  \n"}, {}, {"question": 147},
    {"question": "contract", "language": "fr"},
    {"question": "contract", "filters": {"article_number": "147"}},
    {"question": "contract", "filters": {"is_repealed": "false"}},
    {"question": "contract", "filters": {"unknown": "value"}},
])
def test_invalid_requests_fail_with_422_before_generation(body):
    service = FakeService()
    with TestClient(create_app(lambda: service)) as client:
        assert client.post("/ask", json=body).status_code == 422
        assert not service.calls


def test_provider_failure_returns_bad_gateway():
    with TestClient(create_app(FakeService)) as client:
        response = client.post("/ask", json={"question": "provider failure"})
        assert response.status_code == 502
        assert response.json() == {"detail": "Google generation failed (HTTP 429)"}
