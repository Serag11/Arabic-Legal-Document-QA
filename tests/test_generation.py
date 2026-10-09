"""End-to-end orchestration and grounding checks without live Google calls."""

from dataclasses import asdict

import pytest

from src.generation import (
    AnsweringService, GeneratedAnswer, GenerationError, GenerationResponse,
    GoogleGenerator, load_generation_config, load_google_api_key,
)
from src.rag import prepare_documents
from src.retrieval import load_config


class FakeRetriever:
    def __init__(self, articles):
        self.documents = prepare_documents(articles)
        self.manifest = {"config": load_config(), "corpus_sha256": "test-corpus", "chunks_sha256": "test-chunks"}
        self.calls = []

    def search(self, question, k, filters):
        self.calls.append((question, k, filters))
        return [{"passages": [asdict(doc)]} for doc in self.documents]


class FakeGenerator:
    def __init__(self, answer):
        self.answer = GeneratedAnswer.model_validate(answer)
        self.calls = []

    def generate(self, question, context, language, prompt):
        self.calls.append((question, context, language, prompt))
        return GenerationResponse(self.answer, "test-model", {"total_token_count": 10})


def supported(answer="العقد شريعة المتعاقدين وفق المادة 147.", article=147, quote="العقد شريعة المتعاقدين"):
    return {"answer": answer, "insufficient_evidence": False, "citations": [article],
            "evidence": [{"article_number": article, "quote": quote}]}


def service(by_number, answer, numbers=(147,), **config):
    generator = FakeGenerator(answer)
    retriever = FakeRetriever([by_number[n] for n in numbers])
    return AnsweringService(retriever, generator, dict(load_generation_config(), **config)), generator, retriever


def test_supported_answer_returns_citations_exact_source_and_run_config(by_number):
    app, generator, retriever = service(by_number, supported())
    result = app.ask("ما القوة الملزمة للعقد؟")
    assert not result["insufficient_evidence"]
    assert result["sources"] == ["Egyptian Civil Code, Article 147"]
    assert result["evidence"] == [{"citation": result["sources"][0], "quote": "العقد شريعة المتعاقدين", "language": "ar"}]
    context = generator.calls[0][1]
    assert context[0]["source_text"] == by_number[147]["ar_text"]
    assert "embedding_text" not in context[0]
    assert generator.calls[0][2] == "ar"
    assert result["configuration"]["retrieval"]["model_id"] == "intfloat/multilingual-e5-base"
    assert result["configuration"]["generation"]["retrieval_top_k"] == retriever.calls[0][1]
    assert result["configuration"]["prompt_sha256"]


@pytest.mark.parametrize("answer", [
    supported(article=999),
    supported(quote="fabricated source evidence"),
    supported(answer="Article 999 permits revocation."),
    dict(supported(), citations=[]),
    dict(supported(), evidence=[]),
])
def test_unverifiable_generation_is_not_returned_as_a_supported_answer(by_number, answer):
    app, _, _ = service(by_number, answer)
    result = app.ask("What is the binding force of a contract?")
    assert result["insufficient_evidence"]
    assert result["sources"] == result["evidence"] == []
    assert result["reason"] == "invalid_source_grounding"


def test_unsupported_question_discards_any_provider_sources(by_number):
    answer = dict(supported(), insufficient_evidence=True)
    app, _, _ = service(by_number, answer)
    result = app.ask("What is the capital of Japan?")
    assert result["insufficient_evidence"] and not result["sources"]
    assert "sufficient evidence" in result["answer"]


@pytest.mark.parametrize("question", ["", "  \n\t"])
def test_empty_question_is_rejected_before_retrieval_or_api(by_number, question):
    app, generator, retriever = service(by_number, supported())
    with pytest.raises(ValueError, match="nonempty"):
        app.ask(question)
    assert not retriever.calls and not generator.calls


def test_no_context_skips_google_and_returns_localized_insufficient_evidence(by_number):
    app, generator, _ = service(by_number, supported(), numbers=())
    result = app.ask("ما أثر العقد؟")
    assert result["insufficient_evidence"] and result["answer"].startswith("لا تتوفر")
    assert not generator.calls


def test_context_budget_does_not_cut_passages(by_number):
    app, generator, _ = service(by_number, supported(), max_context_characters=1)
    assert app.ask("contract")["reason"] == "no_retrieved_evidence"
    assert not generator.calls


def test_repeal_notice_and_english_only_source_remain_visible(by_number):
    for number, question in [(54, "Article 54"), (1022, "المادة ١٠٢٢")]:
        field = "repeal_note" if number == 54 else "text_en"
        app, generator, _ = service(by_number, supported("See the cited article.", number, by_number[number][field]), numbers=(number,))
        result = app.ask(question)
        assert not result["insufficient_evidence"]
        assert result["evidence"][0]["language"] == "en"
        meta = generator.calls[0][1][0]
        assert meta["is_repealed"] == (number == 54)
        if number == 1022:
            assert "missing_ar_text_in_source" in meta["qa_flags"]
            assert "النص العربي للمادة 1022 غير موجود" in result["answer"]
            assert result["source_notes"]


def test_dotenv_loading_uses_file_without_exposing_other_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=test-file-key\nUNRELATED_SECRET=unrelated\n")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-environment-key")
    assert load_google_api_key(env) == "test-file-key"
    env.write_text("UNRELATED_SECRET=unrelated\n")
    assert load_google_api_key(env) == "test-environment-key"
    monkeypatch.delenv("GOOGLE_API_KEY")
    with pytest.raises(GenerationError, match="Set GOOGLE_API_KEY"):
        load_google_api_key(env)


def test_google_api_errors_are_redacted(monkeypatch):
    from google import genai
    from google.genai import errors

    monkeypatch.setattr("src.generation.load_google_api_key", lambda: "test-secret")

    class FailingClient:
        def __init__(self, **kwargs):
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate_content(self, **kwargs):
            # Use the native JSON-schema API: legacy response_schema rejects
            # Pydantic's additionalProperties fields on the live Gemini endpoint.
            assert kwargs["config"].response_schema is None
            assert kwargs["config"].response_json_schema["additionalProperties"] is False
            raise errors.ClientError(429, {"error": {"message": "test-secret quota failure", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(genai, "Client", FailingClient)
    with pytest.raises(GenerationError) as caught:
        GoogleGenerator(load_generation_config()).generate("question", [], "en", "prompt")
    assert "429" in str(caught.value) and "test-secret" not in str(caught.value)
