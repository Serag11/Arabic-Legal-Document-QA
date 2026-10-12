"""FastAPI service with one shared retriever per application lifetime."""

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from .generation import AnsweringService, GenerationError
from .retrieval import E5Encoder
from .streaming import stream_events


ASK_EXAMPLES = {
    "arabic_question": {
        "summary": "Arabic legal question (no filters needed)",
        "value": {"question": "ما تعريف عقد الإيجار؟", "language": "ar"},
    },
    "english_question": {
        "summary": "English question with English-source filtering",
        "value": {"question": "What is a lease?", "language": "en", "filters": {"language": "en", "is_repealed": False}},
    },
    "repeal_lookup": {
        "summary": "Look up a repealed article (English notice, Arabic answer)",
        "value": {"question": "ما حالة المادة 54 في المصدر؟", "language": "ar", "filters": {"article_number": 54, "is_repealed": True}},
    },
}


class MetadataFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Literal["ar", "en"] | None = None
    article_number: StrictInt | None = Field(default=None, ge=1, examples=[147])
    is_repealed: StrictBool | None = None
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    source_book: str | None = None
    content_kind: Literal["article_text", "repeal_notice"] | None = None


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [ASK_EXAMPLES["arabic_question"]["value"]]})
    question: str = Field(min_length=1, description="A question about the supplied Egyptian Civil Code corpus.")
    language: Literal["ar", "en"] | None = None
    filters: MetadataFilters = Field(default_factory=MetadataFilters, description="Optional exact-match constraints combined with AND. Omit for normal questions.")

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Question must be nonempty")
        return value


class SourceEvidence(BaseModel):
    citation: str
    quote: str
    language: Literal["ar", "en"]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    insufficient_evidence: bool
    evidence: list[SourceEvidence]
    source_notes: list[str] = Field(default_factory=list)
    reason: str | None = None
    configuration: dict
    usage: dict
    model_version: str | None = None
    trace_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    indexed_documents: int
    indexed_articles: int


def create_app(service_factory=None):
    factory = service_factory or AnsweringService

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = factory()
        if service_factory is None:
            # Exercise cached embedding loading before declaring the API ready;
            # exact-number lookups alone do not validate semantic retrieval.
            retriever = app.state.service.retriever
            retriever.encoder = E5Encoder(retriever.config)
        try:
            yield
        finally:
            getattr(app.state.service, "tracer", None) and app.state.service.tracer.flush()
            app.state.service = None

    application = FastAPI(title="Arabic Legal Document QA", version="0.1.0", lifespan=lifespan)

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        manifest = request.app.state.service.retriever.manifest
        return {"status": "ok", "indexed_documents": manifest["indexed_count"],
                "indexed_articles": manifest["article_count"]}

    @application.post("/ask", response_model=AskResponse)
    def ask(body: Annotated[AskRequest, Body(openapi_examples=ASK_EXAMPLES)], request: Request):
        try:
            return request.app.state.service.ask(
                body.question, language=body.language,
                filters=body.filters.model_dump(exclude_none=True),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None

    @application.post(
        "/ask/stream",
        response_class=StreamingResponse,
        responses={200: {"description": "Server-sent events: `context` (retrieved citations), `delta` (provisional "
                         "answer text), then `final` (the validated AskResponse, authoritative) or `error`.",
                         "content": {"text/event-stream": {}}}},
    )
    async def ask_stream(body: Annotated[AskRequest, Body(openapi_examples=ASK_EXAMPLES)], request: Request):
        service, filters = request.app.state.service, body.filters.model_dump(exclude_none=True)

        def work(emit):
            try:
                result = service.ask(
                    body.question, language=body.language, filters=filters,
                    on_context=lambda articles: emit("context", {"articles": articles}),
                    on_text=lambda text: emit("delta", {"text": text}),
                )
            except ValueError as exc:
                raise RuntimeError(f"Invalid request: {exc}") from None
            except GenerationError as exc:
                raise RuntimeError(str(exc)) from None
            emit("final", AskResponse.model_validate(result).model_dump(mode="json"))

        return StreamingResponse(stream_events(work), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return application


app = create_app()
