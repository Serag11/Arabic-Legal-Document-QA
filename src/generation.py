"""Grounded Google generation; secrets are loaded only when generation is needed."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock
import time

from pydantic import BaseModel, ConfigDict, Field, StrictInt

if __package__:
    from .metrics import get_metrics
    from .observability import get_tracer
    from .online_eval import FaithfulnessSampler
    from .streaming import AnswerTextStream
    from .retrieval import Retriever
else:
    from metrics import get_metrics
    from observability import get_tracer
    from online_eval import FaithfulnessSampler
    from streaming import AnswerTextStream
    from retrieval import Retriever


ROOT = Path(os.environ.get("LEGAL_QA_ROOT", Path(__file__).resolve().parent.parent)).resolve()
DEFAULT_GENERATION_CONFIG = ROOT / "config/generation.json"
PROMPT_PATH = ROOT / "prompts/answer_v1.txt"
ARABIC = re.compile(r"[\u0600-\u06ff]")
ANSWER_ARTICLE = re.compile(r"(?:\barticle\s*|المادة\s*|مادة\s*)[(:\s]*([0-9٠-٩۰-۹]+)", re.I)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_number: StrictInt
    quote: str = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    insufficient_evidence: bool
    citations: list[StrictInt]
    evidence: list[Evidence]


class GenerationError(RuntimeError):
    """A provider/response error with a safe message, never a raw request or key."""


@dataclass
class GenerationResponse:
    answer: GeneratedAnswer
    model_version: str | None = None
    usage: dict | None = None


def load_generation_config(path=DEFAULT_GENERATION_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["provider"] != "google" or config["api_key_env"] != "GOOGLE_API_KEY":
        raise ValueError("Baseline requires Google with GOOGLE_API_KEY")
    if config["retrieval_top_k"] < 1 or config["max_context_characters"] < 1:
        raise ValueError("Retrieval/context limits must be positive")
    return config


def load_google_api_key(env_file=None):
    from dotenv import dotenv_values

    values = dotenv_values(env_file if env_file is not None else ROOT / ".env")
    key = values.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not key.strip():
        raise GenerationError("Set GOOGLE_API_KEY in the project root .env or environment")
    return key.strip()


def load_fallback_models(env_file=None):
    """Optional comma-separated GOOGLE_FALLBACK_MODELS, tried in order when a model's quota runs out."""
    from dotenv import dotenv_values

    values = dotenv_values(env_file if env_file is not None else ROOT / ".env")
    names = values.get("GOOGLE_FALLBACK_MODELS") or os.environ.get("GOOGLE_FALLBACK_MODELS") or ""
    return [name.strip() for name in names.split(",") if name.strip()]


def quota_day():
    """Free-tier daily quotas reset at midnight Pacific time."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/Los_Angeles")).date()


class GoogleGenerator:
    def __init__(self, config, fallback_models=None):
        self.config = config
        fallbacks = load_fallback_models() if fallback_models is None else fallback_models
        self.models = list(dict.fromkeys([config["model"], *fallbacks]))
        self.exhausted = {}  # model -> quota day on which its daily quota (or access) ran out

    def generate(self, question, context, language, system_prompt, on_text=None):
        """One answer from the first model with quota left, falling back down the configured list.

        A model whose daily quota is exhausted (or that the key cannot use) is skipped until the quota
        resets; one that stays rate-limited after retries is skipped for this call only. Nothing falls
        back once streamed text has been shown.
        """
        emitted = False

        def track(text):
            nonlocal emitted
            emitted = True
            on_text(text)

        today = quota_day()
        candidates = [model for model in self.models if self.exhausted.get(model) != today] or self.models[:1]
        for index, model in enumerate(candidates):
            try:
                return self._generate_with_retries(model, question, context, language, system_prompt,
                                                   track if on_text else None, lambda: emitted)
            except GenerationError as failure:
                status = getattr(failure, "status_code", None)
                if getattr(failure, "daily_quota", False) or status == 404:
                    self.exhausted[model] = today
                if emitted or index == len(candidates) - 1 or not (status in (429, 503, 404)):
                    raise

    def _generate_with_retries(self, model, question, context, language, system_prompt, on_text, emitted):
        """Retry transient provider limits (HTTP 429 per-minute, 503) with a short backoff, never a daily-quota 429."""
        backoff = list(self.config.get("provider_retry_backoff_seconds", []))
        for attempt in range(len(backoff) + 1):
            try:
                return self._generate_once(question, context, language, system_prompt, on_text, model=model)
            except GenerationError as failure:
                transient = getattr(failure, "status_code", None) in (429, 503) and not getattr(failure, "daily_quota", False)
                if not transient or emitted() or attempt == len(backoff):
                    raise
                time.sleep(backoff[attempt])

    def _thinking_config(self, model, types):
        # Gemini 3 takes a thinking level, 2.5 a token budget.
        if model.startswith("gemini-3") and self.config.get("thinking_level"):
            return types.ThinkingConfig(thinking_level=self.config["thinking_level"])
        return types.ThinkingConfig(thinking_budget=self.config["thinking_budget"])

    def _generate_once(self, question, context, language, system_prompt, on_text=None, model=None):
        from google import genai
        from google.genai import errors, types
        from pydantic import ValidationError

        contents = json.dumps({"question": question, "answer_language": language, "passages": context}, ensure_ascii=False)
        try:
            with genai.Client(
                api_key=load_google_api_key(), vertexai=False,
                http_options=types.HttpOptions(timeout=self.config["timeout_ms"]),
            ) as client:
                request = dict(
                    model=model or self.config["model"], contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                        thinking_config=self._thinking_config(model or self.config["model"], types),
                        response_mime_type="application/json", response_json_schema=GeneratedAnswer.model_json_schema(),
                    ),
                )
                if on_text is None:
                    response = client.models.generate_content(**request)
                    if isinstance(response.parsed, GeneratedAnswer):
                        answer = response.parsed
                    else:
                        answer = GeneratedAnswer.model_validate_json(response.text or "")
                else:
                    # The structured JSON arrives in fragments; surface the growing "answer" text.
                    parser, fragments, response = AnswerTextStream(), [], None
                    for response in client.models.generate_content_stream(**request):
                        fragment = response.text or ""
                        fragments.append(fragment)
                        if delta := parser.feed(fragment):
                            on_text(delta)
                    answer = GeneratedAnswer.model_validate_json("".join(fragments))
        except errors.APIError as exc:
            failure = GenerationError(f"Google generation failed (HTTP {exc.code})")
            failure.status_code = exc.code
            failure.daily_quota = exc.code == 429 and "PerDay" in str(exc)
            raise failure from None
        except ValidationError:
            raise GenerationError("Google returned an invalid structured answer") from None
        except GenerationError:
            raise
        except Exception:
            raise GenerationError("Google generation failed; check connectivity and configuration") from None
        return GenerationResponse(
            answer=answer, model_version=getattr(response, "model_version", None),
            usage=response.usage_metadata.model_dump(exclude_none=True) if response.usage_metadata else {},
        )


def build_context(results, max_characters):
    """Include whole source passages only; never present normalized text as evidence."""
    context = []
    size = 0
    for result in results:
        for passage in result["passages"]:
            meta = passage["metadata"]
            record = {key: meta[key] for key in (
                "article_number", "citation", "language", "is_repealed", "repeal_note", "qa_flags", "content_kind",
            )}
            record["source_text"] = passage["source_text"]
            encoded_size = len(json.dumps(record, ensure_ascii=False))
            if size + encoded_size > max_characters:
                continue
            context.append(record)
            size += encoded_size
    return context


def validate_grounding(answer, context):
    """Verify citation membership and verbatim evidence, not semantic faithfulness."""
    if answer.insufficient_evidence:
        return []
    if not answer.answer.strip() or not answer.citations or not answer.evidence:
        raise ValueError("Supported answer lacks citations/evidence")
    cited = set(answer.citations)
    available = {p["article_number"] for p in context}
    if not cited <= available:
        raise ValueError("Answer cites an article outside the supplied context")
    inline = {int(m.group(1)) for m in ANSWER_ARTICLE.finditer(answer.answer)}
    if not inline <= cited:
        raise ValueError("Inline article citation differs from declared sources")
    verified = []
    for evidence in answer.evidence:
        if evidence.article_number not in cited or not evidence.quote.strip():
            raise ValueError("Evidence does not match declared sources")
        matches = [p for p in context if p["article_number"] == evidence.article_number and evidence.quote in p["source_text"]]
        if not matches:
            raise ValueError("Evidence is not an exact source excerpt")
        source = matches[0]
        verified.append({"citation": source["citation"], "quote": evidence.quote, "language": source["language"]})
    if {e.article_number for e in answer.evidence} != cited:
        raise ValueError("Every cited article needs supporting evidence")
    return verified


class AnsweringService:
    def __init__(self, retriever=None, generator=None, config=None, index=None, prompt_path=None, tracer=None, metrics=None, sampler=None):
        self.config = config or load_generation_config()
        self.retriever = retriever or Retriever(index or ROOT / "data/retrieval/chroma")
        self.generator = generator or GoogleGenerator(self.config)
        self.prompt = Path(prompt_path or PROMPT_PATH).read_text(encoding="utf-8")
        self.tracer = tracer or get_tracer()
        self.metrics = metrics or get_metrics()
        if self.metrics.drift and not self.metrics.drift.matches(self.retriever.manifest["config"]):
            self.metrics.drift = None  # baseline was built with another embedding model
        self.sampler = sampler if sampler is not None else FaithfulnessSampler.from_environment(self.metrics, self.tracer)
        # Only the shared CPU encoder/index needs serializing, not the provider call.
        self.retrieval_lock = Lock()

    def ask(self, question, *, language=None, filters=None, on_context=None, on_text=None):
        question = question.strip()
        if not question:
            raise ValueError("Question must be nonempty")
        language = language or ("ar" if ARABIC.search(question) else "en")
        if language not in ("ar", "en"):
            raise ValueError("Answer language must be ar or en")
        with self.tracer.observation("ask", input={"question": question, "language": language, "filters": filters or {}}) as root:
            started = time.perf_counter()
            try:
                result, context = self._ask(question, language, filters, on_context, on_text)
            except GenerationError as exc:
                self.metrics.observe_provider_error(exc)
                raise
            self.metrics.observe_stage("total", time.perf_counter() - started)
            self.metrics.observe_answer(result)
            root.update(output={key: result.get(key) for key in ("answer", "sources", "insufficient_evidence", "reason", "usage")},
                        metadata={"model_version": result.get("model_version")})
            if root.trace_id:
                result["trace_id"] = root.trace_id
            if self.sampler:
                self.sampler.maybe_submit(question, result, context)
            return result

    def _ask(self, question, language, filters, on_context, on_text):
        with self.tracer.observation("retrieval", as_type="retriever", input={"question": question, "filters": filters or {}}) as span:
            options = {"on_query_vector": self.metrics.observe_query_vector} if getattr(self.retriever, "supports_query_vector", False) else {}
            started = time.perf_counter()
            with self.retrieval_lock:
                # Article numbers repeat across indexed documents, so answers stay within one corpus.
                scoped = {**(filters or {}), "corpus_id": self.config["corpus_id"]}
                results = self.retriever.search(question, k=self.config["retrieval_top_k"], filters=scoped, **options)
            self.metrics.observe_stage("retrieval", time.perf_counter() - started)
            self.metrics.observe_results(results)
            context = build_context(results, self.config["max_context_characters"])
            span.update(output=[{"article_number": p["article_number"], "language": p["language"], "citation": p["citation"]}
                                for p in context])
        if on_context:
            on_context([{"article_number": p["article_number"], "citation": p["citation"], "language": p["language"]}
                        for p in context])
        run = {
            "generation": self.config, "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "retrieval": self.retriever.manifest["config"],
            "corpus_sha256": self.retriever.manifest["corpus_sha256"],
            "chunks_sha256": self.retriever.manifest["chunks_sha256"],
            "filters": filters or {}, "answer_language": language,
            "context_passage_count": len(context),
        }

        def insufficient(reason, usage=None, model_version=None):
            text = "لا تتوفر في النصوص المسترجعة أدلة كافية للإجابة عن هذا السؤال." if language == "ar" else "The retrieved passages do not provide sufficient evidence to answer this question."
            return ({"answer": text, "sources": [], "insufficient_evidence": True, "evidence": [],
                     "reason": reason, "configuration": run, "usage": usage or {}, "model_version": model_version}, context)

        if not context:
            return insufficient("no_retrieved_evidence")
        with self.tracer.observation(
            "generation", as_type="generation", model=self.config["model"],
            model_parameters={key: self.config[key] for key in ("temperature", "max_output_tokens") if key in self.config},
            input={"question": question, "answer_language": language, "passages": context},
            metadata={"prompt_version": self.config.get("prompt_version"), "prompt_sha256": run["prompt_sha256"]},
        ) as span:
            extra = {"on_text": on_text} if on_text else {}
            started = time.perf_counter()
            response = self.generator.generate(question, context, language, self.prompt, **extra)
            self.metrics.observe_stage("generation", time.perf_counter() - started)
            usage = {key: value for key, value in (response.usage or {}).items() if isinstance(value, int)}
            span.update(output=response.answer.model_dump(), usage_details=usage or None,
                        model=response.model_version or self.config["model"])
        if response.answer.insufficient_evidence:
            return insufficient("model_reported_insufficient_evidence", response.usage, response.model_version)
        with self.tracer.observation("grounding", as_type="guardrail", input=response.answer.model_dump()) as span:
            try:
                evidence = validate_grounding(response.answer, context)
                span.update(output={"valid": True})
            except ValueError as exc:
                # The discarded candidate and the failing check are kept in the trace for diagnosis.
                span.update(output={"valid": False, "error": str(exc)}, level="WARNING", status_message=str(exc))
                rejected, _ = insufficient("invalid_source_grounding", response.usage, response.model_version)
                rejected["reason_detail"] = str(exc)
                return rejected, context
        citations = list(dict.fromkeys(response.answer.citations))
        source_map = {p["article_number"]: p["citation"] for p in context}
        answer_text = response.answer.answer
        source_notes = []
        if language == "ar":
            missing = sorted({p["article_number"] for p in context if p["article_number"] in citations
                              and "missing_ar_text_in_source" in p["qa_flags"]})
            for number in missing:
                note = f"ملاحظة: النص العربي للمادة {number} غير موجود في نسخة المصدر؛ استندت الإجابة إلى النص الإنجليزي."
                source_notes.append(note)
                answer_text += "\n\n" + note
        return ({"answer": answer_text, "sources": [source_map[n] for n in citations],
                 "insufficient_evidence": False, "evidence": evidence,
                 "source_notes": source_notes,
                 "configuration": run, "usage": response.usage or {}, "model_version": response.model_version}, context)
