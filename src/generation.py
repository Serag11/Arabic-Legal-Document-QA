"""Grounded Google generation; secrets are loaded only when generation is needed."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, StrictInt

if __package__:
    from .retrieval import Retriever
else:
    from retrieval import Retriever


ROOT = Path(__file__).resolve().parent.parent
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


class GoogleGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, question, context, language, system_prompt):
        from google import genai
        from google.genai import errors, types
        from pydantic import ValidationError

        contents = json.dumps({"question": question, "answer_language": language, "passages": context}, ensure_ascii=False)
        try:
            with genai.Client(
                api_key=load_google_api_key(), vertexai=False,
                http_options=types.HttpOptions(timeout=self.config["timeout_ms"]),
            ) as client:
                response = client.models.generate_content(
                    model=self.config["model"], contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_output_tokens"],
                        thinking_config=types.ThinkingConfig(thinking_budget=self.config["thinking_budget"]),
                        response_mime_type="application/json", response_json_schema=GeneratedAnswer.model_json_schema(),
                    ),
                )
            if isinstance(response.parsed, GeneratedAnswer):
                answer = response.parsed
            else:
                answer = GeneratedAnswer.model_validate_json(response.text or "")
        except errors.APIError as exc:
            raise GenerationError(f"Google generation failed (HTTP {exc.code})") from None
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
    def __init__(self, retriever=None, generator=None, config=None, index=None):
        self.config = config or load_generation_config()
        self.retriever = retriever or Retriever(index or ROOT / "data/retrieval/chroma")
        self.generator = generator or GoogleGenerator(self.config)
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def ask(self, question, *, language=None, filters=None):
        question = question.strip()
        if not question:
            raise ValueError("Question must be nonempty")
        language = language or ("ar" if ARABIC.search(question) else "en")
        if language not in ("ar", "en"):
            raise ValueError("Answer language must be ar or en")
        results = self.retriever.search(question, k=self.config["retrieval_top_k"], filters=filters)
        context = build_context(results, self.config["max_context_characters"])
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
            return {"answer": text, "sources": [], "insufficient_evidence": True, "evidence": [],
                    "reason": reason, "configuration": run, "usage": usage or {}, "model_version": model_version}

        if not context:
            return insufficient("no_retrieved_evidence")
        response = self.generator.generate(question, context, language, self.prompt)
        if response.answer.insufficient_evidence:
            return insufficient("model_reported_insufficient_evidence", response.usage, response.model_version)
        try:
            evidence = validate_grounding(response.answer, context)
        except ValueError:
            return insufficient("invalid_source_grounding", response.usage, response.model_version)
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
        return {"answer": answer_text, "sources": [source_map[n] for n in citations],
                "insufficient_evidence": False, "evidence": evidence,
                "source_notes": source_notes,
                "configuration": run, "usage": response.usage or {}, "model_version": response.model_version}
