# Stage 4 — API, installation, and container packaging

`src/api.py` exposes `POST /ask` and `GET /health`. Lifespan startup constructs
one `AnsweringService` and shared retriever; a lock protects the baseline's
CPU encoder initialization/inference. Requests do not reload the index.

Pydantic validates questions, optional answer language, and typed metadata
filters. Empty/whitespace-only questions and malformed filters return HTTP 422.
Responses include answer text, article citations, exact evidence, source-language
notes, insufficient-evidence status, configuration, and token usage. Sanitized
Google failures return HTTP 502 rather than a purported legal answer.

The project is installable with setuptools; `legal-qa` is the CLI entry point.
Configuration, prompts, index and `.env` are external application resources;
`LEGAL_QA_ROOT` specifies their deployment root. The Docker image places these
resources under `/app`, runs as a non-root user, and uses CPU-only PyTorch.

`compose.yaml` supplies the key through `.env` at runtime. The local Chroma
index and existing Hugging Face cache are mounted read-only. Retrieval queries
a temporary runtime copy of the index. No key, corpus/index, cached model
weights, or local MLflow store is included in the build context. Container
startup is offline for embedding models. The API health check uses `/health`.

Offline API tests cover startup loading once, health counts, cited responses,
request validation, and provider failure. The full offline suite currently
passes 64 tests. Docker build and Compose startup succeeded. The running
container is healthy on port 8000. `/health` returned 2,242 indexed documents
and 1,149 articles; a blank `/ask` question returned 422; an Arabic Article 54
query returned HTTP 200 and the correct article citation with a repeal answer.
The live service uses the user-selected `gemini-3.1-flash-lite` model. A wheel
was built successfully at `/tmp/opencode/arabic_legal_document_qa-0.1.0-py3-none-any.whl`.
