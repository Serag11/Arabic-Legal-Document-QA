# Arabic-Legal-Document-QA

Article-aware Arabic/English retrieval and cited answers from the bilingual
Egyptian Civil Code corpus. E5 embeddings and Chroma run locally; Google's
`gemini-3.1-flash-lite` generates answers from retrieved source passages.

## Quickstart

Install [uv](https://docs.astral.sh/uv/), make the DVC-tracked corpus available,
and set `GOOGLE_API_KEY` in a `.env` file in the project root (see `.env.example`).
The Drive remote upload is still pending; corpus availability is a prerequisite
on a new checkout. Then run:

```bash
uv sync --group evaluation
uv run python src/retrieval.py download && uv run dvc repro
docker compose up --build -d
```

The second command explicitly downloads pinned E5 weights once, then builds and
evaluates the local index. The third command starts FastAPI on port 8000 with
the local index and Hugging Face cache mounted read-only. `/ask` sends the
question plus retrieved source passages to Google's API. Generation calls use
the account's quota/billing; no local generation weights are downloaded.

Answers return `answer`, `sources` (article citations), `insufficient_evidence`,
exact source `evidence`, and reproducibility/usage metadata. Arabic questions
default to Arabic answers; `--language ar` or `--language en` overrides this.
Unsupported questions return a localized insufficient-evidence response.
Provider failures return a nonzero CLI exit status rather than an invented answer.

## API and architecture

`POST /ask` runs article-aware retrieval → whole source context → Google
structured generation → exact citation/quotation validation. One retriever is
loaded at application startup. `GET /health` reports indexed document/article
counts; API documentation is available at `http://localhost:8000/docs`.

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"ما تعريف الإيجار وفق المادة 558؟"}'
```

Empty/invalid questions return HTTP 422; provider failures return HTTP 502.
The `/docs` request selector contains usable Arabic, English and repeal-lookup
examples. For a normal question, send only `question` and optionally the answer
`language`; omit `filters`. Filters are exact matches combined with AND, so
placeholder values such as `"book": "string"` will not match the corpus.
Article-number filters must be positive integers. The top-level `language`
controls the answer language; `filters.language` restricts source passages.

```json
{"question":"ما تعريف عقد الإيجار؟","language":"ar"}
```

The baseline serializes CPU inference/generation to avoid duplicate model loads.
Container startup needs the prepared index and cached model; it does not fetch
model weights. A deployed wheel uses `LEGAL_QA_ROOT` to locate external
configuration, prompts, index, and the optional `.env` file.

The encoder is initialized during startup so cached-model errors are detected
before readiness. Only the Hugging Face `hub` model-cache directory is mounted;
personal Hugging Face token files are not needed for this public cached model.

Retrieval supports exact article-number lookup and metadata filters:

```bash
uv run python src/retrieval.py query "Article 1022"
uv run python src/rag.py ask "What is a lease?" --filters '{"language":"en","is_repealed":false}'
```

## Tests and reports

- Offline tests: `uv run python -m pytest tests/ -q` (no live Google calls).
- Explicit live smoke run: `uv run python src/evaluate_generation.py`.
- Retrieval baseline: `reports/retrieval-baseline.md`.
- Generation baseline: `reports/generation-baseline.md`.
- Configuration: `config/retrieval.json`, `config/generation.json`.
- Source exceptions, reproduction details and tests: `tests/README.md`.

## Experiments and measured quality gate

The answer evaluation contains 60 language-specific questions from the 30
user-reviewed pairs, with reference source answers/contexts and article labels.
Five retrieval configurations are compared on the same fixed 24-question
subset. The selected configuration is validated on all 60 questions; unchanged
benchmark results are reused. Generation and RAGAS judging use Google quota.

```bash
uv run --group evaluation python -m src.experiments prepare
uv run --group evaluation python -m src.experiments run
uv run --group evaluation python -m src.experiments register
python -m src.quality_gate
uv run --group evaluation mlflow ui --backend-store-uri sqlite:///data/mlflow/tracking.db
```

Tracking/model artifacts remain local under ignored `data/mlflow/` and `mlruns/`.
Measured reports are kept under `reports/experiments/`. The registered model
packages the selected configuration and index, not Google's weights or API key.

The measured champion is **top-k 5**, registered as
`models:/ArabicLegalDocumentQA@champion` (version 2). Full validation found the
expected article in 55/56 supported questions, answered with expected citations
in 51/56, and correctly refused all four unsupported probes. It produced 52/56
supported answers. Raw faithfulness was 0.9962 and coverage-adjusted faithfulness
0.9250. Details and limitations are in `reports/answer-evaluation.md`. The
running API also uses top-k 5; registry registration itself does not deploy code.

When loading the registered artifact, set `LEGAL_QA_ROOT` to the project/resource
root (or supply the key through the environment) so credentials remain external
to the model artifact.

The CI gate requires at least 20 scored supported questions, complete coverage
of the fixed dataset, no operational errors, and coverage-adjusted faithfulness
of at least **0.75**. It rejects stale scores when code, prompts, configuration,
dependencies, or the dataset change. Abstentions contribute zero to the adjusted
score, while unsupported probes are assessed separately through refusal accuracy.
The offline gate never makes API calls; live evaluation is an explicit command.

RAGAS judges are automated estimates, not legal review. Claim extraction preserves
legal alternatives/conditions and is versioned; generator and judge use the same
model family. Reference-answer judging is optional (`score_answer_accuracy`),
disabled in the initial quota-conscious run. Exact quotation checks establish
source traceability independently of those scores.
