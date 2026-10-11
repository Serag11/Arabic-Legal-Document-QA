# Arabic-Legal-Document-QA

Article-aware Arabic/English retrieval and cited answers from the bilingual
Egyptian Civil Code corpus. E5 embeddings and Chroma run locally; Google's
`gemini-3.1-flash-lite` generates answers from retrieved source passages.

## Quickstart

Install [uv](https://docs.astral.sh/uv/) and set `GOOGLE_API_KEY` in a `.env` file
in the project root (see `.env.example`). The DVC-tracked artifacts have been
pushed to the Google Drive remote `storage`. From the project root, run:

```bash
uv sync --group evaluation
uv run dvc pull -r storage && uv run python src/retrieval.py download && uv run dvc repro
docker compose up --build -d
```

The second command pulls the tracked artifacts from `storage` (complete Google
Drive authentication if prompted), explicitly downloads pinned E5 weights once,
then reproduces any out-of-date pipeline stages. To pull artifacts again on an
existing setup, run `uv run dvc pull -r storage`.
The third command starts FastAPI on port 8000 with
the local index and Hugging Face cache mounted read-only. `/ask` sends the
question plus retrieved source passages to Google's API. Generation calls use
the account's quota/billing; no local generation weights are downloaded.

Answers return `answer`, `sources` (article citations), `insufficient_evidence`,
exact source `evidence`, and reproducibility/usage metadata. Arabic questions
default to Arabic answers; `--language ar` or `--language en` overrides this.
Unsupported questions return a localized insufficient-evidence response.
Provider failures return a nonzero CLI exit status rather than an invented answer.

## Self-contained published image

The Stage 6 delivery target is the public Docker Hub repository
`serag11/arabic-legal-document-qa`. Once the first release/image is published,
set `GOOGLE_API_KEY` in your shell and run:

```bash
docker pull serag11/arabic-legal-document-qa:latest
docker run -d --name legal-qa -p 8000:8000 -e GOOGLE_API_KEY serag11/arabic-legal-document-qa:latest
```

This image includes the tokenizer, compact ONNX E5 weights, and Chroma index.
It needs no repository checkout, DVC credentials, model download, or host data
mounts. Generation still uses Google's API. `/health` and `/docs` are available
at `http://localhost:8000`. For a reproducible deployment, use the published
`sha-<full-git-commit>` tag or image digest instead of `latest`.

Weights use **FP16 storage with FP32 CPU computation**, reducing the ONNX model
from 1.11 GB to 555 MB. The INT8 experiment was smaller but lost two expected
Arabic retrieval hits, so it was rejected. The selected compact model preserves
all 60 evaluated generator contexts exactly; existing generation scores are
reused on that basis, rather than represented as a new RAGAS run. The actual
PyTorch-free image also passed this full offline comparison.

The development quickstart above retains the original PyTorch/Compose path.
See [`reports/stage6-delivery.md`](reports/stage6-delivery.md) for the artifact
release bootstrap, build commands, verification results, and publication status.

## CI and artifact delivery

`.github/workflows/delivery.yml` runs lint, offline tests, the measured quality
gate, the compact-runtime equivalence gate, Docker builds, and offline container
checks. Successful `main` builds publish `latest` and a commit-specific tag to
Docker Hub. Pull requests build/test without publishing or using Docker secrets.

The workflow needs repository Actions secrets `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN`. It downloads a pinned GitHub Release bundle and verifies its
archive and per-file checksums using `config/runtime-artifacts.json`. Google
Drive/DVC authentication and `GOOGLE_API_KEY` are not needed in CI. DVC remains
the local corpus/index versioning and reproduction mechanism.
Private GitHub release assets use the workflow's built-in GitHub token with
`contents: read`; no additional GitHub token secret is needed.

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
