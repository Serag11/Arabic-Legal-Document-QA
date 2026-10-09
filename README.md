# Arabic-Legal-Document-QA

Article-aware Arabic/English retrieval and cited answers from the bilingual
Egyptian Civil Code corpus. E5 embeddings and Chroma run locally; Google's
`gemini-2.5-flash` generates answers from retrieved source passages.

## Quickstart

Install [uv](https://docs.astral.sh/uv/), make the DVC-tracked corpus available,
and set `GOOGLE_API_KEY` in a `.env` file in the project root (see `.env.example`).
The Drive remote upload is still pending; corpus availability is a prerequisite
on a new checkout. Then run:

```bash
uv sync
uv run python src/retrieval.py download && uv run dvc repro
uv run python src/rag.py ask "وفق المادة 147، هل يجوز لأحد المتعاقدين نقض العقد وحده؟"
```

The second command explicitly downloads pinned E5 weights once, then builds and
evaluates the local index. `ask` uses the cached embedding model and sends the
question plus retrieved source passages to Google's API. Generation calls use
the account's quota/billing; no local generation weights are downloaded.

Answers return `answer`, `sources` (article citations), `insufficient_evidence`,
exact source `evidence`, and reproducibility/usage metadata. Arabic questions
default to Arabic answers; `--language ar` or `--language en` overrides this.
Unsupported questions return a localized insufficient-evidence response.
Provider failures return a nonzero CLI exit status rather than an invented answer.

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

Exact quotation/citation checks establish source traceability; broader answer
faithfulness evaluation is a later stage. FastAPI/service packaging is next.
