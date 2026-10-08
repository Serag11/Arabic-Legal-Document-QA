# Arabic Legal Document Q&A — Project Plan

This plan follows the Project 2 requirements in *The MLOps Practitioner Handbook* (pp. 64–69). The immediate goal is to make the Egyptian Civil Code corpus reliable before building retrieval and generation on top of it.

## Current status

- [X] Column-aware bilingual PDF extraction in `src/extract_egyptian_civil_code.py`.
- [X] Structured JSONL corpus with 1,149 consecutive article records.
- [X] Arabic source text and separate search-normalized Arabic text.
- [X] Repealed article records retained and flagged.
- [X] DVC extraction stage in `dvc.yaml`; current `dvc status` is clean.
- [X] Random 20-article visual QA HTML/crops generated in `data/qa_civil_code/`.
- [X] Article 558 bold Arabic body preserved; Article 1022's absent Arabic text documented as a source exception.
- [X] Glyph-position-based reordering for zero-width alef variants applied during extraction and covered by regression tests.
- [X] Add automated corpus validation and run it as a DVC stage.
- [ ] Manually review the 20 sampled article crops and the remaining flagged records; record findings.
- [ ] Configure a DVC remote and make the full corpus pipeline reproducible for others.
- [ ] Build article-aware retrieval, then generation and the `/ask` API.

## Phase 1 — Make the corpus trustworthy (next)

### 1. Investigate extraction defects

- [X] Compare Articles 558 and 1022 with the original PDF: Article 558's Arabic body is bold; Article 1022 has no Arabic body in the source.
- [X] Check the Article 1021–1022 boundary against the source; preserve Article 1022 as an explicit missing-Arabic source exception.
- [ ] Review every record with a `qa_flags` entry, prioritizing single-column article anchors and missing text.
- [ ] Inspect `data/qa_civil_code/qa_random_articles.html` and compare all 20 sampled records with their PDF crops.
- [ ] Record extraction issues, fixes, and the manual sample-review outcome in `reports/corpus-extraction.md`.

### 2. Add automated validation

- [X] Create tests for consecutive, unique integer article numbers and the expected corpus range.
- [X] Assert that active articles have Arabic and English text, with the documented Article 1022 source exception and repealed-article handling.
- [X] Assert repealed records are retained, flagged, and carry repeal metadata.
- [X] Check article length bounds, citations, and metadata.
- [X] Add regression checks for Articles 1–3, 147, 558, and 1022.
- [ ] Add a specific regression assertion for Article 1021's extracted text and its boundary with Article 1022.
- [X] Run validation after extraction as a DVC stage so extraction cannot silently produce an invalid corpus.

### 3. Standardize and version the corpus

- [X] Use `ar_text` as the canonical source-text field in the extractor and corpus tests; carry it forward into ingestion and the API.
- [X] Keep source-faithful Arabic separate from normalized retrieval text; do not use normalized text as a legal quotation.
- [ ] Configure a DVC remote, then verify `dvc push` and `dvc pull` from a clean checkout.
- [X] Verify `dvc repro` regenerates the structured corpus and QA artifacts from the tracked source PDF.

**Phase 1 exit criteria:** corpus tests pass; active article text is present in both languages except the documented source exception; flagged records and the 20-item visual sample have been reviewed; DVC can reproduce the extraction and a remote is available to collaborators.

## Phase 2 — Build article-aware retrieval

- [ ] Create an ingestion module that reads validated JSONL and preserves article number, citation, book, chapter, section, topic, language, and repeal status as metadata.
- [ ] Begin with one chunk per article. For very long articles, split only at paragraph boundaries and repeat the article citation/number on every chunk.
- [ ] Select a multilingual embedding model and document its version/configuration.
- [ ] Build a persistent vector index from the structured corpus; make index creation reproducible rather than committing an opaque, untracked index.
- [ ] Add a retrieval-only path and try a small set of Arabic and English questions with known expected article citations.
- [ ] Test bilingual retrieval: equivalent Arabic and English questions should retrieve the same relevant article where applicable.
- [ ] Add filters for hierarchy and repeal metadata where useful.

**Phase 2 exit criteria:** a reproducible index exists, and a small hand-checked retrieval set returns the expected article citations in both Arabic and English.

## Phase 3 — Create the first end-to-end RAG baseline

- [ ] Add a generator that answers only from retrieved article context.
- [ ] Define a clear insufficient-evidence response; do not guess when the retrieved passages do not support an answer.
- [ ] Ensure generated answers return article citations, not chunk IDs or PDF page numbers.
- [ ] Create `src/rag.py` with separate ingestion/retrieval/query responsibilities.
- [ ] Add tests for empty questions, retrieval citations, unsupported questions, and source formatting.
- [ ] Track configuration such as chunking, embedding model, retrieval top-k, and prompt version for every baseline run.

**Example API contract:**

```json
POST /ask
{"question": "ما أثر العقد بين المتعاقدين؟"}
```

```json
{
  "answer": "...",
  "sources": ["Egyptian Civil Code, Article 147"]
}
```

## Phase 4 — Package and expose the service

- [ ] Create a `src/` package and `pyproject.toml` configuration suitable for installing the project.
- [ ] Implement a FastAPI `/ask` endpoint with Pydantic request/response schemas; reject empty questions with HTTP 422.
- [ ] Implement `/health` reporting service health and the number of indexed documents/articles.
- [ ] Load the retriever/index once during application startup, not once per request.
- [ ] Add API tests that do not require a live external LLM call (mock the generator/retriever boundary).
- [ ] Add a Dockerfile and Compose configuration so a reviewer can run the service locally.
- [ ] Update `README.md` with a short project description, architecture, example request, and exactly three quickstart commands.

## Phase 5 — Evaluate and track experiments

- [ ] Build a fixed, manually checked evaluation set of at least 20 questions with reference answers/contexts and expected article citations.
- [ ] Include Arabic and English questions, paraphrases, cross-language retrieval checks, and questions that should be refused as unsupported.
- [ ] Measure retrieval quality and RAG answer quality; log the metric definitions and baseline results.
- [ ] Expand the evaluation set to at least 50 questions for the final RAGAS/monitoring deliverables.
- [ ] Run at least five MLflow experiments comparing chunking/retrieval configurations; log `chunk_size`, `overlap`, `embedding_model`, retrieval settings, and faithfulness.
- [ ] Register the best configuration/model artifact in MLflow after selecting it based on measured results.
- [ ] Add a CI quality gate that fails if faithfulness drops below the agreed threshold (the handbook specifies 0.75 for the 20-question gate).

## Phase 6 — Add CI and delivery automation

- [ ] Add GitHub Actions for lint, tests, evaluation/quality gate, Docker build, and image publication.
- [ ] Store credentials in GitHub Actions secrets; never commit API keys or `.env` files.
- [ ] Track the source PDF and derived corpus with DVC; verify clean-checkout reproduction using `dvc pull` and `dvc repro`.
- [ ] Publish the container image and document how to run it without cloning the repository.

## Phase 7 — Serving, optimization, and observability

Complete these after the baseline RAG service, evaluation, and CI are working.

- [ ] Wrap the RAG pipeline with BentoML and serve the generative model with vLLM (or document the API-based model used initially).
- [ ] Add streaming `/ask` responses and verify progressive output with `curl`.
- [ ] Run Locust at 50 concurrent users and save the report under `reports/`.
- [ ] Add batch re-indexing and test it with at least one additional document.
- [ ] Document a canary rollout configuration and rollout stages.
- [ ] Compare original and AWQ 4-bit generation using RAGAS; report quality, latency, and model size.
- [ ] Evaluate reranker distillation if a reranker is part of the system.
- [ ] Add Langfuse traces for every `/ask` request, including retrieval and generation spans; attach the faithfulness score when available.
- [ ] Track answer faithfulness, embedding/cosine drift, and token usage/cost; add Prometheus/Grafana panels and an alert threshold.
- [ ] Document the final architecture and measured results in the README and project reports.

## Suggested order for the next work session

1. Review all 20 entries in `data/qa_civil_code/qa_random_articles.html` against their source crops.
2. Review the remaining `qa_flags` entries in the generated corpus; the known 558/1022 cases are already handled.
3. Review the 20-line sample in `data/qa_civil_code/qa_zero_width_alefs.html`, paying attention to both the displayed source crop and the before/after candidate text.
4. Record what you checked, any remaining extraction problems, and any approved correction in `reports/corpus-extraction.md`.
5. Add the Article 1021 regression assertion, then configure a DVC remote and verify `dvc push` / `dvc pull` from a clean checkout.

After this manual audit, proceed to Phase 2: build reproducible, article-aware ingestion and retrieval. Keep the corpus validation stage passing as the ingestion work begins.
