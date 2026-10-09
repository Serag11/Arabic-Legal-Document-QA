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

## Phase 1 — Make the corpus trustworthy (remaining work deferred)

### 1. Investigate extraction defects

- [X] Compare Articles 558 and 1022 with the original PDF: Article 558's Arabic body is bold; Article 1022 has no Arabic body in the source.
- [X] Check the Article 1021–1022 boundary against the source; preserve Article 1022 as an explicit missing-Arabic source exception.
- [ ] Review every record with a `qa_flags` entry, prioritizing single-column article anchors and missing text.
- [ ] Inspect `data/qa_civil_code/qa_random_articles.html` and compare all 20 sampled records with their PDF crops.
- [X] Record known extraction issues, bold-body and zero-width alef fixes, source exceptions, and current review status in `reports/corpus-extraction.md`.
- [ ] Complete the report with the 20-article and flagged-record manual review outcomes.

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
- [X] Configure Google Drive DVC remote `storage`; use local user OAuth settings for the personal Drive destination (browser authorization pending).
- [ ] Push tracked artifacts to `storage` and verify `dvc pull` from a clean checkout.
- [X] Verify `dvc repro` regenerates the structured corpus and QA artifacts from the tracked source PDF.

**Phase 1 exit criteria:** corpus tests pass; active article text is present in both languages except the documented source exception; flagged records and the 20-item visual sample have been reviewed; DVC can reproduce the extraction and a remote is available to collaborators.

## Phase 2 — Build article-aware retrieval

**Goal:** validated JSONL → article-aware chunks → multilingual embeddings → persistent Chroma index → cited retrieval results. This implements the article-first guidance on handbook pp. 64–66.

**Sequence decision:** proceed with Phase 2 development now and return to the unchecked Phase 1 tasks. Automated corpus validation must pass before each index build. The handbook's 20-article visual review remains outstanding, so initial retrieval results are provisional until that audit is completed.

**Chosen defaults:** persistent local Chroma; include repealed articles with explicit status and source repeal notices. Embedding provider/model selection is deferred until implementation, before the first real index build.

### 1. Define ingestion and result contracts

- [X] Create `src/rag.py` ingestion entry point with source-traceable document/result contracts and separate loading/preparation responsibilities.
- [X] Implement embedding, indexing, and retrieval in the separate `src/retrieval.py` entry point after selecting the model.
- [X] Preserve integer article number, citation, book, chapter, section, topic, source page, language, QA flags, and repeal metadata.
- [X] Assign stable IDs containing corpus identity, article number, language, and chunk position.
- [X] Retain source-faithful text for returned passages and quotations; use normalized Arabic separately for embedding input.
- [X] Prepare Article 1022's available English text only and retain its missing-Arabic source-exception flag.

**Acceptance:** each chunk references an existing corpus article, carries that article's number and citation, and contains a passage from that article (or its source repeal notice). Splitting Article 147 into several chunks must leave every chunk traceable to Article 147.

### 2. Implement article-aware chunking

- [X] Begin with one chunk per article per available language.
- [X] Inspect paragraph boundaries: Arabic line breaks are flattened; 509 articles retain parenthesized numeric markers. Maximum Arabic/English article lengths are 1,404/2,141 characters.
- [X] Token-audit all passages and split only at source provision boundaries: English Article 1143 is 518 tokens; source-verified provision anchors on PDF pages 168–169 produce two exact-source chunks. All other language documents fit whole.
- [X] Prepare and index repealed records as English source repeal-notice chunks without inventing original provision text.
- [X] Record whole-article strategy, no size limit, and zero overlap in the preparation manifest; enforce model input limits when embeddings are selected.

**Preparation verified:** the initial whole-article output has 2,241 documents. Token-safe E5 preparation has 2,242 chunks covering all 1,149 articles: 1,092 Arabic and 1,150 English (including 56 repeal notices and the two Article 1143 English chunks). Maximum chunk length is 450 tokens including prefix/special tokens, with zero overlap. Use `uv run python src/retrieval.py prepare`; a persistent Chroma index is now built.

**Acceptance:** chunks never cross article boundaries or cut arbitrarily through provisions; no passage is silently truncated to fit the embedding model. Document any provision too long for the selected model.

### 3. Select multilingual embeddings

- [X] Check local resources and shortlist multilingual local models in `reports/embedding-model-options.md`: initially 18 GiB free disk, 5.2 GiB available RAM, GTX 1650 with 4 GiB VRAM. User approved `intfloat/multilingual-e5-base`.
- [ ] Compare candidates using the same paired bilingual questions.
- [X] Pin `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a` in `config/retrieval.json`: 768 dimensions, 512 tokens including prefixes/special tokens, normalized vectors, `query: ` / `passage: ` formatting.
- [X] Apply consistent Arabic normalization to documents and queries, preserving source text separately.

**Acceptance:** a concrete embedding configuration is selected and documented, with measured Arabic/English retrieval results.

### 4. Build the persistent index reproducibly

- [X] Build the Chroma collection from generated chunks and explicit embeddings.
- [X] Support language, hierarchy, repeal-status, and integer article-number filters.
- [X] Record corpus hash, embedding configuration, normalization version/code hash, chunking configuration, runtime precision/device, package versions, memory, and document counts in an index manifest.
- [X] Add test-gated preparation, index construction, and evaluation to DVC, with declared code/configuration dependencies and generated outputs. Model weights are an explicitly downloaded, pinned external prerequisite.
- [X] Verify rebuilds and persistence/reload with offline tests and a repeated real DVC run; use temporary runtime copies to prevent query/evaluation writes from changing tracked Chroma artifacts.

**Acceptance:** the index can be reconstructed from the tracked corpus and recorded configuration and loaded in a fresh process. Index binary files need not be byte-identical; document IDs, content, configuration, and retrieval checks must remain reproducible.

### 5. Implement retrieval-only queries

- [X] Return ranked source passages, article citations, language, hierarchy, and repeal status.
- [X] Group duplicate language/chunk matches by article while preserving useful supporting passages.
- [X] Support direct article-number lookup, including repealed articles, so explicit article queries do not depend entirely on semantic similarity.
- [X] Keep article citations as the public source identity for the later `/ask` response.
- [X] Test source mapping, missing Arabic, repeal notices, paragraph splitting, filters, persistence/reload, immutable-artifact reads, and rebuild behavior without live external model calls (32 tests pass).

**Acceptance:** results identify verifiable legal articles and clearly identify repeal notices.

### 6. Establish bilingual evaluation and experiment records

- [X] Create a draft paired evaluation set in `evaluation/retrieval_questions.jsonl`: 28 supported pairs over 27 articles plus two unsupported pairs; verify exact reference excerpts and citations.
- [X] User confirmed manual review of the bilingual evaluation set; approval and file hash are recorded in `evaluation/review.json`.
- [X] Include paraphrases, long articles, repeal queries, and the missing-language exception.
- [X] Measure article-level Hit@5/Recall@5, MRR@5, and bilingual top-1 agreement; report Arabic and English, semantic retrieval, and exact lookup separately.
- [X] Save configuration and per-question results in `reports/retrieval-results.json` for later MLflow comparisons, with token budget, source-provision chunk strategy, zero overlap, and pinned embedding model.
- [X] Record the provisional baseline and Arabic Article 1065 failure in `reports/retrieval-baseline.md`: semantic Hit@5 is 24/25 Arabic and 25/25 English; exact lookup is 3/3 in each language.

**Evaluation handoff:** RAGAS faithfulness requires generated answers and will be measured in the later end-to-end chunking experiments. Retrieval-only scores are not a substitute. Retain the paired questions for the later RAGAS scorecard and contract-law versus family-law query-shift scenario described on p. 65.

**Phase 2 exit criteria:** corpus validation passes; a persistent index rebuilds reproducibly; source-mapping and metadata tests pass; and a hand-checked bilingual evaluation produces a documented retrieval baseline. Complete the deferred visual corpus audit before treating this baseline as verified corpus quality.

**Stage 2 accepted:** user confirmed manual evaluation review and authorized Stage 3. Deferred Stage 1 source-audit/remote tasks and candidate-model comparisons remain follow-ups, not blockers to starting generation.

## Phase 3 — Create the first end-to-end RAG baseline

- [ ] Add a generator that answers only from retrieved article context.
- [ ] Define a clear insufficient-evidence response; do not guess when the retrieved passages do not support an answer.
- [ ] Ensure generated answers return article citations, not chunk IDs or PDF page numbers.
- [ ] Extend Phase 2's `src/rag.py` entry point with generation and end-to-end query responsibilities.
- [ ] Add tests for empty questions, retrieval citations, unsupported questions, and source formatting.
- [ ] Track configuration such as chunking, embedding model, retrieval top-k, and prompt version for every baseline run.

**Example API contract:**

```json
POST /ask
{"question": "ما أثر العقد بين المتعاقدين؟"}
```

```JSON
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
- [ ] Add a reranker baseline and distill it; compare retrieval and RAGAS quality before/after as required by the handbook's Project 2 optimization deliverables (p. 66).
- [ ] Add Langfuse traces for every `/ask` request, including retrieval and generation spans; attach the faithfulness score when available.
- [ ] Track answer faithfulness, embedding/cosine drift, and token usage/cost; add Prometheus/Grafana panels and an alert threshold.
- [ ] Document the final architecture and measured results in the README and project reports.

## Suggested order for the next work session

1. Start Stage 3 cited answer generation using Google's API and `GOOGLE_API_KEY` from the root `.env`.
2. Test supported questions, insufficient evidence, repeal notices, and the missing-language exception.
3. Retain the Arabic Article 1065 miss and deferred source QA/model comparisons as follow-ups.

**Return to Phase 1:** review all 20 sampled article crops and every flagged record; review the zero-width alef diagnostic sample; document findings in `reports/corpus-extraction.md`; add the Article 1021 regression assertion; configure a DVC remote and verify clean-checkout push/pull reproduction. Keep these tasks unchecked until completed, and rebuild the index after any corpus corrections.
