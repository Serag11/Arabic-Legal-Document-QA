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
- [X] Configure a DVC remote and push tracked artifacts (user confirmed).
- [ ] Verify the full corpus pipeline is reproducible from a clean checkout using the remote.
- [X] Build article-aware retrieval, then generation and the `/ask` API.

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
- [X] Configure Google Drive DVC remote `storage` (user confirmed configuration is complete).
- [X] Push tracked artifacts to `storage` (user confirmed successful push).
- [ ] Verify `dvc pull` from a clean checkout.
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

- [X] Add Google `gemini-2.5-flash` generation from retrieved context in `src/generation.py`, using `GOOGLE_API_KEY` from root `.env`.
- [X] Define localized insufficient-evidence responses for empty retrieval, model-reported unsupported questions, and invalid source grounding.
- [X] Return article citations and validated verbatim source excerpts; reject citations outside context and fabricated quotes.
- [X] Extend `src/rag.py` with the `ask` command while preserving chunk-preparation usage.
- [X] Add offline tests for empty questions, citations, unsupported questions, exact source formatting, missing Arabic, repeal metadata, credential loading, and redacted provider errors.
- [X] Record generation/prompt settings, retrieval configuration, corpus/chunk hashes, filters, response model version, and token usage for every answer.

**Stage 3 baseline:** six live Google smoke cases passed; see `reports/generation-baseline.md` and `reports/generation-smoke.json`. These check source identity and abstention, not a RAGAS faithfulness score. Service/API packaging remains Stage 4.

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

- [X] Create installable `src/` package with setuptools and `legal-qa` / `legal-qa-evaluate` entry points; external resources are located via `LEGAL_QA_ROOT` for deployment.
- [X] Implement FastAPI `/ask` with Pydantic schemas and HTTP 422 for empty/invalid questions.
- [X] Implement `/health` reporting service status and indexed document/article counts.
- [X] Load a shared retriever at startup; serialize the baseline CPU path to avoid duplicate encoder loads.
- [X] Add offline API lifecycle, response, validation, and provider-error tests.
- [X] Build Docker image and verify Compose startup/health and a live cited `/ask` response; use read-only index/cache mounts and runtime `.env` credentials.
- [X] Update README with architecture, request examples, and exactly three quickstart commands.

## Phase 5 — Evaluate and track experiments

- [X] Derive fixed answer-evaluation records from the user-reviewed retrieval pairs, preserving their reference excerpts, source contexts, and expected citations.
- [X] Include Arabic/English paraphrases, source exceptions, opposite-language retrieval experiment, and unsupported refusal probes.
- [X] Measure retrieval hit/recall/MRR, citation/answer coverage, refusal accuracy and RAGAS faithfulness; document metric definitions and results in `reports/answer-evaluation.md`.
- [X] Expand to 60 language-specific questions (30 reviewed pairs): 56 supported questions and four refusal probes; record provenance in `evaluation/answer_questions.manifest.json`.
- [X] Run five MLflow retrieval comparisons on the same 24-question subset; log chunk size/units/strategy, overlap, embedding model/revision, retrieval settings and faithfulness. After the API loader fix, the user reran all comparisons and full 60-question validation; top-k 5 is selected.
- [X] Register `ArabicLegalDocumentQA` version 2 with alias `champion`; verify the alias, completed registration, and matching refreshed evaluation run. Version 1 reload/prediction was verified previously. Credentials and Google weights are not stored.
- [X] Add and locally pass the CI gate: complete fixed-dataset coverage, >=20 scored supported questions, no operational errors, current provenance and coverage-adjusted faithfulness >=0.75. Refreshed result: 0.9250 with 52 scored supported answers.

**Stages 4 and 5 complete:** installable package, healthy Docker/FastAPI service, 69 offline tests, five comparative runs, refreshed full 60-question evaluation, registered champion and locally passing CI gate. The service and registered champion both use top-k 5. Live GitHub execution and delivery/publishing are Stage 6.

## Phase 6 — Add CI and delivery automation

- [X] Add GitHub Actions for lint, offline tests, measured quality/runtime-equivalence gates, Docker build, offline container checks, and conditional Docker Hub publication in `.github/workflows/delivery.yml`.
- [X] Store Docker Hub credentials in GitHub Actions secrets (user confirmed setup); runtime Google credentials remain external to the image.
- [X] Track the source PDF and derived corpus with DVC; configure the remote and push tracked artifacts (user confirmed remote setup and push).
- [ ] Verify clean-checkout reproduction using `dvc pull` and `dvc repro`.
- [X] Use a checksum-pinned GitHub Release runtime bundle for CI instead of DVC/Google Drive authentication; create the local `runtime-v1.tar.gz` bundle and descriptor.
- [X] Export and compare FP32, INT8, and compact ONNX models. Select FP16 weight storage with FP32 CPU computation: 555 MB, all 60 generator contexts unchanged; reject INT8's retrieval drop from 55/56 to 53/56.
- [X] Build and locally verify a self-contained, non-root, PyTorch-free image with bundled tokenizer/model/index; document running it without cloning the repository.
- [X] Authenticate GitHub CLI and verify repository Actions secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.
- [X] Publish `runtime-v1` with the runtime archive and manifest; verify GitHub's archive SHA256 matches `config/runtime-artifacts.json` (529,911,987 bytes).
- [X] Run the workflow on GitHub and verify Docker Hub publication to public `serag11/arabic-legal-document-qa`: run `38151794649` passed for commit `efef5f3`, including full offline image checks and anonymous pull.

**Stage 6 delivery design:** DVC stays local for corpus/index reproducibility. CI obtains runtime artifacts from a pinned GitHub Release and verifies checksums. The additional deployment gate permits reuse of the current generation scores only after exact context equivalence on the complete 60-question evaluation; changed contexts require fresh answer evaluation. See `reports/stage6-delivery.md` for bootstrap instructions and verification results.

**Storage constraint:** remove intermediate export models, expanded runtime bundles, and temporary Stage 6 images/build-cache records after local verification. Retain only the compressed release archive for upload; perform subsequent image builds in GitHub Actions.

## Phase 7 — Serving, optimization, and observability

Complete these after the baseline RAG service, evaluation, and CI are working.

- [ ] Wrap the RAG pipeline with BentoML and serve the generative model with vLLM (or document the API-based model used initially).
- [X] Add streaming `POST /ask/stream` (SSE: context, delta, validated final) and verify progressive output with `curl`.
- [X] Run Locust at 50 concurrent users (simulated provider latency) and save the report under `reports/`; see `reports/load-test.md`. Retrieval-only locking raised throughput from 0.5 to ~13 req/s.
- [ ] Add batch re-indexing and test it with at least one additional document.
- [ ] Document a canary rollout configuration and rollout stages.
- [ ] Compare original and AWQ 4-bit generation using RAGAS; report quality, latency, and model size.
- [ ] Add a reranker baseline and distill it; compare retrieval and RAGAS quality before/after as required by the handbook's Project 2 optimization deliverables (p. 66).
- [X] Add Langfuse traces for every `/ask` request (ask, retrieval, generation, grounding spans; rejected candidate kept). Attaching faithfulness scores to traces is still open.
- [X] Track answer faithfulness (sampled), embedding/cosine drift, and token usage/cost; Prometheus rules (promtool-tested) and Grafana dashboard added in `monitoring/`; containers not yet run (see `reports/monitoring.md`).
- [ ] Document the final architecture and measured results in the README and project reports.

## Suggested order for the next work session

1. Proceed to Stage 7 serving/optimization/observability; Stage 6 CI and public image delivery are verified.
2. Verify clean-checkout reproduction with `dvc pull` and `dvc repro`; remote configuration and artifact upload are complete.
3. Retain source QA and the Arabic Article 1065 retrieval miss as follow-ups.

**Return to Phase 1:** review all 20 sampled article crops and every flagged record; review the zero-width alef diagnostic sample; document findings in `reports/corpus-extraction.md`; add the Article 1021 regression assertion; verify clean-checkout reproduction from the configured and populated DVC remote. Keep these tasks unchecked until completed, and rebuild the index after any corpus corrections.
