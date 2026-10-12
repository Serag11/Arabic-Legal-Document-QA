# Clean-checkout reproducibility check

Verified on 2026-10-12 from commit `6535a9f` (branch `stage7-observability`).

**Procedure:** `git clone` into an empty directory → `uv sync --group evaluation` → copy the local
git-ignored `.dvc/config.local` (OAuth client settings) → `dvc pull -r storage` → `dvc repro`.

**`dvc pull`:** 55 files fetched from the Google Drive remote (53 added to the workspace), including
the source PDF, corpus, QA crops, chunks and Chroma index. It reused this machine's cached Google
OAuth token, so a collaborator's first pull will still go through an interactive Google login.

**`dvc repro`:** the stages were out of date (the committed `dvc.lock` predates later code and test
changes), so DVC re-ran every stage: re-extracted the corpus from the PDF, ran the 113-test suite
(passed), re-prepared chunks, re-embedded and rebuilt the index, and re-evaluated retrieval.

| Stage output | Pulled artifact vs regenerated |
| --- | --- |
| `data/egyptian_civil_code_articles.jsonl` (corpus) | identical |
| `data/qa_civil_code/` (QA crops and pages) | identical |
| `data/retrieval/chunks.jsonl`, `chunks.manifest.json` | identical |
| `evaluation/answer_questions.*` | identical |
| `reports/retrieval-results.metrics.json` | identical |
| `data/retrieval/chroma/` | differs (binary files are not byte-reproducible, as expected; contents are checked through the identical retrieval metrics) |
| `reports/retrieval-results.json` | differs (per-query timings and lower-ranked supporting passages) |

**Limits of this check:** the same machine, user, Hugging Face model cache and Google token were
used, so it shows the pipeline and remote are consistent, not that a stranger with no cache could
do it without the login and the explicit `src/retrieval.py download` step. The committed
`dvc.lock` was not updated; it will be refreshed with the final re-evaluation.
