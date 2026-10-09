# Egyptian Civil Code — Corpus Extraction Report

## Scope and artifacts

The bilingual source PDF, `data/1576751803.pdf`, is converted by
`src/extract_egyptian_civil_code.py` into
`data/egyptian_civil_code_articles.jsonl`. The corpus contains 1,149 consecutive
article records, numbered 1–1149, with article citations, legal hierarchy,
source pages, and repeal metadata. Arabic source text (`ar_text`) is kept
separate from search-normalized Arabic (`ar_text_normalized`) and English
(`text_en`).

Extraction uses column-aware positioning to avoid interleaving the Arabic and
English columns. Arabic-Indic article numbers are converted to integers.
Repealed articles are retained rather than silently removed.

## Extraction defects and repairs

### Missing bold Arabic body text — Article 558

The extractor normally filters bold lines as hierarchy or topic headings.
Source inspection showed that Article 558's operative Arabic text is itself
bold, so this filtering incorrectly removed the article body.

The fix preserves bold Arabic body lines within Article 558's boundaries.
Regression checks verify the lease definition, including `عقد يلتزم المؤجر`
and `أجر معلوم`, and check that heading text and Article 559's body do not
leak into Article 558.

### Alef–lam switching — zero-width alef detection in the source PDF

Some Arabic words had alef and lam in the wrong extracted order. We fixed this
using character geometry from the source PDF, rather than globally replacing
`ال`, which is valid in ordinary definite-article words.

The extractor examines raw PDF glyphs for all four alef variants (`ا`, `أ`,
`إ`, `آ`). An alef with bounding-box width at most 0.1 PDF units immediately
followed by lam in the Arabic column is a candidate. The correction is mapped
to the matching word using glyph positions and the exact character offset in
the source span. Confirmed pairs are reordered at their own offsets. Matching
is performed against the original token before applying swaps, allowing
multiple confirmed pairs within one word to be corrected together.

The extractor also joins touching Arabic word boxes when a visually continuous
word was split across PDF text spans.

Verified regression examples:

| Article | Corrected source-text phrase |
| --- | --- |
| 1 | `مبادئ الشريعة الإسلامية` |
| 2 | `لا يجوز إلغاء` |
| 3 | `بالتقويم الميلادي` |

The diagnostic scan identified 3,501 zero-width alef variants across 2,389
lines. These are diagnostic counts, not the number of applied swaps.
`data/qa_civil_code/qa_zero_width_alefs.html` displays a deterministic
20-line sample (seed 42), source crops, and candidate before/after text.
The HTML candidate correction is a review preview; it is not a record of
every extraction-time correction.

### Source exception — Article 1022

Article 1022 has no Arabic body in the source PDF on page 147. It is an active
article, not a repealed one. Its Arabic fields remain empty, its English text
is retained, and it carries `missing_ar_text_in_source`. Ingestion must handle
the missing language explicitly without fabricating Arabic source text.

## Automated validation and reproducibility

`tests/test_corpus.py` checks numbering, required text with documented source
exceptions, repeal notices, article length bounds, citations, hierarchy
metadata, Arabic normalization, and the known extraction regressions.
The last verified pipeline run passed all 14 tests. That extraction produced
56 repealed records and nine QA-flagged records.

The source PDF and generated corpus/QA artifacts are DVC-tracked. `dvc.yaml`
defines extraction followed by corpus validation. Reproduce and validate with:

```bash
uv run dvc repro
```

Google Drive remote `storage` has been configured by the project owner.
Authentication settings are configured locally. The development dependencies
include DVC's Google Drive support.
Remote upload and clean-checkout retrieval must be recorded separately when
verified; configuring the remote alone does not demonstrate reproduction.

The upload was attempted with `uv run dvc push -r storage`. Google Drive
support was installed, and the client's incompatible `pyOpenSSL` dependency
was pinned to 24.2.1 (within PyDrive2's supported range). Authentication is
configured to use the service account. The initial HTTP 404 folder-access
error was resolved by granting access. A subsequent push reached upload but
failed for 45 files with HTTP 403: `Service Accounts do not have storage quota`.
For the personal Google Drive destination, authentication has now been
switched locally to user OAuth (`gdrive_use_service_account=false`). The
pending authorization/upload process was stopped when Stage 1 was paused
to start Stage 2. Resume with `uv run dvc push -r storage` when returning
to remote verification.
No successful upload or clean-checkout pull is claimed.

## Manual review status and remaining work

The targeted source inspections and regression examples above are completed.
The full 20-article visual sample review and review of every flagged record
have not yet been recorded as completed.

- [ ] Compare all 20 records in `data/qa_civil_code/qa_random_articles.html`
  with their source crops and record per-article findings.
- [ ] Review every record with `qa_flags` and document its disposition.
- [ ] Review the zero-width alef diagnostic sample against the source crops.
- [ ] Add a dedicated Article 1021/1022 boundary regression assertion.
- [ ] Verify remote upload and retrieval from a clean checkout.

Automated checks protect structure and known regressions; they do not certify
every extracted passage or translation. Stage 2 ingestion development can
proceed, with initial retrieval results considered provisional until the
remaining manual audit is completed. Rebuild the index after corpus repairs.
