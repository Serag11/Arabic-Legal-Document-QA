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

### Source exception — Articles 1021 and 1022

Article 1022's own Arabic cell is empty in the source PDF (page 147). It is an
active article, not a repealed one. Its Arabic fields remain empty, its English
text is retained, and it carries `missing_ar_text_in_source`. Ingestion must handle
the missing language explicitly without fabricating Arabic source text.

The manual review below found that the Arabic does exist on the page, in the
**wrong row**. Article 1021's Arabic cell holds three paragraphs: (١) translates
Article 1021's English; (٢) and (٣), highlighted yellow like the whole 1022 row,
translate the second and third sentences of Article 1022's English. The first
sentence of 1022's English (the dominant owner bears the cost of necessary works)
has no Arabic counterpart anywhere on the page. The extractor copies the 1021 cell
literally, so `ar_text` of 1021 currently contains (٢) and (٣). Whether to keep
the literal row structure or move (٢)–(٣) to Article 1022 is an editorial decision
for the project owner; nothing has been moved.

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
The remote now holds the tracked artifacts and a clean-checkout `dvc pull` + `dvc repro` has been
verified; see `reports/reproducibility.md`. Setup history: the client's incompatible `pyOpenSSL`
dependency was pinned to 24.2.1 (within PyDrive2's supported range), a service account was tried first but
Google Drive rejected uploads with HTTP 403 (`Service Accounts do not have storage quota`), so the project
uses user OAuth (`gdrive_use_service_account=false` in the git-ignored `.dvc/config.local`, with the OAuth
client ID and secret kept there as well).

## Manual review (recorded)

Reviewed by comparing each record with a rendered view of the source PDF. The
comparisons were done by AI agents reading the images, with the Article 533 and
Article 1021/1022 findings re-checked by the coordinating assistant; this is a
useful audit, not a certified legal review.

**20 sampled crops** (`data/qa_civil_code/qa_random_articles.html`): no extraction
error found in any of the 20 records (52, 82, 89, 93, 206, 219, 237, 256, 313, 464,
504, 508, 514, 533, 558, 620, 916, 921, 976, 1091), and every article number matches.
Limits of that statement:

- Crops for 237, 504, 916 and 1091 are cut off, so only the visible part was compared
  (237 is "unsure" for its English tail; 504, 916 and 1091 were checked up to the
  crop edge).
- Source spellings and typos are preserved and are not extraction errors: "وغلى" in
  Article 533 (for "وإلى"), "الإلتزام"/"الشئ" in 206, "أجراء"/"أى" in 219, "الأراضى"
  in 921, "o f" in 1022, "to a be a legacy" in 908.

**Flagged records** (`single_column_article_anchor`: 54, 277, 601, 714, 746, 898,
908, 1022, 1090):

| Article | Disposition |
| --- | --- |
| 54 (and repealed 55–80) | Correct: English repeal note only. The source also prints an *Arabic* repeal notice that is not captured (the schema has no Arabic repeal field). Known limitation; Arabic questions about a repealed article are answered from the English notice. |
| 277, 714, 746, 898, 908, 1090 | Correct. 714 and 908 span a page break and are stitched correctly. |
| 601 | Content correct; cosmetic glitch in `ar_text`: `(١ لا ينتهي … المستأجر.) (٢)` where the page shows `(١) لا ينتهي …`. Only record with an unclosed `(digit` marker. Quote validation uses substrings, so answers are unaffected. Not fixed. |
| 1021 / 1022 | Boundary issue described above; decision pending. |

The 24 records with unbalanced parentheses (5, 52, 300, …) were not individually
reviewed beyond Article 52, which matched its source.

**Zero-width alef sample** (`qa_zero_width_alefs.html`, 20 lines): all 20 cards match their source
crops and the corpus holds the corrected forms. A corpus-wide scan for remaining swapped
alef/lam words flagged "ثالثا" (Article 352) and "الثالثة" (Article 1012); both are correct Arabic
("thirdly", "the third paragraph") and the English of Article 352 says "third", so they are not defects.

## Remaining work

- [ ] Decide whether to move Article 1021's Arabic paragraphs (٢)–(٣) to 1022 (editorial; it would also
  change the reviewed s2-022 evaluation probes). The literal layout is pinned by
  `test_article_1021_boundary_with_1022_matches_the_source_rows`.
- [X] Clean-checkout `dvc pull` + `dvc repro`: see `reports/reproducibility.md`.

Automated checks protect structure and known regressions; they do not certify every
extracted passage or translation. Rebuild the index after corpus repairs.
