# Corpus validation

Run `uv run python -m pytest tests/ -q` directly, or run `uv run dvc repro`
to extract the corpus and execute the validation stage.

Known source exceptions:

- **558:** Arabic body text is bold. The extractor preserves the bold lines
  inside this article's boundaries; the regression test checks its lease text.
- **1022:** Arabic text is absent in the source PDF (page 147), as verified by
  source review. Keep the text empty, the English text available, and the QA flag
  `missing_ar_text_in_source`. This is not a repealed article. Ingestion should
  handle the missing language explicitly instead of fabricating source text.

Validation checks numbering, required text with these exceptions, repeal notices,
length bounds, citations, metadata types, normalization, and known regressions.
It does not certify every translation or replace manual comparison with the PDF.
Null hierarchy fields are allowed because not every provision has every heading.
