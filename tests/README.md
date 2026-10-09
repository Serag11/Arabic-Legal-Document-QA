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

## Stage 2 ingestion

`test_ingestion.py` verifies that prepared chunks map to exact source-field
slices and preserve article citations and metadata. It covers English-only
Article 1022, source repeal notices, unexplained missing Arabic, duplicate
article numbers, and repeatable Unicode JSONL output.

Prepare the whole-article baseline with:

```bash
uv run python src/rag.py --out /tmp/opencode/civil-code-chunks.jsonl
```

This writes 2,241 source-traceable documents and a `.manifest.json` companion
containing the corpus hash and preparation configuration. It does not perform
embedding or indexing. Run corpus validation before preparing an index.

## Local E5 / Chroma retrieval

The selected model and immutable revision are recorded in `config/retrieval.json`.
Download once explicitly; subsequent commands use cached files only:

```bash
uv run python src/retrieval.py download
uv run dvc repro evaluate_retrieval
uv run python src/retrieval.py query "ما أثر العقد بين المتعاقدين؟"
uv run python src/retrieval.py query "Article 1022"
uv run python src/retrieval.py query "lease obligations" --filters '{"language":"en","is_repealed":false}'
```

The baseline uses CPU-only PyTorch to keep runtime installation manageable.
CUDA-package downloads were stopped after unusually slow progress; GPU execution
has not been measured. The encoder can select CUDA when a compatible GPU build
is installed, but changing the runtime requires a separately recorded experiment.

Preparation counts tokens including E5 prefixes and special tokens. Over-budget
articles are split only at source-provision boundaries, with exact source
offsets and no overlap. An indivisible over-budget provision raises an error.
Arabic queries use the same search normalization as Arabic passages.

English Article 1143 is 518 tokens including the prefix and special tokens.
Its English numbering was not retained by this edition. The five provision
starts in the configuration were checked against PDF pages 168–169 and the
five corresponding numbered Arabic provisions. These source anchors permit
splitting without an arbitrary token or sentence cut; other whole articles fit.

`test_retrieval.py` uses a deterministic fake encoder and real local Chroma;
it does not download model files. It verifies provision splitting/token safety,
source mapping, metadata filters, source exceptions, persistent reload, rebuilds
without stale IDs, and rejection of mismatched configuration.

Queries group passage matches by article. Explicit article-number queries use
exact lookup and return no similarity score. Semantic scores are cosine
similarity, not confidence or an unsupported-question decision. Out-of-domain
probes are excluded from retrieval relevance metrics. The user confirmed manual
review; approval of the evaluation-file hash is recorded in `evaluation/review.json`.

Chroma updates local SQLite bookkeeping when loading a collection. Retrieval
therefore opens a temporary runtime copy of the persisted index so queries and
evaluation do not mutate DVC-tracked index files. The copy lives for the lifetime
of the retriever and is automatically cleaned up. `.dvcignore` excludes Python
bytecode/test caches from pipeline dependency hashes.

## Stage 3 Google answer generation

```bash
uv run python src/rag.py ask "وفق المادة 147، هل يجوز نقض العقد بإرادة طرف واحد؟"
uv run python src/rag.py ask "What is a lease under Article 558?" --language en
```

The root `.env` supplies `GOOGLE_API_KEY`; the key is neither tracked nor added
to answer reports. `config/generation.json` selects `gemini-2.5-flash`.
`test_generation.py` uses fake retrieval/provider boundaries, makes no Google
calls, and tests exact evidence, citation membership, insufficient evidence,
empty inputs, context limits, source exceptions, dotenv loading and error redaction.

Explicit live smoke evaluation (six billable Google generation calls):

```bash
uv run python src/evaluate_generation.py
```

This saves `reports/generation-smoke.json`. It is not part of DVC reproduction
or the offline test suite, and does not measure semantic faithfulness/RAGAS.
