# Stage 2 — local bilingual retrieval baseline

## Configuration and artifacts

- Model: `intfloat/multilingual-e5-base`, revision
  `d128750597153bb5987e10b1c3493a34e5a4502a`.
- Runtime: CPU FP32, six threads, batches of four; PyTorch `2.7.1+cpu`,
  Sentence Transformers `5.7.0`, Transformers `4.57.6`, Chroma `1.5.9`.
- CUDA dependency downloads were stopped after unusually slow progress.
  GPU fit and performance have not been measured. Completed download cache
  entries from that attempt were retained; the installed project uses CPU PyTorch.
- Input: 512-token maximum including special tokens and literal `query: ` /
  `passage: ` prefixes. Arabic queries and passages share extractor search
  normalization; returned quotations use original source text.
- Vectors: 768 dimensions, L2-normalized, cosine distance in local Chroma.
- Corpus SHA256:
  `3e2d390a79ff93b7f33e5a0e7d0934c444d4999428d4e86d5f29c2c4760473cc`.
- Chunks: `data/retrieval/chunks.jsonl` and its preparation manifest.
- Index: `data/retrieval/chroma/`, with a runtime/configuration manifest.
- Evaluation: `reports/retrieval-results.json` (per-query evidence/configuration)
  and `reports/retrieval-results.metrics.json` (DVC metric summary).

The model was explicitly downloaded once. All preparation, indexing, and
evaluation subsequently completed with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
No default Chroma embedding model is used or downloaded.

## Token audit and source provenance

The initial 2,241 whole-article language documents cover all 1,149 articles.
Only **English Article 1143** exceeded the model budget: **518 tokens**.
Its unnumbered English provisions were checked against PDF pages 168–169 and
the five corresponding numbered Arabic provisions. Exact provision-start
anchors are recorded in the configuration; greedy provision packing produces
two English chunks. No sentence or token was arbitrarily cut.

Final counts: **2,242 chunks**, **1,092 Arabic**, **1,150 English**, including
**56 English repeal notices**. Maximum final input length is **450 tokens**.
Overlap is zero. Every chunk retains article identity, source field and exact
character offsets, hierarchy, language, repeal status, and QA flags.
Article 1022 remains English-only with `missing_ar_text_in_source`.

## Measured retrieval results

The evaluation has 30 Arabic/English pairs (60 queries): 25 semantic pairs,
three exact-number lookup pairs, and two unsupported probes. Each supported
query currently has one seed relevant article; Hit@5 and Recall@5 therefore
coincide. Labels are not exhaustive. The user subsequently confirmed manual
review; approval of the exact file is recorded in `evaluation/review.json`.

| Subset | Queries per language | Arabic Hit/Recall@5 | English Hit/Recall@5 | Arabic MRR@5 | English MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic retrieval | 25 | 96.0% (24/25) | 100% (25/25) | 0.8093 | 0.8893 |
| Exact article lookup | 3 | 100% | 100% | 1.0000 | 1.0000 |
| All supported | 28 | 96.43% (27/28) | 100% (28/28) | 0.8298 | 0.9012 |

MRR@5 assigns reciprocal rank zero when the expected article is outside the
first five distinct article results. Results group retrieved language/chunk
matches by article. Exact lookup is reported separately and has no similarity
score. Bilingual top-1 agreement over supported pairs is **64.29% (18/28)**;
agreement alone does not establish relevance.

Unsupported probes have no seed targets and are excluded from relevance
denominators. Semantic retrieval still returns nearest neighbors for them.
There is no calibrated refusal threshold or answer generator at this stage.
Cosine similarity is not a confidence score.

### Observed failure

Arabic query `s2-024` asks whether independently mortgaged parts of a property
must be valued separately when purging mortgages. Its expected Article **1065**
was absent from the top five; retrieved articles were **331, 1075, 1101, 1063,
1083**. The English equivalent ranked Article 1065 first. Preserve this case
for subsequent cross-language retrieval/model comparisons rather than changing
the evaluation question to improve a score.

## Resource measurements

On the i7-10750H laptop, the final offline build took **228.1 seconds** for
embedding plus indexing, with **7.0 seconds** for model initialization.
Peak process RSS was **1,642.8 MiB (about 1.60 GiB)**, measured with Linux
`getrusage`; this is process memory, not whole-system RAM consumption.
No GPU memory measurement applies to this CPU run. After installation and
downloads, the root filesystem had approximately **9.4 GiB free**.

## Reproduction and checks

```bash
# Explicit model download is a one-time prerequisite on another machine.
uv run python src/retrieval.py download
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run dvc repro
uv run python src/retrieval.py query "Article 1022"
uv run python src/retrieval.py query "lease obligations" --filters '{"language":"en","is_repealed":false}'
```

- **32 tests passed**, including real Chroma persistence, rebuild/stale-ID
  removal, metadata filters, exact source mapping, source exceptions, token
  guards, provision anchors, and immutable-artifact reads. Tests use a fake
  encoder and make no external model calls.
- Real preparation/indexing/evaluation were repeated through DVC with the same
  chunk hash, document counts, and retrieval metric values.
- Evaluation loads the index in a separate process; an additional fresh CLI
  query verified reload and cited output.
- Chroma writes SQLite housekeeping on collection load. Retrieval opens a
  temporary runtime copy so queries/evaluation do not change the DVC-tracked
  index. Python/test caches are excluded from DVC dependency hashes.
- `dvc status` reports the local data and pipeline up to date.

**Stage 2 is accepted following the user's manual evaluation review.** The
latest saved results include the matching manual approval record. Deferred Stage 1 corpus QA,
candidate-model comparison, MLflow experiment tracking, RAGAS faithfulness,
and the Drive upload remain follow-ups. Stage 3 starts with Google-backed
generation using the root `.env` key.
