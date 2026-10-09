# Local multilingual embedding options — Stage 2

## Measured local resources

At inspection, the project, user cache, and `/tmp` share the root filesystem:

- Disk: 123 GiB total, approximately 18 GiB available (86% used).
- RAM: 15 GiB total, 5.2 GiB available; swap already uses 4.6 GiB of 6.2 GiB.
- CPU: Intel Core i7-10750H, six physical cores / 12 threads.
- GPU: NVIDIA GeForce GTX 1650, 4,096 MiB total VRAM, 3,596 MiB free.
- No additional large filesystem is currently mounted.

These are a resource snapshot, not measured model runtime requirements.

## Popular local candidates

All candidates below support multilingual text including Arabic and English.
Sizes are approximate decimal GB for **one main published weight file**,
queried from the publisher's Hugging Face repository. They exclude tokenizers,
runtime dependencies, duplicate formats, caches, and inference activations.

| Model | Published weight file | Vector dimensions | Input limit | Fit and trade-off |
| --- | ---: | ---: | ---: | --- |
| `intfloat/multilingual-e5-small` | 0.47 GB | 384 | 512 tokens | Lowest-resource retrieval-specific baseline; good CPU fallback. |
| `intfloat/multilingual-e5-base` | 1.11 GB | 768 | 512 tokens | Recommended balance of memory footprint and retrieval-oriented training. |
| `intfloat/multilingual-e5-large` | 2.24 GB | 1,024 | 512 tokens | Feasible with small GPU batches/FP16, but more memory without a larger context limit. |
| `BAAI/bge-m3` | 2.27 GB | 1,024 | 8,192 tokens | Useful longer-context challenger; begin with dense retrieval only and small batches. |
| `Qwen/Qwen3-Embedding-0.6B` | 1.19 GB (BF16 weights) | 32–1,024; default 1,024 | 32,768 tokens | Modern instruction-aware challenger; use FP16 rather than BF16 on this GPU, and short actual sequences. |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0.47 GB | 384 | 128 tokens in its Sentence Transformers configuration | Compact similarity model, but the short default limit is a poor match for whole legal articles. |

E5 requires literal `query: ` and `passage: ` prefixes, including for Arabic.
BGE-M3 does not require query instructions. Qwen recommends a task instruction
on queries, without adding it to document passages. Preserve these differences
in a provider adapter and experiment configuration.

## Recommendation

Start with **`intfloat/multilingual-e5-base`**. Its published FP32 weights are
about 1.11 GB; FP16 model weights in memory are roughly half that, before
activation/framework overhead. It is a practical candidate for 4 GiB VRAM
with a small initial batch (for example four documents), and CPU FP32 is a
fallback. Exact memory, latency, and Arabic legal retrieval quality still need
measurement. The user approved this model; its immutable revision and input
configuration are now pinned in `config/retrieval.json`.

**Execution update:** the pinned model is downloaded and the baseline is indexed
using CPU FP32. CUDA dependency downloads were stopped due to unusually slow
progress; the project now uses the smaller CPU-only PyTorch build. Runtime GPU
fit remains unmeasured. See `reports/retrieval-baseline.md` for measured results.

Before embedding, measure every prepared passage using the chosen tokenizer,
including prefixes and special tokens. If any passage exceeds 512 tokens,
split at numbered-provision boundaries or select a longer-context candidate;
do not silently truncate. Corpus character lengths alone cannot prove token fit.

Use **BGE-M3** as the longer-context comparison if preserving whole articles
without splitting becomes important. Qwen3-Embedding-0.6B is another feasible
candidate, but the advertised long context does not mean 32K-token batches fit
this GPU. Compare candidates sequentially on the same bilingual questions,
measuring article-level retrieval quality, latency, and peak memory.

The 2,241-document corpus is small. Raw FP32 vectors consume approximately
3.4 MB at 384 dimensions, 6.9 MB at 768, or 9.2 MB at 1,024, before Chroma's
metadata/index overhead. Model/runtime installation is the main disk cost,
not the vectors. Download one required weight format and one candidate at a
time rather than cloning every model repository. PyTorch/CUDA dependencies
can add several GB, depending on the selected build.

The GTX 1650 does not provide native BF16 support; FP16 or FP32 is the appropriate
starting precision. Validate the selected PyTorch build's GPU architecture
support during setup, and do not assume FlashAttention support on this card.

## Evaluation prepared

`evaluation/retrieval_questions.jsonl` contains 30 Arabic/English pairs:
28 supported pairs over 27 articles and two unsupported probes. Citation and
exact-reference checks passed. Direct-number repeal/missing-language cases
must be reported separately from semantic questions, so lookup success is not
mistaken for embedding quality. The set is draft pending manual review;
retrieval results are now recorded in `reports/retrieval-results.json`, with
separate semantic-retrieval and direct-lookup metrics.

## Publisher sources

- [E5 small](https://huggingface.co/intfloat/multilingual-e5-small)
- [E5 base](https://huggingface.co/intfloat/multilingual-e5-base)
- [E5 large](https://huggingface.co/intfloat/multilingual-e5-large)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

Weight sizes were checked using each model's Hugging Face `api/models/<id>/tree/main`
listing. Published benchmark scores are not a substitute for evaluation on
this corpus.
