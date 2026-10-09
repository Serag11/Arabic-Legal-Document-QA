# Stage 5 — bilingual evaluation and experiment tracking

## Dataset and protocol

`evaluation/answer_questions.jsonl` contains **60 language-specific questions**,
derived deterministically from the **30 user-reviewed Arabic/English pairs**.
There are 56 supported questions and four unsupported probes. References are
the already-reviewed exact source excerpts; English references are not presented
as fabricated Arabic source quotations. Full source contexts and article labels
are preserved. The manifest records corpus/dataset hashes and manual approval.

Five actual retrieval configurations use the same fixed **24-question** subset
(20 supported questions and four refusal probes): top-k 1, 3, 5, 8 and top-k 5
restricted to the opposite source language. Chunk strategy, 512-token budget,
zero overlap, pinned E5 model/revision and generation settings are logged.
The best eligible configuration is then validated on all 60 questions, reusing
its unchanged benchmark records. Evaluation caches are tied to exact fingerprints.

## Metrics

- Article-level retrieval Hit@k, Recall@k and MRR@k over supported questions.
- Supported answer coverage and expected-article citation hit.
- RAGAS faithfulness: statement support against the exact context sent to the
  generator, including source status/language metadata.
- Coverage-adjusted faithfulness: sum of valid supported-answer faithfulness
  scores divided by all supported questions. Abstentions contribute zero here,
  not a fabricated raw RAGAS score.
- Unsupported refusal accuracy, latency and operational/scoring errors.
- Separate Arabic/English retrieval, citation and adjusted-faithfulness results.

The optional RAGAS reference-answer accuracy judge is disabled in this first
quota-conscious protocol; it must not be represented as measured correctness.
Exact citations and faithfulness alone do not establish exhaustive legal accuracy.

The judge uses versioned claim-extraction instructions preserving alternatives,
conditions and exceptions. The default claim extractor incorrectly turned
"only A or B" into the stronger claims "only A" and "only B" during a control
check. The custom extractor retains the legal logic; supported/unsupported
controls scored 1.0/0.0. NLI support judging remains RAGAS's standard procedure.
Generator and judge use the same model family, a limitation of the current
automated scores.

## Provider/quota history

The original Gemini 2.5 Flash and Flash Lite models both returned daily
free-tier quota errors (20 requests). The user requested another model, and
`gemini-3.1-flash-lite` was selected. Burst evaluation attempts hit rate limits;
requests are now paced to 16/minute across generation and judge calls. Daily
quota exhaustion aborts evaluation instead of completing invalid benchmark runs.
Failed attempts are retained as diagnostic MLflow runs, not quality evidence.

## Tracking, selection and gate

MLflow uses a local SQLite backend (`data/mlflow/tracking.db`), with per-variant
experiments and JSON/parameter/metric artifacts. Local tracking/model files are
ignored by Git. Portable measurement reports are kept under `reports/experiments/`.

Selection requires coverage-adjusted faithfulness of at least 0.75 and no
operational errors, then ranks citation hit, refusal accuracy, optional reference
accuracy, faithfulness, and latency. Full validation must have at least 20 scored
supported questions before registration. The registry packages the selected RAG
configuration, prompt, code and Chroma index; external Google weights and API
credentials are not stored.

`.github/workflows/quality.yml` runs a standard-library-only offline gate. It
requires the current pipeline fingerprint, every fixed question exactly once,
valid measured scores, no operational errors, at least 20 scored questions and
coverage-adjusted faithfulness >= 0.75. Stale/partial/low-quality measurements
fail, including the absence of a completed full validation. Live evaluation is
explicit and is not silently executed in CI.

## Measured comparison

Each comparison has 20 supported questions and four unsupported probes.

| Variant | Retrieval Hit@k | Expected-citation hit | Scored answers | Faithfulness, answered | Coverage-adjusted faithfulness | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-k 1 | 0.850 | 0.800 | 16 | 1.000 | 0.800 | 0 |
| Top-k 3 | 0.900 | 0.900 | 18 | 0.986 | 0.888 | 0 |
| Top-k 5 | 0.950 | 0.900 | 18 | 1.000 | 0.900 | 1 |
| **Top-k 8** | **0.950** | **0.950** | **19** | **0.982** | **0.933** | **0** |
| Opposite-language top-k 5 | 0.700 | 0.650 | 14 | 1.000 | 0.700 | 0 |

All four refusal probes were correctly refused in each comparison. Top-k 5 had
one recorded generation/scoring failure and was ineligible under the zero-error
selection rule. The opposite-language run exposes source-language availability
limits, including English-only notices and Article 1022, and misses the adjusted
faithfulness threshold. These are documented outcomes, not hidden failures.

**Top-k 8 was selected.** Full validation covers all 60 questions, reusing its
24 unchanged benchmark records and evaluating the remaining 36:

- Expected article retrieved: **55/56 (98.21%)** supported questions.
- Supported answers with expected citations: **52/56 (92.86%)**.
- Raw RAGAS faithfulness over 52 scored answers: **0.9936**.
- Coverage-adjusted faithfulness: **0.9226**, passing the 0.75 gate.
- Unsupported refusals: **4/4**; operational errors: **zero**.
- Arabic retrieval hit: **27/28**; English retrieval hit: **28/28**.
- Arabic expected-citation hit: **25/28**; English: **27/28**.

Raw faithfulness excludes abstentions; the adjusted value includes them as zero.
Evaluation `mean_seconds` includes queueing, the API pacer and judge calls, not
just serving latency. The same reviewed questions were used for configuration
selection and expanded validation; this is not an independent held-out test.

`reports/experiments/summary.json` identifies the five run IDs and full-validation
run `4194a08fc48840ac9768b8b741098860`. Detailed reports retain per-question
answers, true retrieved contexts, source evidence, token usage, reference evidence,
model/prompt versions and pipeline fingerprint.

## Registry and verification

`ArabicLegalDocumentQA` **version 1**, alias **`champion`**, was registered and
reloaded successfully. A fresh prediction returned the correct Article 54 repeal
citation and confirmed retrieval top-k 8. `reports/experiments/registry.json`
records the model URI and registration run. Registration does not deploy the
model automatically; the running service's baseline remains top-k 5.

**64 offline tests pass**, DVC reports up to date, and the measured gate passed
locally with all 60 questions present and 52 scored supported answers. The CI
workflow is added; remote GitHub execution occurs after pushing, not during this
local session.
