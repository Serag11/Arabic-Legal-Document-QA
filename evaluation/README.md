# Stage 2 bilingual retrieval questions — reviewed

`retrieval_questions.jsonl` contains **30 paired examples**: **28 supported
pairs** (56 language-specific queries, 27 distinct target articles) and **2
out-of-domain pairs** (4 language-specific queries). Each pair asks the same
question in Arabic and English and shares its article-level target.

**Status: manually reviewed and approved by the user for Stage 2 acceptance.**
`review.json` records that confirmation and the exact evaluation-file hash.
Measured results are in `reports/retrieval-baseline.md` and
`reports/retrieval-results.json`. The latter retains its historical
`draft_ground_truth` flag because the run preceded approval. Deferred Stage 1
corpus-audit tasks remain recorded separately in `PLAN.md`.

## Source and coverage

Source: `data/egyptian_civil_code_articles.jsonl`. Questions concern what that
source says; they do not certify present-day law or account for later amendments.
All supported examples initially use a single target article. These targets are
seed relevance labels, not exhaustive judgments that every other article is
irrelevant; review may identify additional relevant articles.

Coverage includes introductory rules, contract formation and effects, tort,
unjust enrichment, damages, prescription, sale, leases, contracts for work,
ownership and co-ownership, servitudes, mortgages, and privileges. Most questions
paraphrase provisions without stating an article number. Article 147 has two
separate pairs for binding force and exceptional hardship.

- **558:** lease definition; the source's bold Arabic body is preserved.
- **563 and 1065:** long English articles, with evidence for specific provisions.
- **1143:** longest English article in this corpus (2,141 characters); the query
  targets its opening provision. These examples can help detect truncation and
  source-mapping errors, but do not establish coverage of every paragraph.
- **54 and 389:** explicit article-number queries anchored to the exact
  `repeal_note`. The target is the requested article record, not every article
  mentioned in the notice's range. Neither pair invents repealed provision text
  or an unrecorded decree number/date. Returned evidence should identify repeal
  status and preserve the notice.
- **1022:** explicit article-number query in both languages. Its Arabic source
  is absent, not repealed; the shared reference is the available English text.
  Arabic question wording is a query translation, not fabricated Arabic source
  evidence. Keep this cross-language exception visible in results.

## Record contract

Each UTF-8 JSONL line is one object:

| Field | Meaning |
| --- | --- |
| `id` | Unique, stable pair identifier; retain it across experiment runs. |
| `question_ar`, `question_en` | Nonempty equivalent questions. |
| `expected_article_numbers` | Integer article-level relevance labels. |
| `expected_citations` | Exact corpus citation strings, corresponding in order to the article numbers. |
| `category` | Coverage label; `unsupported` is reserved for out-of-domain probes. |
| `rationale` | Brief explanation of why the source supports the question, or why it is out of domain. |
| `reference_field` | Supported cases: `text_en` or `repeal_note` in the target corpus record. |
| `reference_excerpt` | Supported cases: an exact, unnormalized substring of that source field. |

English evidence is shared by both question languages, allowing source checks
without manufacturing a second quotation. Source spellings and extraction
artifacts are intentionally retained in excerpts. Unsupported records omit
`reference_field` and `reference_excerpt` and have empty expected lists.

## Evaluation use

1. Validate the corpus before building an index. Review and approve the paired
   questions before treating measurements as an approved baseline.
2. Run both language variants with the same retrieval configuration. Preserve
   ranked passages, article numbers, citations, languages, and repeal flags.
   Deduplicate language/chunk hits by article before article-level scoring.
3. Report Hit@k, Recall@k, and MRR separately for Arabic and English over the
   **28 supported pairs only**. Hit@k is whether any target appears in the first
   k distinct articles; Recall@k is the fraction of target articles retrieved;
   reciprocal rank is the inverse rank of the first target, or zero if absent.
   With one target per pair, Hit@k and Recall@k coincide. For bounded rankings,
   label the reciprocal-rank metric MRR@k and state k.
4. Define bilingual agreement explicitly. One useful measure is the fraction of
   supported pairs for which Arabic and English have the same Hit@k outcome;
   also report both-hit, Arabic-only, English-only, and neither-hit counts so
   agreement on failure is visible. If comparing result-set overlap, name and
   define that separate measure.
5. Break out explicit-number cases (54, 389, 1022) from semantic/paraphrase cases.
   Record whether direct lookup, semantic search, or a hybrid route supplied each
   result; direct-number success alone does not demonstrate embedding quality.
6. Record the corpus hash, evaluation-file hash/version, embedding model/version,
   normalization, chunk size and units, overlap, filters, lookup route, and k with
   per-query results for later experiment comparisons. This directory supplies
   questions and evidence, not a retrieval runner or a measured baseline.

### Unsupported probes are not empty-retrieval assertions

The two `unsupported` pairs have no relevant article labels. **Semantic
nearest-neighbor search can still return ranked articles for these questions**;
empty expected lists do not assert that retrieval must return nothing. Keep
these cases out of Recall, Hit, MRR, and supported bilingual-agreement denominators.
They are later abstention/insufficient-evidence and drift probes, to be evaluated
with a defined relevance/abstention policy and, eventually, generated answers.
They do not by themselves establish a contract-law versus family-law shift
benchmark or a calibrated rejection threshold.

## Structural verification

Before use, parse every JSONL line and check unique IDs, nonempty questions,
integer expected article numbers, aligned citations matching the source records,
and exact evidence substrings in the indicated source fields. Require at least
20 supported pairs; require unsupported pairs to have empty expected lists and
no reference fields. These checks were run when creating this draft. They do
not replace manual legal review, retrieval measurement, or later answer-quality
evaluation.
