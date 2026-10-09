# Stage 3 — Google-backed grounded answers

## Implementation

Stage 2 was accepted after the user's manual review and committed as `c88de7c`.
Stage 3 uses the same local E5/Chroma retriever with Google's
**`gemini-2.5-flash`**, confirmed available through the configured account.

The root `.env` supplies `GOOGLE_API_KEY`. File values take precedence over the
environment fallback; only this key is read for client authentication. The key
is never included in configuration, prompts, reports, or Git. `.env.example`
contains a blank placeholder. No generation model weights are downloaded.

- `src/rag.py ask`: end-to-end CLI; existing preparation usage is preserved.
- `src/generation.py`: provider adapter, context construction, source validation,
  localized insufficient-evidence handling, and source-exception notes.
- `config/generation.json`: model, temperature zero, output limit 2,048 tokens,
  thinking budget zero, 60-second request timeout, top five retrieved articles,
  and 24,000-character context budget.
- `prompts/answer_v1.txt`: bilingual source-only instructions, repeal/missing-text
  handling, abstention, and quotation requirements.
- `src/evaluate_generation.py`: explicit six-case live smoke runner.

Whole source passages are supplied as JSON data, using source-faithful text,
not normalized embedding text. The context cap skips whole passages that cannot
fit instead of slicing legal provisions. The Google request uses native
`response_json_schema`; the older `response_schema` conversion rejected
Pydantic's `additionalProperties` fields on this endpoint during setup.

## Response behavior

Supported answers return article citations and exact supporting quotes. The
application rejects article numbers outside the supplied context, inline article
references inconsistent with the declared citations, missing evidence, and
non-source quotations. Every cited article needs an exact source excerpt.
Arabic answers based on Article 1022's English-only source receive an explicit
source-language note; English quotes are preserved rather than fabricated in Arabic.

No context, a model-declared unsupported question, or failed source validation
returns a localized insufficient-evidence answer with empty sources/evidence.
Google API/invalid-response failures raise a sanitized operational error and a
nonzero CLI exit status; they are not misrepresented as a legal answer.

Each answer includes generation/prompt settings, prompt hash, corpus/chunk
hashes, the pinned retrieval configuration, filters, language, effective context
size, response model version, and provider token usage. The root API key is absent.

## Verification

**46 offline tests passed.** They use fake provider/retrieval boundaries and
exercise supported output, empty questions, unsupported responses, citation and
quote rejection, exact source text, context budgets, repeal status, the missing
Arabic exception, `.env` loading, native-schema configuration, and error redaction.
The full DVC pipeline includes these tests but does not call Google's API.

**Six live smoke cases passed**, recorded in `reports/generation-smoke.json`:

| Case | Expected result |
| --- | --- |
| Arabic contract question | Supported answer citing Article 147 |
| English contract question | Supported answer citing Article 147 |
| Arabic repeal query | Source repeal notice, Article 54 |
| Arabic question about English-only source | Article 1022, English evidence, explicit source-language note |
| English medical dosing question | Insufficient evidence, no sources |
| Arabic planetary-physics question | Insufficient evidence, no sources |

The source-identity smoke cases intentionally use explicit article numbers,
isolating generation from known semantic retrieval failures. These tests check
source identity and abstention, **not a RAGAS faithfulness score or exhaustive
legal correctness**. Citation/excerpt validation cannot independently prove
that every statement follows from the quoted evidence. Wider answer evaluation
remains in the later evaluation stages.

## Usage

```bash
uv run python src/rag.py ask "وفق المادة 147، هل يجوز لأحد المتعاقدين نقض العقد وحده؟"
uv run python src/rag.py ask "What is a lease under Article 558?" --language en
uv run python src/evaluate_generation.py
```

The smoke runner makes six Google generation calls and writes its report;
it is intentionally separate from offline tests and reproducible local indexing.
FastAPI/service packaging is Stage 4. Deferred Stage 1 source QA and Drive upload
remain tracked in `PLAN.md`.
