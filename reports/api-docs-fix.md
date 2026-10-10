# `/ask` Swagger example and container semantic-retrieval fix

## Reproduced defects

The automatically generated request example used `question: "string"`, Article
0 and placeholder hierarchy strings, applying every optional filter together.
Those filters cannot match this corpus. The request also exposed a real HTTP 500:
semantic retrieval initialized E5 and the Hugging Face client attempted to read
`/models/huggingface/token`, which the non-root container user could not read.
Earlier container checks used direct article-number lookup, which bypassed the
embedding initialization path.

## Changes

- Named OpenAPI examples for Arabic semantic questions, English-source questions,
  and repeal lookup; the default example omits filters.
- Positive-integer validation for article-number filters (Article 0 returns 422).
- Anonymous `token=False` loading for the public pinned E5 model/tokenizer.
- Mount only the Hugging Face `hub` cache rather than personal token files.
- Initialize the embedding encoder during API startup before reporting readiness.

## Verification

**69 offline tests passed**, including Swagger request validation, encoder
startup loading once, and an unreadable-token regression using the real Hub
authentication helper. The container was rebuilt and restarted.

The actual default `/docs` request:

```json
{"question":"ما تعريف عقد الإيجار؟","language":"ar"}
```

returned **HTTP 200**, an Arabic lease definition, and
`Egyptian Civil Code, Article 558`. Article 0 returned **422**; a valid positive
filter with no match returned **200** with `no_retrieved_evidence`, not a server
error. `/health` reports 2,242 passages and 1,149 articles after model loading.

The user subsequently reran Stage 5 evaluation for this loader change. The
current pipeline fingerprint matches the fresh results, and the quality gate
passes at 0.9250 coverage-adjusted faithfulness with 52 scored supported answers.
Top-k 5 is selected and registered as `ArabicLegalDocumentQA` version 2, alias
`champion`. Results were refreshed rather than reassigned to changed code.
