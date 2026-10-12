# Load test and streaming check (Stage 7)

Tool: Locust 2.46 with `loadtest/locustfile.py` (random fixed evaluation questions;
3:1 `/ask` to `/ask/stream`; 1–3 s think time). Raw CSV/HTML in `reports/loadtest/`.
Machine: the development workstation (6 CPU threads, CPU embeddings).

## 50 concurrent users, simulated provider (1.5 s delay, retrieval and API real)

| Design | `/ask` median | `/ask` p99 | Throughput | Failures |
| --- | ---: | ---: | ---: | ---: |
| Retrieval-only lock (current), 90 s | 1.6 s | 2.0 s | ~13.4 req/s (914 + 277 stream) | 0 / 1,191 |
| One lock around the whole request (previous), 45 s | 27 s | 42 s | 0.5 req/s | 0 / 21 |

- Service overhead above the 1.5 s stub is about 100 ms at 50 users, so the app is not
  the bottleneck; the provider is. Retrieval takes ~40–70 ms and is still serialized, so
  the ceiling is roughly 15–20 requests/s before retrieval queues.
- Streaming: time-to-first-delta median 1.6 s, total 1.7 s (`stream time-to-first-delta`
  and `stream total duration` rows; Locust itself only times streams until headers arrive).
- The previous design held a global lock during the provider call, which made the service
  effectively single-request. That lock is now limited to retrieval.

## Live Gemini (`gemini-3.1-flash-lite`), 8 users, 300 s

Not a capacity result. Google returned HTTP 429/503 for most calls and the service
returns these as HTTP 502 without retry (305 of 363 `/ask`, 91 of 107 stream calls).
Of the 74 requests that were answered: 71 cited the expected article, 1 was rejected
as `invalid_source_grounding`, 2 were model refusals. Successful streams: time to first
delta 13 s median (8.6–23 s), total 14 s. Single trivial prompts ("Say hi.") took
14–26 s directly from the SDK at that time, so provider latency, not streaming or the
service, dominates. Retrieval was 0.04–0.07 s in Langfuse spans.

## Prompt change (not a formal evaluation)

The answer prompt now requires each evidence quote to be one contiguous excerpt (the
cause of three `invalid_source_grounding` refusals: quotes joined with " ... ").
Re-running the six previously failing questions once each: 5 pass with the expected
article (s2-010-en, s2-017-ar, s2-020-en, s2-022-ar, s2-022-en). s2-024-ar still
refuses (Article 1065 retrieval miss). s2-017-ar was a wrong-rule selection rather than
a quoting error and may simply vary between runs. The full 60-question evaluation and
the quality/deployment gates have not been re-run and remain stale.
