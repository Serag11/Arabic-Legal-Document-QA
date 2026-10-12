# Monitoring, drift and cost tracking (Stage 7)

`GET /metrics` exposes Prometheus metrics. `compose.monitoring.yaml` adds Prometheus (alert
rules in `monitoring/alerts.yml`) and Grafana with a provisioned dashboard.

```bash
echo 'GRAFANA_ADMIN_PASSWORD=<choose one>' >> .env
docker compose -f compose.yaml -f compose.monitoring.yaml up
# Grafana http://localhost:3000 (dashboard "Arabic Legal Document QA"), Prometheus http://localhost:9090
```

## What is measured

| Metric | Meaning |
| --- | --- |
| `legal_qa_answers_total{outcome}` | grounded, model_refusal, no_evidence, invalid_grounding, provider_error |
| `legal_qa_stage_seconds{stage}` | retrieval, generation, total latency histograms |
| `legal_qa_tokens_total{type}` / `legal_qa_estimated_cost_usd_total` | prompt, completion, thinking tokens; cost estimate |
| `legal_qa_query_centroid_cosine` (+ baseline mean/std gauges) | embedding drift of live queries |
| `legal_qa_top1_similarity` | cosine similarity of the best retrieved article |
| `legal_qa_faithfulness` | sampled online RAGAS faithfulness |
| `legal_qa_provider_errors_total{status}`, `legal_qa_http_*` | provider and HTTP errors |

**Cost** is an estimate: tokens x the per-million prices in `config/generation.json`
(0.30 input / 2.50 output USD for `gemini-3.5-flash-lite`, thinking billed as output). Those prices come
from third-party price trackers, not Google's pricing page (for the previous 3.1 model such sources
disagreed with each other), so verify them before relying on the cost figures.

**Drift**: `config/drift_baseline.json` holds the centroid of the 60 evaluation-question
embeddings (`uv run python -m src.drift` rebuilds it; it is bound to the embedding model and
revision and disabled if they differ). Each semantic query records its cosine to that centroid.

**Faithfulness**: set `ONLINE_FAITHFULNESS_SAMPLE_RATE` (0-1, default off) to score that
fraction of grounded answers with the same RAGAS judge used in evaluation, in a background
worker with a bounded backlog. It needs the evaluation dependencies (`uv sync --group
evaluation`), which the slim runtime image lacks, so sampling disables itself there. Scores
are attached to the Langfuse trace as `faithfulness`.

## Alerts (`monitoring/alerts.yml`, each unit-tested with `promtool test rules`)

| Alert | Fires when | Basis |
| --- | --- | --- |
| GroundingRejectionHigh | >10% of answers discarded (30 min, >=20 answers) | 5% before, ~1-2% after the prompt change |
| ProviderErrorsHigh | >20% provider failures (10 min) | live run saw mostly 429/503 |
| LatencyP95High | p95 total > 40 s | Gemini took 14-26 s per call |
| QueryDrift | mean query cosine 3 sigma below the reference mean (1 h) | see below |
| RefusalRateHigh | >25% refusals (1 h) | 4/60 refusal probes in evaluation |
| FaithfulnessLow | sampled faithfulness < 0.75 (6 h, >=10 samples) | CI gate threshold |
| EstimatedCostHigh | > 5 USD in 24 h | example budget, adjust |

Thresholds are starting points, not tuned on production traffic.

## Drift check on unseen questions (`uv run python -m src.drift` baseline: mean 0.878, std 0.018)

| Query set | mean cosine to centroid |
| --- | ---: |
| New civil-law questions, English (6) | 0.859 |
| New civil-law questions, Arabic (4) | 0.855 |
| Family / criminal / personal-status questions (3) | 0.852 |
| Off-topic (weather, poem, cooking) (3) | 0.799 |

The alert threshold is 0.823. It separates off-topic traffic (0.80) from in-domain questions
(0.855+), with only ~0.03 margin. It does **not** separate questions on legal subjects the
Code does not cover (0.852), because they look like civil-law questions in embedding space;
RefusalRateHigh is the signal for that shift. The reference mean is also optimistic, since the
centroid was fit on the same 60 questions; unseen in-domain questions score ~0.02 lower. Small
samples (10-13 queries), so treat the numbers as indicative.

## Verified and not verified

- Verified: 104 offline tests; alert rules and all 7 alert scenarios with promtool (v3.5.0);
  every dashboard query parses as PromQL; dashboard and alert metric names match the exported
  metrics (test); a live service with Langfuse and 100% faithfulness sampling recorded outcomes,
  stage latency, tokens (4,914 prompt / 460 completion), cost (~0.0019 USD), drift and top-1
  similarity for 4 requests; 2 grounded answers were scored 1.0 by the judge.
- Not verified: the Prometheus and Grafana containers were not started (Docker daemon was off
  and the disk had 4 GB free), so the compose file was only checked with `docker compose
  config`, and the dashboard was never rendered in Grafana.
