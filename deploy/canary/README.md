# Canary rollout

How a new image reaches users without a big-bang release. Everything the service needs
(code, prompt, config, ONNX model, index) is baked into the image, so a release is one image
and **rollback is switching back to the previous image**. Generation still calls Google, and
both tracks share that quota.

**Status:** the configuration here passes `docker compose config`, but the canary stack has
never been started (no Docker daemon was available) and has not been exercised under load.
Rehearse it once with `CANARY_IMAGE` equal to the stable image before relying on it.

## Setup

```bash
export STABLE_IMAGE=serag11/arabic-legal-document-qa@sha256:<digest currently in production>
export CANARY_IMAGE=serag11/arabic-legal-document-qa:sha-<candidate commit>
CANARY_PERCENT=0 docker compose -f deploy/canary/compose.canary.yaml up -d
```

`gateway` (nginx, port 8000) sends `CANARY_PERCENT`% of requests to `api-canary` by hashing the
request ID, passes streaming responses through unbuffered, and adds an `X-Served-By` header.
`X-Canary: always|never` pins a request, which is how you smoke-test the canary at 0%:

```bash
curl -s -H 'X-Canary: always' localhost:8000/health
```

To change the share, re-run `up -d` with a new `CANARY_PERCENT` (the gateway container is
recreated; in-flight requests finish or are retried by clients).
Prometheus (port 9090) scrapes both tracks with a `track` label and loads `monitoring/alerts.yml`.

## Before any traffic (release gate)

1. CI is green on the candidate commit: lint, offline tests, `quality_gate` (coverage-adjusted
   faithfulness >= 0.75 on the fixed 60 questions), `deployment_gate`, image smoke tests.
   If the prompt, retrieval, model or index changed, the full evaluation must have been re-run
   and its report committed; the gates refuse stale evaluations.
2. Image published as `sha-<commit>`; record the digest of the version it replaces.
3. Smoke test at 0%: `X-Canary: always` for `/health`, one Arabic `/ask`, one English `/ask`,
   one `/ask/stream`, and one repeal lookup (Article 54).

## Stages

A stage advances only when its hold time **and** minimum sample are both met and every gate
below passes. Minimum samples exist because alerts here need at least 20 answers to mean
anything; at a small share, create traffic yourself by replaying the evaluation questions
with `X-Canary: always` instead of waiting for users.

| Stage | Canary share | Hold at least | Canary answers needed |
| --- | ---: | --- | ---: |
| 0 smoke | 0% (header only) | until smoke tests pass | 5 |
| 1 | 5% | 30 min | 50 |
| 2 | 25% | 2 h | 100 |
| 3 | 50% | 4 h | 200 |
| 4 promote | 100% | keep the previous image stopped but available for 24 h | n/a |

Promote = make `CANARY_IMAGE` the new `STABLE_IMAGE`, set `CANARY_PERCENT=0`, then start the next
candidate from there. Run the canary with `CANARY_FAITHFULNESS_SAMPLE_RATE=1` (needs the
evaluation dependencies in the image; otherwise rely on the offline gate) so faithfulness is
judged on every canary answer rather than a rare sample.

## Gates (evaluate over the stage window, after the minimum sample)

Compare canary to stable (replace `1h` with the window):

| Signal | Gate | PromQL (per `track`) |
| --- | --- | --- |
| Provider errors | canary <= stable + 5 pp, and < 20% | `sum by (track) (increase(legal_qa_answers_total{outcome="provider_error"}[1h])) / sum by (track) (increase(legal_qa_answers_total[1h]))` |
| Grounding rejections | canary <= stable + 3 pp, and < 10% | same with `outcome="invalid_grounding"` |
| Refusals | canary <= stable + 5 pp, and < 25% | same with `outcome=~"model_refusal\|no_evidence"` |
| Latency | canary p95 <= 1.25x stable p95 | `histogram_quantile(0.95, sum by (le, track) (rate(legal_qa_stage_seconds_bucket{stage="total"}[1h])))` |
| Faithfulness | canary mean >= 0.75 (and not >0.05 below stable if sampled there) | `sum by (track) (increase(legal_qa_faithfulness_sum[1h])) / sum by (track) (increase(legal_qa_faithfulness_count[1h]))` |
| Query drift | no `LegalQAQueryDrift` alert | built-in alert |
| Cost per answer | canary tokens per answer <= 1.2x stable | `sum by (track) (increase(legal_qa_tokens_total[1h])) / sum by (track) (increase(legal_qa_answers_total[1h]))` |
| HTTP errors | no 5xx other than provider-status 502 | `sum by (track, status) (increase(legal_qa_http_requests_total{status=~"5.."}[1h]))` |

Provider errors depend on Google, not on the release: if stable shows the same rate, the
comparison is fair; if both are failing, pause the rollout rather than judging it.

## Rollback

Roll back immediately (no waiting for hold times) if any critical alert fires on the canary
(`LegalQAProviderErrorsHigh`, `LegalQAFaithfulnessLow`), grounding rejections exceed 10%, or a
smoke test fails:

```bash
CANARY_PERCENT=0 docker compose -f deploy/canary/compose.canary.yaml up -d gateway   # stops new canary traffic
docker compose -f deploy/canary/compose.canary.yaml stop api-canary                  # then stop the canary
```

Stable was never changed, so no further action is needed. Record the failing signal and the
Langfuse trace IDs (rejected candidates are kept in the `grounding` span) in the release notes.

## Limits of this setup

- Single host, one nginx process: it is a rollout mechanism, not high availability.
- The share is per request, not per user, so a user can see both versions across requests.
- Index or prompt changes are visible to users as a different answer for the same question;
  the stages above do not check answer correctness beyond the metrics listed.
