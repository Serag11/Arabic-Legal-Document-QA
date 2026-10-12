"""The gate must reject low scores, missing coverage, stale results, and errors."""

from copy import deepcopy

import pytest

from src.quality_gate import check_report
from src.experiments import retrieval_metrics, summarize


def good_report():
    return {"pipeline_fingerprint": "current", "records": [
        {"id": str(i), "expected_article_numbers": [147], "faithfulness": 0.8}
        for i in range(20)
    ]}


def test_gate_accepts_threshold_and_sufficient_sample_size():
    report = good_report()
    assert check_report(report, "current", [str(i) for i in range(20)])["scored_questions"] == 20


@pytest.mark.parametrize("change", ["low", "nan", "missing", "error", "stale", "duplicate"])
def test_gate_fails_for_unusable_evidence(change):
    report = deepcopy(good_report())
    if change == "low":
        for r in report["records"]:
            r["faithfulness"] = 0.7
    elif change == "nan":
        report["records"][0]["faithfulness"] = float("nan")
    elif change == "missing":
        report["records"][0]["faithfulness"] = None
    elif change == "error":
        report["records"][0]["error"] = "Scoring failed"
    elif change == "stale":
        report["pipeline_fingerprint"] = "old"
    else:
        report["records"][0]["id"] = "1"
    with pytest.raises(ValueError):
        check_report(report, "current", [str(i) for i in range(20)])


def test_abstention_and_operational_failure_do_not_inflate_quality():
    base = {"language": "en", "expected_article_numbers": [147], "expected_citations": ["Article 147"],
            "seconds": 1, **retrieval_metrics([147], [147])}
    records = [dict(base, faithfulness=1.0, answer_accuracy=1.0, result={"sources": ["Article 147"], "insufficient_evidence": False}),
               dict(base, faithfulness=None, answer_accuracy=None, result={"sources": [], "insufficient_evidence": True})]
    summary = summarize(records)
    assert summary["faithfulness_answered"] == 1.0
    assert summary["faithfulness_coverage_adjusted"] == 0.5
    assert summary["supported_answer_coverage"] == summary["expected_citation_hit"] == 0.5


def test_judge_calls_retry_transient_failures_but_not_daily_quota(monkeypatch):
    import asyncio

    from src.experiments import with_retries

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("src.experiments.asyncio.sleep", fake_sleep)
    outcomes = [RuntimeError("503"), RuntimeError("503"), {"faithfulness": 1.0}]

    async def flaky():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert asyncio.run(with_retries(flaky)) == {"faithfulness": 1.0} and sleeps == [5, 15]

    async def always_failing():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        asyncio.run(with_retries(always_failing))

    daily = RuntimeError("quota")
    daily.daily_quota = True
    calls = []

    async def exhausted():
        calls.append(1)
        raise daily

    with pytest.raises(RuntimeError, match="quota"):
        asyncio.run(with_retries(exhausted))
    assert len(calls) == 1
