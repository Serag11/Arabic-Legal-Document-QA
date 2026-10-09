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
