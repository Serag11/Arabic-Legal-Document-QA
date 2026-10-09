"""Small explicit live-API smoke evaluation; separate from offline tests/DVC."""

import argparse
import json
from pathlib import Path
import time

if __package__:
    from .generation import AnsweringService, GenerationError
else:
    from generation import AnsweringService, GenerationError


CASES = [
    ("contract-ar", "وفق المادة 147، هل يجوز لأحد المتعاقدين نقض العقد وحده؟", 147),
    ("contract-en", "Can one party unilaterally revoke a contract under Article 147?", 147),
    ("repeal-ar", "ما حالة المادة 54 في النص المقدم؟", 54),
    ("english-only-ar", "وفقا للمادة 1022، من يتحمل نفقات استعمال حق الارتفاق والمحافظة عليه؟", 1022),
    ("unsupported-medical-en", "What influenza vaccine dose is recommended for a five-year-old child?", None),
    ("unsupported-physics-ar", "كيف أحسب سرعة الإفلات من سطح المريخ باستخدام كتلته ونصف قطره؟", None),
]


def run_smoke(output):
    service = AnsweringService()
    records = []
    for identifier, question, number in CASES:
        start = time.perf_counter()
        try:
            result = service.ask(question)
            passed = (result["insufficient_evidence"] and not result["sources"]) if number is None else (
                not result["insufficient_evidence"] and f"Egyptian Civil Code, Article {number}" in result["sources"])
            record = {"id": identifier, "question": question, "expected_article_number": number,
                      "passed": passed, "result": result, "seconds": time.perf_counter() - start}
        except GenerationError as exc:
            record = {"id": identifier, "question": question, "passed": False, "error": str(exc)}
        records.append(record)
        print(json.dumps({"id": identifier, "passed": record["passed"],
                          "sources": record.get("result", {}).get("sources", []), "error": record.get("error")}, ensure_ascii=False))
    report = {"evaluation_type": "live-smoke-not-ragas", "passed": sum(r["passed"] for r in records),
              "total": len(records), "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/generation-smoke.json"))
    args = parser.parse_args()
    result = run_smoke(args.out)
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
