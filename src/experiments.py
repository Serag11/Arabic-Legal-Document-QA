"""Reproducible bilingual evaluation, live RAGAS judging, and local MLflow runs."""

import argparse
import asyncio
import importlib.metadata
import json
import math
import logging
from threading import Lock
from pathlib import Path
import time

from .generation import AnsweringService, GoogleGenerator, ROOT, build_context, load_generation_config, load_google_api_key
from .rag import DEFAULT_CORPUS, load_articles
from .retrieval import Retriever, sha256, write_json
from .quality_gate import fingerprint


DEFAULT_QUESTIONS = ROOT / "evaluation/answer_questions.jsonl"
DEFAULT_EXPERIMENTS = ROOT / "config/experiments.json"
CLAIM_PROMPT = ROOT / "prompts/faithfulness_claims_v1.txt"
def pipeline_fingerprint(root=ROOT):
    return fingerprint(root)


def prepare_questions(output=DEFAULT_QUESTIONS):
    pairs_path = ROOT / "evaluation/retrieval_questions.jsonl"
    review = json.loads((ROOT / "evaluation/review.json").read_text())
    if review["status"] != "approved" or review["evaluation_sha256"] != sha256(pairs_path):
        raise ValueError("Reviewed retrieval questions have changed; review approval must match")
    articles = {a["article_number"]: a for a in load_articles(ROOT / DEFAULT_CORPUS)}
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    rows = []
    for pair in pairs:
        for language in ("ar", "en"):
            numbers = pair["expected_article_numbers"]
            contexts = []
            for number in numbers:
                article = articles[number]
                field = "repeal_note" if article["is_repealed"] else ("ar_text" if language == "ar" and article["ar_text"] else "text_en")
                contexts.append(article[field])
            rows.append({
                "id": f"{pair['id']}-{language}", "pair_id": pair["id"], "language": language,
                "question": pair[f"question_{language}"], "category": pair["category"],
                "expected_article_numbers": numbers, "expected_citations": pair["expected_citations"],
                "reference_answer": pair.get("reference_excerpt", ""),
                "reference_contexts": contexts,
                "reference_answer_kind": "reviewed_verbatim_source_excerpt" if numbers else "unsupported_probe",
            })
    if len(rows) < 50 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Need at least 50 distinct language-specific evaluation records")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"question_count": len(rows), "pair_count": len(pairs), "supported_count": sum(bool(r["expected_article_numbers"]) for r in rows),
                "questions_sha256": sha256(output), "reviewed_pairs_sha256": sha256(pairs_path),
                "manual_review": review, "corpus_sha256": sha256(ROOT / DEFAULT_CORPUS),
                "note": "60 language-specific questions derived from 30 user-reviewed pairs; reference answers are their reviewed source excerpts, not invented translations."}
    write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


class ApiPacer:
    """Shared smooth limiter for generation and every RAGAS provider call."""
    def __init__(self, rpm):
        self.interval = 60 / rpm
        self.next_call = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_call)
            self.next_call = scheduled + self.interval
        time.sleep(max(0, scheduled - now))


class PacedGenerator(GoogleGenerator):
    def __init__(self, config, pacer):
        super().__init__(config)
        self.pacer = pacer

    def generate(self, *args, **kwargs):
        self.pacer.wait()
        return super().generate(*args, **kwargs)


class RagasJudge:
    def __init__(self, model, pacer=None, score_accuracy=False):
        import instructor
        from google import genai
        from ragas.llms import InstructorLLM
        from ragas.metrics.collections import AnswerAccuracy, Faithfulness

        logging.getLogger("instructor").setLevel(logging.CRITICAL)
        logging.getLogger("ragas").setLevel(logging.CRITICAL)

        self.client = genai.Client(api_key=load_google_api_key(), vertexai=False)
        adapted = instructor.from_genai(self.client, use_async=True, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)
        model_settings = {"thinking_config": {"thinking_level": "minimal"}} if model.startswith("gemini-3") else {"thinking_config": {"thinking_budget": 0}}
        class PacedLLM(InstructorLLM):
            async def agenerate(self, *args, **kwargs):
                if pacer:
                    await asyncio.to_thread(pacer.wait)
                return await super().agenerate(*args, **kwargs)

        self.llm = PacedLLM(client=adapted, model=model, provider="google", temperature=0, max_retries=1,
                                 generation_config=model_settings)
        self.faithfulness = Faithfulness(llm=self.llm)
        self.faithfulness.statement_generator_prompt.instruction = CLAIM_PROMPT.read_text()
        self.answer_accuracy = AnswerAccuracy(llm=self.llm, max_retries=1) if score_accuracy else None

    async def score(self, row, answer, context):
        faithful = await self.faithfulness.ascore(user_input=row["question"], response=answer,
                                                retrieved_contexts=[json.dumps(p, ensure_ascii=False) for p in context])
        values = {"faithfulness": float(faithful.value), "answer_accuracy": None}
        if self.answer_accuracy:
            accuracy = await self.answer_accuracy.ascore(user_input=row["question"], response=answer, reference=row["reference_answer"])
            values["answer_accuracy"] = float(accuracy.value)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values() if value is not None):
            raise ValueError("Non-finite or invalid RAGAS metric")
        return values

    async def sanity_check(self):
        contexts = ["A contract may be revoked only by mutual consent or for reasons provided by law."]
        scores = []
        for answer in (contexts[0], "One party may always revoke a contract without the other party's consent."):
            result = await self.faithfulness.ascore(user_input="Can one party revoke a contract?", response=answer, retrieved_contexts=contexts)
            scores.append(float(result.value))
        if scores != [1.0, 0.0]:
            raise ValueError(f"Judge sanity check failed: expected [1, 0], got {scores}")
        return {"supported_control": scores[0], "unsupported_control": scores[1]}

    async def close(self):
        await self.client.aio.aclose()
        self.client.close()


def retrieval_metrics(numbers, expected):
    ranks = [numbers.index(n) + 1 for n in expected if n in numbers]
    return {"retrieval_hit": int(bool(ranks)), "retrieval_recall": len(ranks) / len(expected) if expected else None,
            "retrieval_rr": 1 / min(ranks) if ranks else 0}


def summarize(records):
    supported = [r for r in records if r["expected_article_numbers"]]
    unsupported = [r for r in records if not r["expected_article_numbers"]]
    scored = [r for r in supported if r.get("faithfulness") is not None]
    mean = lambda values: sum(values) / len(values) if values else 0.0
    metrics = {
        "supported_count": len(supported), "scored_count": len(scored), "unsupported_count": len(unsupported),
        "retrieval_hit": mean([r["retrieval_hit"] for r in supported]),
        "retrieval_recall": mean([r["retrieval_recall"] for r in supported]),
        "retrieval_mrr": mean([r["retrieval_rr"] for r in supported]),
        "supported_answer_coverage": mean([int(not r.get("result", {}).get("insufficient_evidence", True)) for r in supported]),
        "expected_citation_hit": mean([int(bool(set(r.get("result", {}).get("sources", [])) & set(r["expected_citations"]))) for r in supported]),
        "faithfulness_answered": mean([r["faithfulness"] for r in scored]) if scored else None,
        "faithfulness_coverage_adjusted": mean([r.get("faithfulness") or 0.0 for r in supported]),
        "answer_accuracy_coverage_adjusted": mean([r.get("answer_accuracy") or 0.0 for r in supported]) if any(r.get("answer_accuracy") is not None for r in supported) else None,
        "unsupported_refusal_accuracy": mean([int(r.get("result", {}).get("insufficient_evidence", False) and not r.get("result", {}).get("sources")) for r in unsupported]),
        "mean_seconds": mean([r["seconds"] for r in records]),
        "error_count": sum("error" in r for r in records),
    }
    for language in ("ar", "en"):
        group = [r for r in supported if r["language"] == language]
        metrics[f"retrieval_hit_{language}"] = mean([r["retrieval_hit"] for r in group])
        metrics[f"faithfulness_{language}"] = mean([r.get("faithfulness") or 0.0 for r in group])
        metrics[f"citation_hit_{language}"] = mean([int(bool(set(r.get("result", {}).get("sources", [])) & set(r["expected_citations"]))) for r in group])
    return metrics


class PreparedRetriever:
    def __init__(self, base, hits):
        self.manifest = base.manifest
        self.hits = hits

    def search(self, question, **kwargs):
        return self.hits[question]


def daily_quota_exhausted(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "daily_quota", False) or (getattr(exc, "code", None) == 429 and "PerDay" in str(exc)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def evaluate_variant(base, judge, rows, variant, config, output, reused=None, pacer=None):
    initial_fingerprint = pipeline_fingerprint()
    hits = {}
    for row in rows:
        filters = {"language": "en" if row["language"] == "ar" else "ar"} if variant["document_language"] == "opposite" else {}
        hits[row["question"]] = base.search(row["question"], k=variant["top_k"], filters=filters)
    generation = dict(load_generation_config(), retrieval_top_k=variant["top_k"])
    service = AnsweringService(PreparedRetriever(base, hits), PacedGenerator(generation, pacer) if pacer else GoogleGenerator(generation), generation)
    semaphore = asyncio.Semaphore(config["concurrency"])

    async def one(row):
        async with semaphore:
            start = time.perf_counter()
            found = hits[row["question"]]
            context = build_context(found, generation["max_context_characters"])
            record = dict(row, retrieved_article_numbers=[r["article_number"] for r in found],
                          retrieved_contexts=context, faithfulness=None, answer_accuracy=None)
            record.update(retrieval_metrics(record["retrieved_article_numbers"], row["expected_article_numbers"]))
            try:
                result = await asyncio.to_thread(service.ask, row["question"], language=row["language"])
                record["result"] = result
                if row["expected_article_numbers"] and not result["insufficient_evidence"]:
                    record.update(await judge.score(row, result["answer"], context))
            except Exception as exc:
                # Raw judge/provider exceptions may contain credentials or requests.
                # Record failure explicitly and penalize missing scores, never fabricate.
                record["error"] = f"Generation or RAGAS scoring failed ({type(exc).__name__})"
                if daily_quota_exhausted(exc):
                    raise RuntimeError("Daily Google quota exhausted; experiment aborted without a quality score") from None
            record["seconds"] = time.perf_counter() - start
            return record

    records = list(reused or [])
    existing = {r["id"] for r in records}
    tasks = [asyncio.create_task(one(row)) for row in rows if row["id"] not in existing]
    try:
        for task in asyncio.as_completed(tasks):
            record = await task
            records.append(record)
            write_json(output.with_suffix(".partial.json"), {"variant": variant, "records": records})
            print(f"{variant['name']}: {len(records)}/{len(rows)}", flush=True)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        write_json(output.with_suffix(".partial.json"), {"variant": variant, "status": "aborted", "records": records})
        raise
    records.sort(key=lambda r: r["id"])
    if pipeline_fingerprint() != initial_fingerprint:
        raise RuntimeError("Pipeline changed during live evaluation; scores cannot be accepted")
    return {"variant": variant, "metrics": summarize(records), "records": records,
            "generation_config": generation, "retrieval_config": base.manifest["config"],
            "pipeline_fingerprint": initial_fingerprint, "questions_sha256": sha256(DEFAULT_QUESTIONS),
            "reused_record_count": len(reused or []),
            "corpus_sha256": base.manifest["corpus_sha256"],
            "judge_model": config["judge_model"], "judge_prompt_version": config["judge_prompt_version"],
            "judge_prompt_sha256": sha256(CLAIM_PROMPT),
            "packages": {name: importlib.metadata.version(name) for name in ("ragas", "instructor", "google-genai", "mlflow")}}


async def run_experiments(config_path=DEFAULT_EXPERIMENTS):
    import mlflow

    prepare_questions()
    config = json.loads(config_path.read_text())
    rows = [json.loads(line) for line in DEFAULT_QUESTIONS.read_text().splitlines()]
    benchmark = [r for r in rows if r["pair_id"] in config["benchmark_pair_ids"]]
    tracking = ROOT / "data/mlflow"
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking / 'tracking.db'}")
    base = Retriever(ROOT / "data/retrieval/chroma")
    pacer = ApiPacer(config["requests_per_minute"])
    judge = RagasJudge(config["judge_model"], pacer, config["score_answer_accuracy"])
    outputs = []
    try:
        sanity = await judge.sanity_check()
        for variant in config["variants"]:
            output = ROOT / "reports/experiments" / f"{variant['name']}.json"
            mlflow.set_experiment(f"CivilCode/{variant['name']}")
            with mlflow.start_run(run_name=variant["name"]) as run:
                mlflow.log_params({
                    "chunk_size": base.config["max_tokens"], "chunk_size_units": "tokens_including_prefix_and_special_tokens",
                    "chunk_strategy": base.config["chunk_strategy"], "overlap": base.config["overlap"],
                    "embedding_model": base.config["model_id"], "embedding_revision": base.config["model_revision"],
                    "retrieval_top_k": variant["top_k"], "document_language": variant["document_language"],
                    "generation_model": load_generation_config()["model"], "judge_model": config["judge_model"],
                    "question_count": len(benchmark), "pipeline_fingerprint": pipeline_fingerprint(),
                    "judge_prompt_version": config["judge_prompt_version"],
                })
                cached = json.loads(output.read_text()) if output.exists() else None
                if cached and cached.get("pipeline_fingerprint") == pipeline_fingerprint() and not cached["metrics"]["error_count"]:
                    report = cached
                    report["cached_results_reused"] = True
                else:
                    report = await evaluate_variant(base, judge, benchmark, variant, config, output, pacer=pacer)
                report["mlflow_run_id"] = run.info.run_id
                report["judge_sanity_check"] = sanity
                write_json(output, report)
                mlflow.log_metrics({key: value for key, value in report["metrics"].items() if value is not None})
                mlflow.log_artifact(str(output), artifact_path="evaluation")
                mlflow.log_artifact(str(DEFAULT_QUESTIONS), artifact_path="evaluation")
                outputs.append({"file": str(output.relative_to(ROOT)), "variant": variant, "metrics": report["metrics"], "mlflow_run_id": run.info.run_id})
    finally:
        await judge.close()
    eligible = [r for r in outputs if r["metrics"]["scored_count"] > 0
                and r["metrics"]["faithfulness_coverage_adjusted"] >= config["faithfulness_threshold"] and not r["metrics"]["error_count"]]
    best = max(eligible, key=lambda r: (r["metrics"]["expected_citation_hit"], r["metrics"]["unsupported_refusal_accuracy"],
                                      r["metrics"]["answer_accuracy_coverage_adjusted"] or 0.0, r["metrics"]["faithfulness_coverage_adjusted"], -r["metrics"]["mean_seconds"])) if eligible else None
    summary = {"experiments": outputs, "selected": best, "pipeline_fingerprint": pipeline_fingerprint(),
               "benchmark_question_count": len(benchmark), "full_question_count": len(rows),
               "faithfulness_threshold": config["faithfulness_threshold"], "minimum_scored_questions": config["minimum_scored_questions"],
               "selection_rule": "Meet coverage-adjusted faithfulness gate and zero operational errors; maximize expected-citation hit, unsupported refusal accuracy, answer accuracy, faithfulness, then minimize latency."}
    write_json(ROOT / "reports/experiments/summary.json", summary)
    print(json.dumps(summary, indent=2))
    if best is None:
        raise RuntimeError("No experiment met the quality gate; do not register a champion")
    # Validate the chosen configuration on every reviewed language-specific
    # question, including benchmark cases and the remaining corpus scenarios.
    judge = RagasJudge(config["judge_model"], pacer, config["score_answer_accuracy"])
    validation_output = ROOT / "reports/experiments/selected-full-validation.json"
    try:
        mlflow.set_experiment("CivilCode/selected-full-validation")
        with mlflow.start_run(run_name="selected-full-validation") as run:
            prior = json.loads((ROOT / best["file"]).read_text())
            report = await evaluate_variant(base, judge, rows, best["variant"], config, validation_output, reused=prior["records"], pacer=pacer)
            report["mlflow_run_id"] = run.info.run_id
            mlflow.log_params({"retrieval_top_k": best["variant"]["top_k"], "document_language": best["variant"]["document_language"],
                               "chunk_size": base.config["max_tokens"], "overlap": 0, "embedding_model": base.config["model_id"],
                               "question_count": len(rows), "pipeline_fingerprint": pipeline_fingerprint()})
            mlflow.log_metrics({key: value for key, value in report["metrics"].items() if value is not None})
            write_json(validation_output, report)
            mlflow.log_artifact(str(validation_output), artifact_path="evaluation")
            summary["validation_file"] = str(validation_output.relative_to(ROOT))
            summary["validation_metrics"] = report["metrics"]
            summary["validation_run_id"] = run.info.run_id
    finally:
        await judge.close()
    write_json(ROOT / "reports/experiments/summary.json", summary)
    if report["metrics"]["error_count"] or report["metrics"]["scored_count"] < config["minimum_scored_questions"] or report["metrics"]["faithfulness_coverage_adjusted"] < config["faithfulness_threshold"]:
        raise RuntimeError("Selected configuration did not pass full validation; do not register a champion")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "register"))
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_questions(), indent=2))
    elif args.command == "run":
        try:
            asyncio.run(run_experiments())
        except Exception as exc:
            raise SystemExit(f"Live evaluation failed ({type(exc).__name__}); check API quota and saved experiment records") from None
    else:
        from .registry import register_best
        print(json.dumps(register_best(), indent=2))


if __name__ == "__main__":
    main()
