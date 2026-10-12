"""Sampled online faithfulness: judge a fraction of grounded answers off the request path."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import random
from threading import Lock

LOGGER = logging.getLogger(__name__)


class FaithfulnessSampler:
    """Scores a random fraction of answers with a RAGAS-style judge in one background worker.

    The judge is created lazily so the serving image does not need the evaluation
    dependencies unless sampling is enabled. Work beyond max_pending is dropped, never queued
    without bound, so a slow judge cannot back up the service.
    """

    def __init__(self, rate, judge_factory, metrics, tracer, max_pending=8, rng=random.random):
        self.rate, self.judge_factory, self.metrics, self.tracer = rate, judge_factory, metrics, tracer
        self.max_pending, self.rng = max_pending, rng
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faithfulness")
        self.pending, self.lock, self.judge, self.disabled = 0, Lock(), None, False

    @classmethod
    def from_environment(cls, metrics, tracer, env=None):
        env = os.environ if env is None else env
        rate = float(env.get("ONLINE_FAITHFULNESS_SAMPLE_RATE", "0") or 0)
        if rate <= 0:
            return None
        if not 0 < rate <= 1:
            raise ValueError("ONLINE_FAITHFULNESS_SAMPLE_RATE must be in (0, 1]")

        def factory():
            from .experiments import RagasJudge
            from .generation import ROOT
            import json

            model = json.loads((ROOT / "config/experiments.json").read_text(encoding="utf-8"))["judge_model"]
            return RagasJudge(model)

        return cls(rate, factory, metrics, tracer)

    def maybe_submit(self, question, result, context):
        if self.disabled or result.get("insufficient_evidence") or self.rng() >= self.rate:
            return False
        with self.lock:
            if self.pending >= self.max_pending:
                return False
            self.pending += 1
        self.executor.submit(self._score, question, result["answer"], context, result.get("trace_id"))
        return True

    def _score(self, question, answer, context, trace_id):
        try:
            if self.judge is None:
                self.judge = self.judge_factory()
            score = asyncio.run(self.judge.score({"question": question}, answer, context))["faithfulness"]
            self.metrics.faithfulness.observe(score)
            if trace_id:
                self.tracer.score(trace_id, "faithfulness", score)
        except ImportError:
            self.disabled = True
            LOGGER.warning("Online faithfulness needs the evaluation dependencies; sampling disabled", exc_info=True)
        except Exception:
            self.metrics.faithfulness_failures.inc()
            LOGGER.warning("Online faithfulness scoring failed", exc_info=True)
        finally:
            with self.lock:
                self.pending -= 1

    def close(self):
        self.executor.shutdown(wait=True)
