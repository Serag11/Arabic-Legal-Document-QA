"""Locust load test for /ask and /ask/stream using the fixed evaluation questions.

Locust times a streamed response only until its headers arrive, so /ask/stream is also
reported as "stream time-to-first-delta" and "stream total duration". It records answer outcomes as separate rows (grounded_expected,
grounded_other, refused:<reason>) so the same run also measures answer quality.
"""

import json
import random
from pathlib import Path
import time

from locust import HttpUser, between, events, task

ROWS = [json.loads(line) for line in
        (Path(__file__).resolve().parent.parent / "evaluation/answer_questions.jsonl").read_text(encoding="utf-8").splitlines()]


def record(environment, name, elapsed_ms=0, exception=None):
    environment.events.request.fire(request_type="OUTCOME", name=name, response_time=elapsed_ms,
                                    response_length=0, exception=exception, context={})


def classify(row, result):
    if result.get("insufficient_evidence"):
        return f"refused:{result.get('reason')}"
    expected = {f"Egyptian Civil Code, Article {n}" for n in row["expected_article_numbers"]}
    return "grounded_expected" if expected & set(result["sources"]) else "grounded_other"


class Asker(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def ask(self):
        row = random.choice(ROWS)
        with self.client.post("/ask", json={"question": row["question"], "language": row["language"]},
                              name="/ask", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            record(self.environment, f"answer {classify(row, response.json())}")

    @task(1)
    def ask_stream(self):
        row = random.choice(ROWS)
        start, first_delta, final = time.perf_counter(), None, None
        with self.client.post("/ask/stream", json={"question": row["question"], "language": row["language"]},
                              name="/ask/stream", stream=True, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            event = None
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    if event == "delta" and first_delta is None:
                        first_delta = (time.perf_counter() - start) * 1000
                    elif event == "final":
                        final = json.loads(line[6:])
                    elif event == "error":
                        response.failure(f"stream error: {line[6:80]}")
                        return
            if final is None:
                response.failure("stream ended without a final event")
                return
        if first_delta is not None:
            record(self.environment, "stream time-to-first-delta", first_delta)
        record(self.environment, "stream total duration", (time.perf_counter() - start) * 1000)
        record(self.environment, f"answer {classify(row, final)}")


@events.quitting.add_listener
def fail_on_errors(environment, **kwargs):
    if environment.stats.total.fail_ratio > 0.01:
        environment.process_exit_code = 1
