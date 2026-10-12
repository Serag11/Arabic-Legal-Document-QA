"""Load-test service: real retrieval and API, simulated generation latency (no provider calls)."""

import os
import time

from .api import create_app
from .generation import AnsweringService, GeneratedAnswer, GenerationResponse


class StubGenerator:
    """Returns a grounded answer quoting the top passage after a configurable provider delay."""

    def __init__(self, latency=1.5, token_delay=0.02):
        self.latency, self.token_delay = latency, token_delay

    def generate(self, question, context, language, system_prompt, on_text=None):
        time.sleep(self.latency)
        top = context[0]
        answer = f"See Article {top['article_number']} for the governing rule."
        if on_text:
            for word in answer.split(" "):
                on_text(word + " ")
                time.sleep(self.token_delay)
        parsed = GeneratedAnswer(
            answer=answer, insufficient_evidence=False, citations=[top["article_number"]],
            evidence=[{"article_number": top["article_number"], "quote": top["source_text"][:80]}],
        )
        return GenerationResponse(parsed, "stub", {"total_token_count": 0})


def create_stub_app():
    return create_app(lambda: AnsweringService(generator=StubGenerator(
        float(os.environ.get("STUB_LATENCY_SECONDS", "1.5")), float(os.environ.get("STUB_TOKEN_DELAY_SECONDS", "0.02")))))


app = create_stub_app()
