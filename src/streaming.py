"""Incremental answer-text extraction and a thread-to-async bridge for server-sent events."""

import asyncio
import json
import re
from threading import Event, Thread

ANSWER_KEY = re.compile(r'"answer"\s*:\s*"')
HIGH_SURROGATE_TAIL = re.compile(r"\\u[dD][89abAB][0-9a-fA-F]{2}$")


class AnswerTextStream:
    """Turns streamed structured-JSON fragments into the decoded text of the "answer" field."""

    def __init__(self):
        self.buffer = ""
        self.emitted = ""

    def feed(self, fragment):
        self.buffer += fragment
        start = ANSWER_KEY.search(self.buffer)
        if not start:
            return ""
        raw, index = self.buffer[start.end():], 0
        while index < len(raw) and raw[index] != '"':
            index += 2 if raw[index] == "\\" else 1
        closed = index < len(raw)
        raw = raw[:index]
        if not closed:
            # Hold back an incomplete escape sequence until the rest arrives.
            raw = re.sub(r"\\(u[0-9a-fA-F]{0,3})?$", "", raw)
            raw = HIGH_SURROGATE_TAIL.sub("", raw)
            if raw.endswith("\\"):
                raw = raw[:-1]
        try:
            text = json.loads(f'"{raw}"')
        except ValueError:
            return ""
        if not text.startswith(self.emitted):
            return ""
        delta, self.emitted = text[len(self.emitted):], text
        return delta


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_events(work):
    """Run work(emit) in one worker thread and yield its (event, data) pairs as SSE frames.

    One thread owns the whole request so tracing context and the service's locks behave
    exactly as in the non-streaming path. Stops emitting if the client disconnects.
    """
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    cancelled = Event()
    finished = object()

    def emit(event, data):
        if cancelled.is_set():
            raise ConnectionAbortedError("Client disconnected")
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def run():
        try:
            work(emit)
        except ConnectionAbortedError:
            pass
        except Exception as exc:  # reported as an event: headers are already sent
            if not cancelled.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, ("error", {"detail": str(exc) or type(exc).__name__}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, finished)

    Thread(target=run, daemon=True).start()
    try:
        while (item := await queue.get()) is not finished:
            yield sse(*item)
    finally:
        cancelled.set()
