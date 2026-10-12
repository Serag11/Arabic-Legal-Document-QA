"""Optional Langfuse tracing. Disabled unless LANGFUSE_* credentials are set; never raises into callers."""

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("LEGAL_QA_ROOT", Path(__file__).resolve().parent.parent)).resolve()
LOGGER = logging.getLogger(__name__)


class _Observation:
    """Best-effort wrapper: tracing failures are logged, not raised."""

    def __init__(self, observation=None):
        self.observation = observation

    @property
    def trace_id(self):
        return getattr(self.observation, "trace_id", None)

    def update(self, **fields):
        if self.observation is None:
            return
        try:
            self.observation.update(**fields)
        except Exception:
            LOGGER.warning("Langfuse update failed", exc_info=True)

    def score_trace(self, name, value, comment=None):
        if self.observation is None:
            return
        try:
            self.observation.score_trace(name=name, value=value, comment=comment)
        except Exception:
            LOGGER.warning("Langfuse score failed", exc_info=True)


_NOOP = _Observation()


class Tracer:
    def __init__(self, client=None):
        self.client = client

    @property
    def enabled(self):
        return self.client is not None

    @classmethod
    def from_environment(cls, env_file=None):
        from dotenv import dotenv_values

        values = {**dotenv_values(env_file if env_file is not None else ROOT / ".env"), **os.environ}
        public, secret = values.get("LANGFUSE_PUBLIC_KEY"), values.get("LANGFUSE_SECRET_KEY")
        if not public or not secret:
            return cls()
        try:
            from langfuse import Langfuse

            return cls(Langfuse(
                public_key=public, secret_key=secret,
                base_url=values.get("LANGFUSE_BASE_URL") or values.get("LANGFUSE_HOST"),
                environment=values.get("LANGFUSE_TRACING_ENVIRONMENT") or "default",
            ))
        except Exception:
            LOGGER.warning("Langfuse is configured but could not start; tracing disabled", exc_info=True)
            return cls()

    @contextmanager
    def observation(self, name, as_type="span", **fields):
        if self.client is None:
            yield _NOOP
            return
        try:
            context = self.client.start_as_current_observation(name=name, as_type=as_type, **fields)
            observation = context.__enter__()
        except Exception:
            LOGGER.warning("Langfuse could not start %s", name, exc_info=True)
            yield _NOOP
            return
        wrapped = _Observation(observation)
        try:
            yield wrapped
        except BaseException as exc:
            wrapped.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            try:
                context.__exit__(*sys.exc_info())
            except Exception:
                LOGGER.warning("Langfuse could not end %s", name, exc_info=True)
            raise
        try:
            context.__exit__(None, None, None)
        except Exception:
            LOGGER.warning("Langfuse could not end %s", name, exc_info=True)

    def score(self, trace_id, name, value):
        """Attach a score to a finished trace (e.g. a sampled online faithfulness result)."""
        if self.client is None or not trace_id:
            return
        try:
            self.client.create_score(trace_id=trace_id, name=name, value=value, data_type="NUMERIC")
        except Exception:
            LOGGER.warning("Langfuse score failed", exc_info=True)

    def flush(self):
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception:
            LOGGER.warning("Langfuse flush failed", exc_info=True)


_default = None


def get_tracer():
    global _default
    if _default is None:
        _default = Tracer.from_environment()
    return _default
