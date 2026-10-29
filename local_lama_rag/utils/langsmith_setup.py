"""
==========================================================
LangSmith observability
==========================================================
LangSmith traces automatically once these env vars are set --
no code-level instrumentation needed elsewhere; every LangChain
Runnable (.invoke/.stream calls in every agent) gets traced,
including nested spans for the LangGraph workflow.

Requires a LANGCHAIN_API_KEY in your .env (from smith.langchain.com).
If it's absent, tracing is silently skipped rather than raising --
observability is a nice-to-have, not something that should crash
a run just because a key wasn't configured yet.
==========================================================
"""

import os

import config
from utils.logger import get_logger

logger = get_logger("langsmith")


def setup_langsmith():
    if not getattr(config, "LANGSMITH_ENABLED", False):
        return

    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning(
            "LANGSMITH_ENABLED=True but LANGCHAIN_API_KEY is not set -- "
            "tracing disabled. Get a key at smith.langchain.com and add it to .env."
        )
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", config.LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://eu.api.smith.langchain.com")
    logger.info(f"LangSmith tracing enabled -- project: {os.environ['LANGCHAIN_PROJECT']}")
