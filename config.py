"""Single place where models are chosen. Override with env vars (.env) or CLI flags."""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-flash-lite-latest")
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "gemini/gemini-flash-latest")
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
# Retries with exponential backoff on 429 / transient errors (handled by LiteLLM).
NUM_RETRIES = int(os.getenv("NUM_RETRIES", "8"))
# dspy's local response cache. Off by default so every run makes fresh calls.
# Provider-side prompt caching (e.g. Gemini implicit caching) still applies.
CACHE = os.getenv("CACHE", "false").lower() in ("1", "true", "yes")


def _lm(model, **kwargs):
    return dspy.LM(model, cache=CACHE, num_retries=NUM_RETRIES, **kwargs)


def setup(judge_model=None):
    """Configure dspy's default LM (the judge). Returns it."""
    lm = _lm(judge_model or JUDGE_MODEL)
    dspy.configure(lm=lm)
    return lm


def reflection_lm(model=None):
    return _lm(model or REFLECTION_MODEL, temperature=1.0)
