"""Single place where models are chosen. Override with env vars (.env) or CLI flags."""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

# Judges by short name. Anything not listed here is used as a raw LiteLLM model
# string, e.g. --model ollama_chat/gpt-oss-safeguard:20b.
# `needs` names the env var that must be set before that judge can be used.
JUDGES = {
    "gemini": {"model": "gemini/gemini-flash-lite-latest"},
    # groq/openai/gpt-oss-safeguard-20b or openrouter/openai/gpt-oss-safeguard-20b
    "gpt-oss": {"model": os.getenv("GPT_OSS_MODEL"), "needs": "GPT_OSS_MODEL"},
    # self-hosted OpenAI-compatible server on the DTU GPU, e.g. hosted_vllm/<model>
    "dtu": {
        "model": os.getenv("DTU_MODEL"),
        "api_base": os.getenv("DTU_API_BASE"),
        "api_key": os.getenv("DTU_API_KEY"),
        "needs": "DTU_MODEL",
    },
    # azure/<deployment>; LiteLLM reads AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
    "azure": {"model": os.getenv("AZURE_MODEL"), "needs": "AZURE_MODEL"},
    # not an LLM, handled by judge.JevJudge; needs TYPESAFE_API_KEY
    "jev": {"model": "typesafe/jev-latest"},
}

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini")
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "gemini/gemini-flash-latest")
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
# Retries with exponential backoff on 429 / transient errors (handled by LiteLLM).
NUM_RETRIES = int(os.getenv("NUM_RETRIES", "8"))
# dspy's local response cache. Off by default so every run makes fresh calls.
# Provider-side prompt caching (e.g. Gemini implicit caching) still applies.
CACHE = os.getenv("CACHE", "false").lower() in ("1", "true", "yes")


def resolve(name=None):
    """Judge name or raw model string -> (model string, extra LM kwargs)."""
    name = name or JUDGE_MODEL
    spec = dict(JUDGES.get(name, {"model": name}))
    needs = spec.pop("needs", None)
    model = spec.pop("model")
    if not model:
        raise SystemExit(f"Judge '{name}' is a placeholder: set {needs} in .env")
    return model, {k: v for k, v in spec.items() if v}


def judge_lm(name=None):
    model, kwargs = resolve(name)
    return dspy.LM(model, cache=CACHE, num_retries=NUM_RETRIES, **kwargs)


def reflection_lm(model=None):
    model, kwargs = resolve(model or REFLECTION_MODEL)
    return dspy.LM(model, cache=CACHE, num_retries=NUM_RETRIES, temperature=1.0, **kwargs)
