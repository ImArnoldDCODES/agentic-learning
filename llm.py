"""
Shared model-calling code for every lesson.

One place for the client, the default models, and retry handling, so each
lesson file can focus on its pattern instead of on plumbing.

Retries (guide section 6.3):
- Transient errors are retried with exponential backoff and jitter:
  rate limits (429) and server trouble (500, 503).
- Anything else, such as a bad request (400), is our fault, so it fails fast.
- Optionally falls back to a second model if the first stays unavailable.
"""
import os
import random
import sys
import time

from google import genai
from google.genai import types  # re-exported so lessons can `from llm import types`

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemini-3.1-flash-lite")

TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "503", "UNAVAILABLE")

_client = None


def require_api_key() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit('GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY="your-key"\n'
                 "Get a free key at https://aistudio.google.com")


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY
    return _client


def generate(contents, config=None, model: str | None = None,
             fallback: bool = False, retries: int = 3):
    """Call the model, retrying transient errors. Raises if every attempt fails.

    Use fallback=False inside a multi-step agent conversation: switching models
    halfway through a conversation can break it.
    """
    primary = model or MODEL
    models = [primary]
    if fallback and FALLBACK_MODEL != primary:
        models.append(FALLBACK_MODEL)

    last_error = None
    for m in models:
        for attempt in range(retries):
            try:
                response = client().models.generate_content(
                    model=m, contents=contents, config=config)
                if m != primary:
                    print(f"    (used fallback model {m})")
                return response
            except Exception as e:
                if not any(code in str(e) for code in TRANSIENT):
                    raise
                last_error = e
                if attempt < retries - 1:
                    wait = 10 * 2 ** attempt + random.uniform(0, 3)
                    print(f"    ({m}: {str(e)[:45]}... retrying in {wait:.0f}s)")
                    time.sleep(wait)
    raise last_error


__all__ = ["MODEL", "FALLBACK_MODEL", "client", "generate", "require_api_key", "types"]
