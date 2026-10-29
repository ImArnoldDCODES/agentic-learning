"""
Lesson 2: routing (guide section 2.4).

A small, cheap model classifies each request. Then the request goes to a
specialist built for that kind of request. Low confidence goes to a
clarifying question, never to a guess.

Run:  python3 router.py          (chat with it)
      python3 eval_router.py     (measure its accuracy)
"""
import os
from typing import Literal

from pydantic import BaseModel, Field

from llm import MODEL, generate, require_api_key, types

# Model routing: a cheap model for the easy job (classifying),
# a stronger one for the specialists that do the real work.
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gemini-3.1-flash-lite")
SPECIALIST_MODEL = MODEL
CONFIDENCE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# 1. The route is DATA, not prose: a schema the model must fill in (4.4)
# ---------------------------------------------------------------------------
class Route(BaseModel):
    category: Literal["coding", "research", "support", "unclear"] = Field(
        description="Which specialist should handle this request.")
    confidence: float = Field(
        description="0.0 to 1.0. How sure you are about the category.")
    reason: str = Field(description="One short sentence explaining the choice.")


ROUTER_PROMPT = """You route requests for a software company's assistant.
Pick exactly one category:
- coding: writing, fixing, explaining or reviewing code; error messages; programming questions.
- research: comparing tools or technologies, explaining concepts, finding facts, summarising a topic.
- support: the user's own account, billing, subscription, refunds, login problems, orders.
- unclear: the request is too vague, or truly needs more than one specialist.
If you are unsure, lower your confidence rather than guessing."""


# ---------------------------------------------------------------------------
# 2. Specialists: each one has its own prompt (in a real system, own tools too)
# ---------------------------------------------------------------------------
SPECIALISTS = {
    "coding": "You are a senior software engineer. Give working code and a short "
              "explanation. Ask for the error message or code if it's missing.",
    "research": "You are a careful technical researcher. Give a balanced, concise "
                "answer and say when something may be out of date.",
    "support": "You are a friendly customer support agent for a software product. "
               "You cannot see accounts yet, so explain the next step the user "
               "should take and what details support will need.",
}


# ---------------------------------------------------------------------------
# 3. Route, then hand off
# ---------------------------------------------------------------------------
def route(request: str) -> Route:
    response = generate(
        request,
        types.GenerateContentConfig(
            system_instruction=ROUTER_PROMPT,
            response_mime_type="application/json",
            response_schema=Route,
            temperature=0,  # classification should be as repeatable as possible
        ),
        model=ROUTER_MODEL,
    )
    try:
        return Route.model_validate_json(response.text)
    except Exception:  # bad JSON is treated as "unclear", never as a crash
        return Route(category="unclear", confidence=0.0, reason="router output invalid")


def handle(request: str) -> str:
    r = route(request)
    print(f"  [router] {r.category} (confidence {r.confidence:.2f}): {r.reason}")

    if r.category == "unclear" or r.confidence < CONFIDENCE_THRESHOLD:
        return ("I want to make sure I send this to the right place. Is this about "
                "code, researching a topic, or your account/billing?")

    response = generate(
        request,
        types.GenerateContentConfig(system_instruction=SPECIALISTS[r.category]),
        model=SPECIALIST_MODEL, fallback=True,
    )
    return response.text


if __name__ == "__main__":
    require_api_key()
    print("Router ready. Type a request, or 'quit' to stop.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            print(handle(text), "\n")
