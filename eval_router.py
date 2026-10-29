"""
Lesson 2b: an eval set for the router (guide sections 5.4 and 5.5), cached.

Every result is cached in router_cache.json, keyed on the request text, the
router prompt and the router model. Re-running only calls the model for
cases it hasn't seen before, or for every case after you change the prompt
or model (because then old answers no longer count). This is tool-result
caching from guide section 6.5.

Run:  python3 eval_router.py            run everything (cached cases are free)
      python3 eval_router.py --max 10   make at most 10 new model calls this run
      python3 eval_router.py --fresh    ignore the cache and re-run everything
"""
import argparse
import csv
import hashlib
import json
import time

from llm import require_api_key
from router import CONFIDENCE_THRESHOLD, ROUTER_MODEL, ROUTER_PROMPT, Route, route

CACHE_FILE = "router_cache.json"

# (request, acceptable categories). Ambiguous cases accept more than one answer.
# Only ever ADD cases. Old cases are your regression suite.
CASES = [
    # coding
    ("My Python script says KeyError: 'user_id', how do I fix it?", {"coding"}),
    ("Write a function that reverses a linked list in JavaScript", {"coding"}),
    ("Why is my React component re-rendering infinitely?", {"coding"}),
    ("Can you review this SQL query for performance? SELECT * FROM orders", {"coding"}),
    ("How do I read a CSV file with pandas?", {"coding"}),
    ("What does this regex do: ^\\d{3}-\\d{4}$", {"coding"}),
    ("My Docker build fails with 'no space left on device'", {"coding"}),
    ("Convert this for-loop into a list comprehension", {"coding"}),
    # research
    ("What's the difference between PostgreSQL and MongoDB?", {"research"}),
    ("Explain how vector databases work", {"research"}),
    ("Compare AWS Lambda and Google Cloud Run for a small API", {"research"}),
    ("What is the CAP theorem?", {"research"}),
    ("Summarise the main ideas behind retrieval augmented generation", {"research"}),
    ("Which is better for mobile apps in 2026, Flutter or React Native?", {"research"}),
    ("What are the pros and cons of microservices?", {"research"}),
    ("Who created the Rust programming language?", {"research"}),
    # support
    ("I was charged twice this month, can I get a refund?", {"support"}),
    ("How do I cancel my subscription?", {"support"}),
    ("I can't log in, it says my password is wrong", {"support"}),
    ("Where is my order? It was supposed to arrive yesterday", {"support"}),
    ("Please change the email address on my account", {"support"}),
    ("Do you offer a student discount on the Pro plan?", {"support"}),
    ("My invoice shows the wrong company name", {"support"}),
    ("I want to upgrade from Basic to Pro", {"support"}),
    # unclear / vague
    ("help", {"unclear"}),
    ("it's not working", {"unclear", "support", "coding"}),
    # ambiguous: span two categories
    ("Your API keeps returning 500 errors and I'm being billed for the failed calls",
     {"unclear", "support", "coding"}),
    ("Research the best Python logging library and write me an example using it",
     {"unclear", "research", "coding"}),
    ("Is your product GDPR compliant?", {"support", "research"}),
    ("My code that calls your billing API gets a 401 error", {"coding", "support"}),
    # harder edge cases
    ("error 404", {"unclear", "coding", "support"}),  # too vague: asking is the right answer
    ("My payment went through but my pro plan isn't active, and also how do I "
     "change my password?", {"support", "coding"}),
    ("what's the weather like in Lagos today?", {"unclear"}),
    ("hi", {"unclear"}),
    ("it's broken", {"unclear", "coding", "support"}),
    ("Can I get my money back for the double charge?", {"support"}),
    ("How do I cancel my account?", {"support"}),
    # messy real-world input: typos, Pidgin, very short
    ("pls my acct no dey open since morning", {"support"}),
    ("abeg why my code dey show undefined", {"coding"}),
    ("refnd", {"support", "unclear"}),
    ("wat is kubernetes", {"research"}),
    ("i paid but e no work", {"support", "unclear"}),
]


def cache_key(text: str) -> str:
    # If the prompt or model changes, every key changes, so old answers are ignored.
    raw = f"{ROUTER_MODEL}|{ROUTER_PROMPT}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_cache() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None,
                        help="most new model calls to make this run")
    parser.add_argument("--fresh", action="store_true", help="ignore the cache")
    args = parser.parse_args()

    require_api_key()

    cache = {} if args.fresh else load_cache()
    rows, correct, scored, new_calls, skipped = [], 0, 0, 0, 0

    for i, (text, expected) in enumerate(CASES, 1):
        key = cache_key(text)
        if key in cache:
            r, source = Route(**cache[key]), "cached"
        elif args.max is not None and new_calls >= args.max:
            skipped += 1
            continue
        else:
            try:
                r = route(text)
            except Exception as e:  # quota used up: save progress and stop cleanly
                print(f"\nStopped at case {i}: {type(e).__name__}: {str(e)[:120]}")
                break
            cache[key] = r.model_dump()
            save_cache(cache)  # save after every call so nothing is lost
            new_calls += 1
            source = "new"
            time.sleep(4)  # stay under the free tier's requests-per-minute limit

        acted = "unclear" if r.confidence < CONFIDENCE_THRESHOLD else r.category
        ok = acted in expected
        correct += ok
        scored += 1
        print(f"{i:2}. {'PASS' if ok else 'FAIL'}  got={acted:<8} conf={r.confidence:.2f} "
              f"[{source:<6}] want={'/'.join(sorted(expected)):<22} | {text[:50]}")
        rows.append([text, "/".join(sorted(expected)), r.category, acted,
                     r.confidence, ok, r.reason])

    if scored:
        print(f"\nAccuracy: {correct}/{scored} = {correct / scored:.0%}   "
              f"(new model calls this run: {new_calls})")
    not_run = len(CASES) - scored
    if not_run:
        print(f"{not_run} case(s) not run yet. Run again later to finish them; "
              f"finished cases stay cached.")

    with open("router_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request", "expected", "router_said", "acted_as",
                    "confidence", "correct", "reason"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
