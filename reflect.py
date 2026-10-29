"""
Lesson 3: reflection with a deterministic critic (guide sections 2.3 and 3.1).

Generator: the model writes a Python function from a written spec.
Critic:    pytest. Not a model. Tests either pass or they don't.
Loop:      if tests fail, the failures go back to the model for a revision.
           Hard cap of 3 attempts, then give up and flag it.

The model never sees the tests, only the spec and the failure messages.
That's how it would work for real: the tests stand in for reality.

Run:    python3 reflect.py              all 5 specs (at most 15 model calls)
        python3 reflect.py slugify      just one spec
        GEMINI_MODEL=gemini-3.1-flash-lite python3 reflect.py
                                        a weaker model, to watch the loop fix things

Safety note: this runs model-written code on your Mac. These are small pure
functions with a timeout, which is fine for learning. A real system runs
generated code in a sandbox with no secrets and no network (guide 3.2, 6.1).
"""
import os
import re
import subprocess
import sys
import tempfile
import time

from llm import generate as call_model, require_api_key, types

MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# 1. The task: specs the model sees, tests it doesn't
# ---------------------------------------------------------------------------
SPECS = {
    "roman_to_int": {
        "spec": """def roman_to_int(s: str) -> int
Convert a Roman numeral (I V X L C D M, uppercase only) to an integer, 1 to 3999.
Subtractive pairs are allowed only as IV IX XL XC CD CM.
Raise ValueError for anything that is not a valid, standard Roman numeral,
including the empty string, lowercase letters, 'IIII', 'VV', 'IC' and 'MMMM'.""",
        "tests": """
import pytest
from solution import roman_to_int

@pytest.mark.parametrize("s,n", [("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58),
    ("MCMXCIV", 1994), ("MMMCMXCIX", 3999), ("CDXLIV", 444)])
def test_valid(s, n):
    assert roman_to_int(s) == n

@pytest.mark.parametrize("s", ["", "IIII", "VV", "IC", "MMMM", "iv", "IIV", "XM", "ABC"])
def test_invalid(s):
    with pytest.raises(ValueError):
        roman_to_int(s)
""",
    },
    "merge_intervals": {
        "spec": """def merge_intervals(intervals: list[list[int]]) -> list[list[int]]
Merge overlapping intervals. Each interval is [start, end] with start <= end.
Intervals that touch, like [1, 2] and [2, 3], also merge into [1, 3].
The input may be unsorted. Return the merged intervals sorted by start.
An empty list returns an empty list. Do not modify the input list.""",
        "tests": """
import copy
from solution import merge_intervals

def test_basic():
    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]

def test_touching():
    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]

def test_unsorted():
    assert merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]

def test_contained():
    assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]

def test_empty():
    assert merge_intervals([]) == []

def test_does_not_mutate_input():
    data = [[3, 4], [1, 2], [2, 3]]
    before = copy.deepcopy(data)
    merge_intervals(data)
    assert data == before
""",
    },
    "slugify": {
        "spec": """def slugify(text: str) -> str
Turn a title into a URL slug: lowercase; letters with accents become plain
letters (é -> e); any run of characters that are not a-z or 0-9 becomes a
single hyphen; no hyphens at the start or end. Return '' if nothing is left.""",
        "tests": """
from solution import slugify

def test_basic():
    assert slugify("Hello World") == "hello-world"

def test_accents():
    assert slugify("Café Déjà Vu") == "cafe-deja-vu"

def test_collapse_and_strip():
    assert slugify("  --Agents: Design & Patterns!!  ") == "agents-design-patterns"

def test_numbers():
    assert slugify("Top 10 Tips for 2026") == "top-10-tips-for-2026"

def test_empty():
    assert slugify("") == ""
    assert slugify("!!!") == ""

def test_underscores_become_hyphens():
    assert slugify("snake_case_title") == "snake-case-title"
""",
    },
    "parse_duration": {
        "spec": """def parse_duration(text: str) -> int
Parse a duration like '1h30m', '90s' or '2h 5m 10s' into total seconds.
Units are h, m, s, case-insensitive. Spaces between parts are allowed.
Each unit may appear at most once, in the order h, m, s.
Raise ValueError for an empty string, an unknown unit, a repeated unit,
units out of order, a unit with no number, or any leftover text.""",
        "tests": """
import pytest
from solution import parse_duration

@pytest.mark.parametrize("t,n", [("1h30m", 5400), ("90s", 90), ("2h 5m 10s", 7510),
    ("1H", 3600), ("0m", 0), ("10m5s", 605)])
def test_valid(t, n):
    assert parse_duration(t) == n

@pytest.mark.parametrize("t", ["", "  ", "5d", "1h1h", "5m1h", "h", "10", "1h30", "1h abc"])
def test_invalid(t):
    with pytest.raises(ValueError):
        parse_duration(t)
""",
    },
    "normalize_ng_phone": {
        "spec": """def normalize_ng_phone(number: str) -> str
Normalise a Nigerian mobile number to the form '+234XXXXXXXXXX' (13 characters).
Accept '08031234567', '2348031234567' and '+2348031234567'.
Ignore spaces, hyphens and brackets anywhere in the input.
After normalising, the digits after +234 must be 10 digits starting with
70, 80, 81, 90 or 91. Raise ValueError for anything else.""",
        "tests": """
import pytest
from solution import normalize_ng_phone

@pytest.mark.parametrize("raw", ["08031234567", "2348031234567", "+2348031234567",
    "0803 123 4567", "+234 (803) 123-4567"])
def test_valid_same_number(raw):
    assert normalize_ng_phone(raw) == "+2348031234567"

def test_other_prefixes():
    assert normalize_ng_phone("09151234567") == "+2349151234567"
    assert normalize_ng_phone("07011234567") == "+2347011234567"

@pytest.mark.parametrize("raw", ["", "0803123456", "080312345678", "06031234567",
    "+2340803123456", "+1 555 123 4567", "0803abc4567", "+23480312345678"])
def test_invalid(raw):
    with pytest.raises(ValueError):
        normalize_ng_phone(raw)
""",
    },
}

GENERATOR_PROMPT = """You write one Python function that exactly matches a spec.
Return ONLY the Python code: the function and any imports it needs.
No tests, no examples, no explanation, no markdown fences.
Use only the Python standard library."""


# ---------------------------------------------------------------------------
# 2. Generator: the model
# ---------------------------------------------------------------------------
def generate(spec: str, previous: str | None = None, failures: str | None = None) -> str:
    if previous is None:
        prompt = f"Spec:\n{spec}"
    else:
        prompt = (f"Spec:\n{spec}\n\nYour previous attempt:\n{previous}\n\n"
                  f"It failed these tests:\n{failures}\n\n"
                  "Fix the function so it meets the spec. Return the full function.")
    response = call_model(
        prompt, types.GenerateContentConfig(system_instruction=GENERATOR_PROMPT),
        fallback=True)
    code = response.text or ""
    # Models often add ```python fences anyway. Strip them rather than trusting.
    match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    return (match.group(1) if match else code).strip()


# ---------------------------------------------------------------------------
# 3. Critic: pytest, run in a throwaway folder with a timeout
# ---------------------------------------------------------------------------
def critique(code: str, tests: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmp, "test_solution.py"), "w") as f:
            f.write(tests)
        try:
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"],
                cwd=tmp, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False, "Tests timed out after 20s. Check for infinite loops."
    output = (run.stdout + run.stderr).strip()
    # Keep the feedback lean (guide 4.1): the last part has the failures and summary.
    return run.returncode == 0, output[-2500:]


def summary_line(output: str) -> str:
    lines = [l for l in output.splitlines() if "passed" in l or "failed" in l or "error" in l]
    return lines[-1].strip("= ") if lines else output.splitlines()[-1] if output else "no output"


# ---------------------------------------------------------------------------
# 4. The reflection loop
# ---------------------------------------------------------------------------
def solve(name: str, spec: str, tests: str) -> dict:
    print(f"\n=== {name}")
    code, failures, first_pass = None, None, False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        code = generate(spec, previous=code, failures=failures)
        passed, output = critique(code, tests)
        print(f"  attempt {attempt}: {'PASS' if passed else 'FAIL'}  ({summary_line(output)})")
        if attempt == 1:
            first_pass = passed
        if passed:
            break
        failures = output
        time.sleep(4)  # free-tier rate limit
    os.makedirs("solutions", exist_ok=True)
    with open(f"solutions/{name}.py", "w") as f:
        f.write(code + ("\n" if passed else "\n# FLAGGED: still failing after "
                        f"{MAX_ATTEMPTS} attempts. Needs a human.\n"))
    return {"name": name, "first_pass": first_pass, "final_pass": passed, "attempts": attempt}


if __name__ == "__main__":
    require_api_key()
    chosen = sys.argv[1:] or list(SPECS)
    unknown = [c for c in chosen if c not in SPECS]
    if unknown:
        sys.exit(f"Unknown spec(s): {unknown}. Choose from: {list(SPECS)}")

    results = []
    for name in chosen:
        try:
            results.append(solve(name, SPECS[name]["spec"], SPECS[name]["tests"]))
        except Exception as e:
            print(f"  stopped: {type(e).__name__}: {str(e)[:150]}")
            break

    if results:
        n = len(results)
        first = sum(r["first_pass"] for r in results)
        final = sum(r["final_pass"] for r in results)
        calls = sum(r["attempts"] for r in results)
        print(f"\nWithout reflection (attempt 1 only): {first}/{n} passed")
        print(f"With reflection (up to {MAX_ATTEMPTS} attempts):   {final}/{n} passed")
        print(f"Model calls used: {calls}.  Final code saved in the solutions folder.")
