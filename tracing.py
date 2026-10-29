"""
Lesson 4: a tiny tracer, no service or account needed (guide section 5.6).

A trace is the full record of one run: every model call and tool call,
with inputs, outputs, tokens, timing and errors, nested as "spans".

Use it:   tracer = Tracer("my task")
          with tracer.span("llm", kind="llm") as s:
              ...                       # do the work
              s["tokens_in"] = 220      # record anything you want
          tracer.finish()               # prints the tree, saves JSON

View an old trace:  python3 tracing.py traces/<file>.json
"""
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager


class Tracer:
    def __init__(self, name: str, **attrs):
        self.run_id = uuid.uuid4().hex[:6]
        self.name = name
        self.attrs = attrs
        self.spans = []
        self._stack = []  # the span we're currently inside, so children nest
        self._start = time.perf_counter()

    @contextmanager
    def span(self, name: str, kind: str = "step", **attrs):
        span = {
            "id": len(self.spans),
            "parent": self._stack[-1]["id"] if self._stack else None,
            "name": name, "kind": kind,
            "start_ms": round((time.perf_counter() - self._start) * 1000),
            **attrs,
        }
        self.spans.append(span)
        self._stack.append(span)
        t0 = time.perf_counter()
        try:
            yield span
        except Exception as e:  # record the error, then let it continue upward
            span["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            span["duration_ms"] = round((time.perf_counter() - t0) * 1000)
            self._stack.pop()

    def finish(self, output: str | None = None) -> str:
        run = {
            "run_id": self.run_id, "name": self.name, **self.attrs,
            "duration_ms": round((time.perf_counter() - self._start) * 1000),
            "output": output, "spans": self.spans,
        }
        os.makedirs("traces", exist_ok=True)
        path = f"traces/run_{time.strftime('%Y%m%d_%H%M%S')}_{self.run_id}.json"
        with open(path, "w") as f:
            json.dump(run, f, indent=2, default=str)
        print_trace(run)
        print(f"\nSaved trace: {path}")
        return path


def print_trace(run: dict) -> None:
    spans = run["spans"]
    tok_in = sum(s.get("tokens_in", 0) for s in spans)
    tok_out = sum(s.get("tokens_out", 0) for s in spans)
    llm = [s for s in spans if s["kind"] == "llm"]
    tools = [s for s in spans if s["kind"] == "tool"]
    errors = [s for s in spans if s.get("error") or s.get("ok") is False]

    print("\n" + "=" * 78)
    print(f"run {run['run_id']}: \"{run['name']}\"   {run['duration_ms'] / 1000:.1f}s   "
          f"{tok_in + tok_out} tokens ({tok_in} in / {tok_out} out)")
    print(f"{len(llm)} model calls, {len(tools)} tool calls, {len(errors)} error(s)")
    print("=" * 78)

    children = {}
    for s in spans:
        children.setdefault(s["parent"], []).append(s)

    def show(parent, depth):
        for s in children.get(parent, []):
            bits = [f"{s['duration_ms']:>6}ms"]
            if "tokens_in" in s:
                bits.append(f"{s['tokens_in']:>5} in {s.get('tokens_out', 0):>4} out")
            label = f"{'  ' * depth}{s['kind']:<5} {s['name']}"
            flag = "  <-- ERROR" if s.get("error") or s.get("ok") is False else ""
            print(f"{label:<44} {'  '.join(bits)}{flag}")
            if s.get("summary"):
                print(f"{'  ' * depth}        {s['summary']}")
            show(s["id"], depth + 1)

    show(None, 0)

    timed = [s for s in spans if s["kind"] in ("llm", "tool")]
    if timed:
        slow = max(timed, key=lambda s: s["duration_ms"])
        print(f"\nSlowest span:     {slow['kind']} {slow['name']} ({slow['duration_ms']}ms)")
    if llm:
        big = max(llm, key=lambda s: s.get("tokens_in", 0) + s.get("tokens_out", 0))
        print(f"Most tokens:      {big['name']} "
              f"({big.get('tokens_in', 0) + big.get('tokens_out', 0)} tokens)")
    if run.get("output"):
        print(f"Final answer:     {run['output'][:200]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 tracing.py traces/<file>.json")
    with open(sys.argv[1]) as f:
        print_trace(json.load(f))
