# Agentic learning

Four small, framework-free Python exercises that teach how AI agents actually work: the agent loop, routing, reflection and tracing. Each one is a single file you can read in ten minutes and run on Google Gemini's free tier.

I built these while working through [*Agentic design patterns*](https://baswe.ai), an AI engineering field guide by Bashiri Smith (BASWE). Section numbers in the code, like "guide 2.4", point to that guide. The guide isn't included here, so get it from the author.

## The lessons

| # | Pattern | Files | What you learn |
|---|---------|-------|----------------|
| 1 | The agent loop | `agent.py` | The ~10-line loop behind every agent: call the model, run its tool calls, feed the results back, repeat. |
| 2 | Routing | `router.py`, `eval_router.py` | Classify a request with a cheap model and send it to the right specialist, then measure accuracy with an eval set. |
| 3 | Reflection | `reflect.py` | Generate code, grade it with pytest, feed the failures back, and retry up to 3 times. |
| 4 | Tracing | `tracing.py`, `agent_traced.py` | Record every model call and tool call with tokens and timing, so you can see what an agent did. |

`llm.py` is shared by every lesson. It handles the API client, default models, and retries with backoff for rate limits and server errors.

## Setup

You need Python 3.10 or newer and a free Gemini API key.

1. Get a key at [aistudio.google.com](https://aistudio.google.com). Click **Get API key**. No card is needed.
2. Install the dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Set your key in the terminal. Never put it in the code.
   ```bash
   export GEMINI_API_KEY="your-key"
   ```

On a Mac, use `python3`, not `python`.

The free tier allows roughly 10 requests a minute, and the lessons pause between calls to stay under it. Google may use free-tier prompts to improve its models, so don't send private data.

## Lesson 1: the agent loop

```bash
python3 agent.py
```

The agent answers "How many days until Christmas?" by deciding for itself to call `get_current_date`, then `days_between`, then answering. Watch `tokens in` grow at each step: the whole conversation is resent on every call.

**Experiment: take away its eyes.**

```bash
python3 agent.py --no-date-tool --no-guardrail
```

With no way to check the date, the model invents one and answers confidently. In my run it decided today was May 22, 2024. Its `days_between` call succeeded and the maths was right, but the input was made up. An agent is only as good as what its tools let it see.

Now drop `--no-guardrail`. The system prompt tells it not to guess, so it should ask you instead.

## Lesson 2: routing and your first eval

```bash
python3 router.py        # chat with it; type quit to stop
python3 eval_router.py   # measure it on 42 labelled requests
```

The router must return structured output, a category, a confidence score and a reason, not chat text. Requests below 0.7 confidence get a clarifying question instead of a guess.

The eval caches every answer, so re-runs are free, and the cache resets itself when you change the router prompt or model. Use `--max 10` to spread the calls over several runs if your quota is tight, or `--fresh` to ignore the cache.

**What I found:**

- **The first 30 cases scored 100%.** That usually means the eval is too easy, not that the router is perfect.
- **Self-reported confidence was almost useless.** Nearly everything scored 1.00, so the confidence gate rarely fired. Running the router several times and treating disagreement as "unclear" is a more honest uncertainty signal.
- **The one failure was a bad label.** "error 404" is genuinely vague, so asking for clarification was the right answer. Evals have bugs too.
- **Only ever add cases.** The old ones are your regression suite.

## Lesson 3: reflection with a deterministic critic

```bash
python3 reflect.py
python3 reflect.py normalize_ng_phone                     # just one spec
GEMINI_MODEL=gemini-3.1-flash-lite python3 reflect.py     # weaker model
```

The model writes 5 functions from written specs: Roman numerals, interval merging, URL slugs, duration parsing and Nigerian phone numbers. It never sees the tests, so it can't write code that only passes those exact cases. pytest is the critic. A model reviewing code can be fooled by something that looks plausible, but a test either passes or fails.

The script reports pass rates with and without reflection. With the default model, everything passed on the first attempt, so the loop added nothing, and that's a valid result. Measure before adding complexity. Because the critic is pytest, checking a correct answer costs no extra model calls. Use the weaker model to watch the loop fix things.

> This runs model-written code on your machine. They're small pure functions with a timeout, which is fine for learning. A real system runs generated code in a sandbox with no secrets and no network.

## Lesson 4: tracing

```bash
python3 agent_traced.py "How many days until 1 January 2027, and how many weeks is that?"
python3 tracing.py traces/<file>.json   # re-print any saved run
```

Each run prints a tree of spans and saves it as JSON in `traces/`:

```
run   agent run
  step  step 2
    llm   model call 2            352 in   40 out
            wants: days_between
    tool  days_between                       <-- ERROR
            {'end': '2026-13-25'} -> ValueError: month must be in 1..12
  step  step 3
    llm   model call 3            470 in   40 out
    tool  days_between
            {'end': '2026-12-25'} -> 58
```

From the trace you can see the model made a mistake, got a structured error back and fixed it, which the final answer alone never shows. Real teams send the same kind of spans to a service like Langfuse or LangSmith. Building the raw version once shows you what those tools are recording.

## Where to go next

- **Voting in the router:** route each request 3 times and treat disagreement as unclear.
- **Messier eval cases:** typos, slang, other languages, two requests in one.
- **Planning (guide 2.2):** an agent that writes a plan as data, runs it step by step, and re-plans when a step fails.

## License

MIT for the code in this repository. The guide it follows belongs to its author and isn't covered.
# agentic-learning
