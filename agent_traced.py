"""
Lesson 4: the lesson 1 agent, with tracing (guide section 5.6).

Same tools and loop as agent.py. The only new thing is that every model
call and tool call is wrapped in a span, so you can see the whole run.

Run:  python3 agent_traced.py
      python3 agent_traced.py "How many days until 1 January 2027, and how many weeks is that?"
"""
import json
import sys

from agent import MAX_STEPS, execute, select_tools, tool_config, tool_result_part
from llm import MODEL, generate, require_api_key, types
from tracing import Tracer

PROMPT_VERSION = "v1"  # bump this when you change the prompt or tools


def run_agent(task: str) -> str:
    tools = select_tools()
    config = tool_config(tools)
    tracer = Tracer(task, model=MODEL, prompt_version=PROMPT_VERSION, tools=list(tools))
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]
    seen_calls = set()
    answer = "Stopped: reached max steps without a final answer."

    with tracer.span("agent run", kind="run"):
        for step in range(1, MAX_STEPS + 1):
            with tracer.span(f"step {step}", kind="step"):

                # --- model call ---
                with tracer.span(f"model call {step}", kind="llm", model=MODEL) as s:
                    try:
                        response = generate(contents, config)
                    except Exception as e:
                        s["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                        answer = f"Stopped: model call failed ({s['error']})"
                        break
                    usage = response.usage_metadata
                    s["tokens_in"] = usage.prompt_token_count or 0
                    s["tokens_out"] = usage.candidates_token_count or 0
                    calls = response.function_calls or []
                    s["summary"] = (f"wants: {', '.join(c.name for c in calls)}" if calls
                                    else "answered (no tool call)")

                contents.append(response.candidates[0].content)
                if not calls:
                    answer = response.text or "(empty answer)"
                    break

                # --- tool calls ---
                result_parts = []
                for call in calls:
                    args = dict(call.args or {})
                    with tracer.span(call.name, kind="tool", args=args) as s:
                        key = (call.name, json.dumps(args, sort_keys=True))
                        if key in seen_calls:
                            result = {"ok": False, "error": "Repeated identical call."}
                        else:
                            seen_calls.add(key)
                            result = execute(call.name, args, tools)
                        s["ok"] = result["ok"]
                        s["result"] = result
                        s["summary"] = f"{args} -> {result.get('data', result.get('error'))}"
                    result_parts.append(tool_result_part(call, result))
                contents.append(types.Content(role="user", parts=result_parts))

    tracer.finish(answer)
    return answer


if __name__ == "__main__":
    require_api_key()
    question = " ".join(sys.argv[1:]) or "How many days are left until Christmas this year?"
    run_agent(question)
