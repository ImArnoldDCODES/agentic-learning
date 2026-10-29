"""
Lesson 1: the raw agent loop, no framework (guide sections 1.3 and 2.1).

The model decides what to do, calls tools, reads the results, and repeats
until it answers without a tool call or hits MAX_STEPS.

Run:  python3 agent.py
      python3 agent.py "How many weeks until 1 March 2027?"

Experiments:
      python3 agent.py --no-date-tool                  the agent can't see today's date
      python3 agent.py --no-date-tool --no-guardrail   ...and nothing tells it not to guess
"""
import argparse
import ast
import json
import operator
from datetime import date

from llm import MODEL, generate, require_api_key, types

MAX_STEPS = 10

# A guardrail instruction: tell the model what to do when information is missing.
SYSTEM_PROMPT = ("Never assume today's date or any fact you can't verify with a tool. "
                 "If you lack it, say so and ask the user.")


# ---------------------------------------------------------------------------
# 1. Tools: plain Python functions
# ---------------------------------------------------------------------------
def get_current_date() -> str:
    return date.today().isoformat()


def days_between(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def calculator(expression: str) -> float:
    """Safely evaluate basic arithmetic (no eval())."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** are allowed")
    return ev(ast.parse(expression, mode="eval").body)


# ---------------------------------------------------------------------------
# 2. Tool definitions: the description IS the prompt for each tool (guide 4.2)
# ---------------------------------------------------------------------------
ALL_TOOLS = {
    "get_current_date": {
        "fn": get_current_date,
        "description": "Returns today's date as YYYY-MM-DD. Use this whenever "
                       "the answer depends on what today is.",
        "parameters": None,  # takes no arguments
    },
    "days_between": {
        "fn": days_between,
        "description": "Number of days from start to end. Both dates must be "
                       "YYYY-MM-DD. Returns a negative number if end is before start.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "YYYY-MM-DD"},
                "end": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["start", "end"],
        },
    },
    "calculator": {
        "fn": calculator,
        "description": "Evaluates basic arithmetic like '365 - 42' or '3 * (4 + 5)'. "
                       "Use it instead of doing math in your head.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


def select_tools(date_tool: bool = True) -> dict:
    return {n: t for n, t in ALL_TOOLS.items() if date_tool or n != "get_current_date"}


def tool_config(tools: dict, guardrail: bool = True) -> types.GenerateContentConfig:
    declarations = []
    for name, t in tools.items():
        decl = {"name": name, "description": t["description"]}
        if t["parameters"]:
            decl["parameters"] = t["parameters"]
        declarations.append(decl)
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT if guardrail else None,
        tools=[types.Tool(function_declarations=declarations)],
        # We run the loop ourselves, so turn off the SDK's automatic tool calling.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


# ---------------------------------------------------------------------------
# 3. Execute a tool call: never crash, always return a structured result
# ---------------------------------------------------------------------------
def execute(name: str, args: dict, tools: dict) -> dict:
    tool = tools.get(name)
    if tool is None:
        return {"ok": False, "error": f"Unknown tool '{name}'. Available: {list(tools)}"}
    try:
        return {"ok": True, "data": tool["fn"](**args)}
    except Exception as e:  # the model gets the error and can recover
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def tool_result_part(call, result: dict) -> types.Part:
    return types.Part(function_response=types.FunctionResponse(
        id=call.id, name=call.name, response=result))


# ---------------------------------------------------------------------------
# 4. The loop
# ---------------------------------------------------------------------------
def run_agent(task: str, date_tool: bool = True, guardrail: bool = True) -> str:
    tools = select_tools(date_tool)
    config = tool_config(tools, guardrail)
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]
    seen_calls = set()

    for step in range(1, MAX_STEPS + 1):
        try:
            response = generate(contents, config)
        except Exception as e:  # stop cleanly instead of a giant traceback
            return f"Stopped: model call failed ({type(e).__name__}: {str(e)[:200]})"

        # Append the model's turn exactly as returned (keeps Gemini's internal
        # "thought signatures", which it needs on the next call).
        contents.append(response.candidates[0].content)

        usage = response.usage_metadata
        print(f"\n--- step {step} | tokens in={usage.prompt_token_count} "
              f"out={usage.candidates_token_count}")

        tool_calls = response.function_calls or []
        for part in response.candidates[0].content.parts or []:
            if part.text and part.text.strip() and not part.thought:
                print(f"model: {part.text.strip()}")

        if not tool_calls:  # no tool call = the model decided it's done
            return response.text or "(empty answer)"

        result_parts = []
        for call in tool_calls:
            args = dict(call.args or {})
            key = (call.name, json.dumps(args, sort_keys=True))
            if key in seen_calls:  # loop detection
                result = {"ok": False, "error": "You already made this exact call. "
                                                "Use the earlier result or try something else."}
            else:
                seen_calls.add(key)
                result = execute(call.name, args, tools)
            print(f"tool:  {call.name}({args}) -> {result}")
            result_parts.append(tool_result_part(call, result))
        contents.append(types.Content(role="user", parts=result_parts))

    return "Stopped: reached max steps without a final answer."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesson 1: the agent loop")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--no-date-tool", action="store_true",
                        help="remove get_current_date so the agent can't see today's date")
    parser.add_argument("--no-guardrail", action="store_true",
                        help="remove the 'don't guess, ask' system instruction")
    args = parser.parse_args()
    require_api_key()

    question = " ".join(args.question) or "How many days are left until Christmas this year?"
    print(f"TASK: {question}   (model: {MODEL})")
    answer = run_agent(question, date_tool=not args.no_date_tool,
                       guardrail=not args.no_guardrail)
    print("\nFINAL ANSWER:", answer)
