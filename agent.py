"""
AI Agent Automation - a simple tool-using agent built on the Anthropic API.

Usage:
    python agent.py "Create a file todo.txt with 3 study tasks"   # one-shot task
    python agent.py                                               # interactive mode
"""

import ast
import json
import operator
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5-5"
MAX_STEPS = 10  # safety limit on the agent loop
WORKSPACE = Path("workspace")  # the agent can only touch files inside this folder
WORKSPACE.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are an automation agent. Break the user's task into steps and use the "
    "available tools to complete it. When finished, give a short summary of what you did."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


# --------------------------------------------------------------------------
# Tools (plain Python functions)
# --------------------------------------------------------------------------
def get_current_time() -> str:
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval").body))


def _safe_path(filename: str) -> Path:
    path = (WORKSPACE / filename).resolve()
    if WORKSPACE.resolve() not in path.parents and path != WORKSPACE.resolve():
        raise ValueError("Access outside the workspace is not allowed")
    return path


def write_file(filename: str, content: str) -> str:
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {filename}"


def read_file(filename: str) -> str:
    return _safe_path(filename).read_text(encoding="utf-8")


def list_files() -> str:
    files = [str(p.relative_to(WORKSPACE)) for p in WORKSPACE.rglob("*") if p.is_file()]
    return "\n".join(files) if files else "(workspace is empty)"


def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-agent-automation/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="ignore")[:5000]  # cap output size


# Map tool names to functions
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "write_file": write_file,
    "read_file": read_file,
    "list_files": list_files,
    "fetch_url": fetch_url,
}

# Tool schemas the model sees
TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression, e.g. '(12 + 8) * 3'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file from the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files in the workspace.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fetch_url",
        "description": "Download the text content of a web page (first 5000 characters).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------
def run_agent(task: str, history: list | None = None) -> str:
    messages = history if history is not None else []
    messages.append({"role": "user", "content": task})

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # No more tool calls -> the agent is done
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # Run each requested tool and send results back
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [step {step}] {block.name}({json.dumps(block.input)})")
                try:
                    output = TOOL_FUNCTIONS[block.name](**block.input)
                except Exception as e:  # report errors to the model so it can recover
                    output = f"Error: {e}"
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(output)}
                )
        messages.append({"role": "user", "content": results})

    return "Stopped: reached the maximum number of steps."


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Set ANTHROPIC_API_KEY first (see README.md).")

    if len(sys.argv) > 1:  # one-shot mode
        print("\n" + run_agent(" ".join(sys.argv[1:])))
        return

    print("AI Agent Automation - type a task (or 'exit' to quit)")
    history: list = []
    while True:
        task = input("\nTask> ").strip()
        if task.lower() in {"exit", "quit"}:
            break
        if task:
            print("\n" + run_agent(task, history))


if __name__ == "__main__":
    main()
