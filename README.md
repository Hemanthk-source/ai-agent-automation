# AI Agent Automation

A simple AI agent that takes a task in plain English, decides which tools to use, and carries it out step by step.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your Anthropic API key into .env
```

## Run
```bash
python agent.py "Write a file named plan.txt with a 3-day study plan"   # one-shot
python agent.py                                                         # interactive
```

## How it works
1. Your task + the tool list are sent to Claude.
2. If Claude wants a tool, `run_agent()` executes it and returns the result.
3. This repeats (max 10 steps) until Claude gives a final answer.

## Built-in tools
`get_current_time`, `calculator`, `write_file`, `read_file`, `list_files`, `fetch_url`
(file access is restricted to the `workspace/` folder).

## Add your own tool
1. Write a Python function.
2. Add it to `TOOL_FUNCTIONS`.
3. Add its schema to `TOOLS`.
