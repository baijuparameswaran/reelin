"""The screenplay-material agents (iteration 1).

Each agent is a small, single-responsibility function that takes structured
input and returns structured output, talking to local models via `reel.llm`.
The pipeline (`reel.pipeline`) wires them together, running independent agents
concurrently where possible.
"""
