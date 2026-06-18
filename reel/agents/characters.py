"""Character agent: extract the cast with descriptions and arcs.

Runs independently of structure analysis (the pipeline runs them concurrently).
"""
from __future__ import annotations

from .. import llm

SYSTEM = (
    "You are a casting-minded script analyst. You identify characters and their "
    "dramatic function from source material. You always respond with valid JSON "
    "and nothing else."
)

PROMPT = """\
Identify the characters in the following source material.

Respond with JSON in exactly this shape:
{{
  "characters": [
    {{
      "name": "NAME",
      "role": "protagonist | antagonist | supporting | minor",
      "description": "one or two sentences (age, look, essence)",
      "want": "what they pursue in the story",
      "arc": "how they change, or 'static'",
      "traits": ["trait", "..."]
    }}
  ]
}}

List the most important characters first. SOURCE MATERIAL (title: {title}):
\"\"\"
{text}
\"\"\"
"""

MAX_CHARS = 12000


def extract_characters(source: dict, profile: str | None = None) -> dict:
    profile = profile or llm.agent_profile("characters")
    prompt = PROMPT.format(title=source["title"], text=source["text"][:MAX_CHARS])
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
