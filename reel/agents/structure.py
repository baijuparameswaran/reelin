"""Structure agent: derive logline, themes, and a three-act beat sheet.

Runs independently of character extraction, so the pipeline executes the two
concurrently.
"""
from __future__ import annotations

from .. import llm

SYSTEM = (
    "You are a veteran story analyst and screenwriter. You read source material "
    "and distill its dramatic structure precisely and concisely. You always "
    "respond with valid JSON and nothing else."
)

PROMPT = """\
Analyze the following source material and return its dramatic structure.

Respond with JSON in exactly this shape:
{{
  "logline": "one vivid sentence capturing protagonist, goal, and conflict",
  "genre": "primary genre",
  "themes": ["theme", "..."],
  "tone": "short description of tone/mood",
  "three_act": {{
    "act1_setup": ["beat", "..."],
    "act2_confrontation": ["beat", "..."],
    "act3_resolution": ["beat", "..."]
  }},
  "central_conflict": "one sentence"
}}

SOURCE MATERIAL (title: {title}):
\"\"\"
{text}
\"\"\"
"""

# Cap text sent to small local models to stay within context + memory limits.
MAX_CHARS = 12000


def analyze_structure(source: dict, profile: str | None = None) -> dict:
    profile = profile or llm.agent_profile("structure")
    prompt = PROMPT.format(title=source["title"], text=source["text"][:MAX_CHARS])
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
