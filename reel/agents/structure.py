"""Structure agent: derive logline, themes, and a three-act beat sheet.

Runs independently of character extraction, so the pipeline executes the two
concurrently.
"""
from __future__ import annotations

from .. import llm

SYSTEM = (
    "You are one of the most respected story analysts and screenwriters "
    "working today. You read source material and distill its dramatic "
    "structure precisely and concisely. You always respond with valid JSON "
    "and nothing else."
)

PROMPT = """\
Analyze the following source material and return its dramatic structure.

Do NOT:
- invent beats, themes, a conflict, or a genre the source material doesn't
  actually support
- pad the three_act lists with filler or restated beats just to seem thorough
- add any commentary, preamble, or markdown code fences before or after the
  JSON object

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present, no extra top-level keys, no missing keys:
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


def analyze_structure(
    source: dict, profile: str | None = None, feedback: str | None = None
) -> dict:
    profile = profile or llm.agent_profile("structure")
    prompt = llm.with_feedback(
        PROMPT.format(title=source["title"],
                      text=source["text"][:llm.max_chars(profile)]),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
