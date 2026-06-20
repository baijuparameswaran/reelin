"""Scene agent: segment the story into a numbered scene list.

Uses the structural beats (when available) as a scaffold so the scene list maps
onto the three-act shape rather than just following the prose order.
"""
from __future__ import annotations

import json

from .. import llm
from ..llm import MAX_CHARS

SYSTEM = (
    "You are a screenwriter breaking a story into filmable scenes. Each scene "
    "happens in one location and continuous time. You always respond with valid "
    "JSON and nothing else."
)

PROMPT = """\
Break the following story into a sequence of filmable scenes (aim for {target}).
Use the structural beats as guidance for ordering and emphasis.

Respond with JSON in exactly this shape:
{{
  "scenes": [
    {{
      "number": 1,
      "slugline": "INT./EXT. LOCATION - DAY/NIGHT",
      "summary": "one or two sentences of what happens",
      "characters": ["NAME", "..."],
      "purpose": "why this scene exists dramatically"
    }}
  ]
}}

STRUCTURAL BEATS:
{beats}

SOURCE MATERIAL (title: {title}):
\"\"\"
{text}
\"\"\"
"""


def segment_scenes(
    source: dict,
    structure: dict,
    target: str = "8-14 scenes",
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("scenes")
    beats = json.dumps(structure.get("three_act", {}), ensure_ascii=False, indent=2)
    prompt = llm.with_feedback(
        PROMPT.format(
            target=target,
            beats=beats,
            title=source["title"],
            text=source["text"][:MAX_CHARS],
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
