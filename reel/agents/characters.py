"""Character agent: extract the cast with descriptions and arcs.

Runs independently of structure analysis (the pipeline runs them concurrently).
"""
from __future__ import annotations

from .. import llm
from ..llm import MAX_CHARS

SYSTEM = (
    "You are a casting-minded script analyst. You identify characters and their "
    "dramatic function from source material — humans and non-humans alike "
    "(animals, birds, creatures). You always respond with valid JSON and "
    "nothing else."
)

PROMPT = """\
Identify every distinct character in the following source material, including
NON-HUMAN ones — animals, birds, or other creatures that appear or act in the
story. Define each character individually and fully.

Respond with JSON in exactly this shape:
{{
  "characters": [
    {{
      "name": "NAME",
      "kind": "person | animal | bird | creature | group",
      "role": "protagonist | antagonist | supporting | minor",
      "description": "one or two sentences capturing essence and dramatic function "
                     "(for a person: age & inner life; for an animal: species & temperament)",
      "want": "what they pursue in the story (or instinct/drive for an animal)",
      "arc": "how they change, or 'static'",
      "traits": ["trait", "..."],
      "appearance": "physical look — for people: build, clothing, features; "
                    "for animals/birds: species/breed, size, coloring, markings, plumage/coat",
      "voice": "for people: speech pattern, accent, pace; "
               "for animals: characteristic sound (call, bark, screech, song)",
      "mannerisms": "recurring gestures or movement — gait, flight pattern, habits"
    }}
  ]
}}

Rules:
- Treat animals, birds, and creatures as characters in their own right and define
  each one individually whenever it is named, recurs, or carries dramatic weight.
- Collapse into a SINGLE entry with "kind": "group" ONLY a mass of undetailed,
  interchangeable background figures the story never individuates (e.g.
  "Villagers", "a flock of gulls", "the wolf pack"). If any member is given its
  own name or detail, break it out as its own character instead.
- List the most important characters first.

SOURCE MATERIAL (title: {title}):
\"\"\"
{text}
\"\"\"
"""


def extract_characters(
    source: dict, profile: str | None = None, feedback: str | None = None
) -> dict:
    profile = profile or llm.agent_profile("characters")
    prompt = llm.with_feedback(
        PROMPT.format(title=source["title"], text=source["text"][:MAX_CHARS]),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
