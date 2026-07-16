"""Character agent: extract the cast with descriptions and arcs.

Runs independently of structure analysis (the pipeline runs them concurrently).
"""
from __future__ import annotations

from .. import llm
from ..llm import MAX_CHARS
from ..revision_merge import merge_by_key

SYSTEM = (
    "You are one of the best casting-minded script analysts in the industry. "
    "You identify characters and their dramatic function from source material — "
    "humans and non-humans alike (animals, birds, creatures). You always "
    "respond with valid JSON and nothing else."
)

PROMPT = """\
Identify every distinct character in the following source material, including
NON-HUMAN ones — animals, birds, or other creatures that appear or act in the
story. Define each character individually and fully.

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present for every character, no extra top-level keys, no missing keys,
no markdown code fences or commentary before or after the JSON:
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

Do NOT:
- invent a character, trait, or relationship the source material doesn't
  actually support
- merge two genuinely distinct characters into one entry, or split one
  character into two
- omit a "kind"/"role"/any other schema field for any character, or leave a
  field empty when the source gives you something to say
- add any character not present in the source, even a plausible-sounding one

SOURCE MATERIAL (title: {title}):
\"\"\"
{text}
\"\"\"
"""


def extract_characters(
    source: dict, profile: str | None = None, feedback: str | None = None,
    existing: dict | None = None, revise_keys: set | None = None,
) -> dict:
    """`existing` + `revise_keys` (a set of character `name`s) support a
    scoped revision — same pattern as `reel.agents.casting.cast_characters`:
    the model still sees the full source text, but the caller only trusts its
    output for the targeted names, splicing everything else back in
    byte-identical from `existing["characters"]`."""
    profile = profile or llm.agent_profile("characters")
    prompt = llm.with_feedback(
        PROMPT.format(title=source["title"], text=source["text"][:MAX_CHARS]),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    result = llm.safe_json(raw)
    if revise_keys is not None and existing:
        result["characters"] = merge_by_key(
            existing.get("characters", []), result.get("characters", []),
            lambda c: c.get("name"), revise_keys,
        )
    return result
