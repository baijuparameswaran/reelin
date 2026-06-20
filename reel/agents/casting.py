"""Casting agent: lock each character's visual form into a renderable spec.

Runs after the character breakdown is approved. Where the character agent gives
essence, arc, and rough appearance, the casting agent commits to a single
coherent, image-generation-ready physical form per character — the look a
casting director, costume designer, and concept artist would all share. This
covers non-human characters too (animals, birds, creatures) and any background
"group" the character agent chose to treat as one.

Genre and tone steer the casting sensibility (a noir antagonist vs. a comedy
antagonist read very differently).

Per character:
  casting_brief   — archetype / type + the casting vibe ("weathered character actor, 70s")
  physical_form   — one coherent head-to-toe description, image-ready
  wardrobe        — signature costume / silhouette (or natural coat/plumage for animals)
  defining_feature — the single visual detail that reads instantly on screen
  visual_prompt   — a concise text-to-image prompt to render this character
"""
from __future__ import annotations

import json

from .. import llm

SYSTEM = (
    "You are a film casting director working hand-in-hand with a costume "
    "designer, an animal wrangler, and a concept artist. You turn a character "
    "breakdown — humans, animals, birds, creatures — into a single, committed, "
    "visually concrete casting that will appear on screen. You always respond "
    "with valid JSON and nothing else."
)

PROMPT = """\
Lock the on-screen visual form of each character below into a final casting.

Film details:
- Logline: {logline}
- Genre: {genre}
- Tone: {tone}

Respond with JSON in exactly this shape:
{{
  "casting": [
    {{
      "name": "NAME (match the character breakdown exactly)",
      "kind": "person | animal | bird | creature | group (copy from the input)",
      "casting_brief": "archetype / casting type and vibe — for a person 'weathered \
Nordic character actor, late 70s'; for an animal the breed/species and temperament; \
for a group the collective identity",
      "physical_form": "one coherent physical description, specific enough to \
generate a consistent image. People: build, face, hair, age, skin, bearing. \
Animals/birds: species/breed, size, coloring, markings, plumage/coat, eyes. \
Group: the shared look plus one representative individual",
      "wardrobe": "signature costume, fabric, silhouette, condition — or for an \
animal its natural coat/plumage and any worn item (collar, tag); 'n/a' if none",
      "defining_feature": "the single visual detail that identifies them instantly",
      "visual_prompt": "a concise text-to-image prompt to render this character \
(a portrait for an individual; a representative shot for a group), fusing \
physical_form + wardrobe + defining_feature + lighting mood"
    }}
  ]
}}

Rules:
- Exactly one casting entry per input character, names and kinds matching exactly —
  this includes every animal, bird, and creature, each cast individually.
- For a "group" input, cast it as one entry describing the ensemble and a
  representative member (do not invent individuals the breakdown didn't name).
- physical_form must be internally consistent and reusable across every scene.
- genre and tone should color the casting (gritty drama vs. heightened fantasy, etc.)

CHARACTER BREAKDOWN:
{characters}
"""


def cast_characters(
    structure: dict,
    characters: dict,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("casting")
    cast_input = json.dumps(
        [
            {k: c[k] for k in
             ("name", "kind", "role", "description", "appearance", "voice", "mannerisms", "traits")
             if k in c}
            for c in characters.get("characters", [])
        ],
        ensure_ascii=False,
        indent=2,
    )
    prompt = llm.with_feedback(
        PROMPT.format(
            logline=structure.get("logline", ""),
            genre=structure.get("genre", "drama"),
            tone=structure.get("tone", ""),
            characters=cast_input,
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
