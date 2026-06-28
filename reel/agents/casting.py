"""Casting agent: lock each character's visual form into a renderable spec.

Runs after the character breakdown is approved. Where the character agent gives
essence, arc, and rough appearance, the casting agent commits to a single
coherent, image-generation-ready physical form per character — the look a
casting director, costume designer, and concept artist would all share. This
covers non-human characters too (animals, birds, creatures) and any background
"group" the character agent chose to treat as one.

Genre and tone steer the casting sensibility (a noir antagonist vs. a comedy
antagonist read very differently).

Two layers, kept separate the way a real production does:
  * actor      — the performer cast in the role: their OWN intrinsic, role-
                 independent features (face, build, bearing). An invented but
                 specific, consistent person. This is the casting choice itself
                 and the identity anchor that keeps the face consistent shot to
                 shot.
  * character  — what that actor is TRANSFORMED into for the role: age (makeup/
                 prosthetics), costume, mannerism, and any specific changes that
                 turn the actor into the character.

Per character entry:
  actor.casting_brief   — archetype / casting vibe ("weathered character actor, 70s feel")
  actor.features        — the actor's own face/build/bearing, independent of the role
  actor.visual_prompt   — prompt to render the actor as themselves (neutral)
  character.physical_form — the full on-screen look = actor + transformation, image-ready
  character.age         — how the actor is aged up/down for the role
  character.costume     — signature wardrobe / silhouette (natural coat/plumage for animals)
  character.mannerism   — posture / bearing / gesture that sells the character
  character.defining_feature — the single visual detail that reads instantly on screen
  character.visual_prompt — prompt to render the actor transformed into the character
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

Cast an ACTOR for each role (their own intrinsic look), then describe the
TRANSFORMATION that turns that actor into the character. Keep the two separate.

Respond with JSON in exactly this shape:
{{
  "casting": [
    {{
      "name": "NAME (match the character breakdown exactly)",
      "kind": "person | animal | bird | creature | group (copy from the input)",
      "actor": {{
        "casting_brief": "archetype / casting type and vibe — for a person \
'weathered Nordic character actor, late-career'; for an animal the breed/species \
and temperament; for a group the collective casting identity",
        "features": "the ACTOR's OWN intrinsic, role-independent look — an \
invented but source-grounded person: ONLY include attributes explicitly described \
or directly implied by the source story (e.g. 'tall', 'grey-haired'). Do NOT add \
face shape, eye colour, or hair texture unless the story states them. Keep \
unspecified attributes generic ('medium build', 'indeterminate age range'). \
Animals/birds: species/breed, size, coloring/markings only as described in source. \
This is the identity anchor reused to keep the look consistent across every shot",
        "visual_prompt": "a concise text-to-image prompt to render the ACTOR as \
THEMSELVES — a clear portrait (framing open: headshot through full figure), \
neutral expression, plain studio background, everyday neutral clothing, NO \
character costume or age makeup"
      }},
      "character": {{
        "physical_form": "the full on-screen look = the actor PLUS the \
transformation (age + costume + mannerism). One coherent, image-ready head-to-toe \
description, internally consistent and reusable across every scene",
        "age": "how the actor is aged up/down for the role (e.g. 'aged ~20 years \
via makeup and prosthetic lines'); 'as cast' if no change",
        "costume": "signature costume, fabric, silhouette, condition — or for an \
animal its natural coat/plumage and any worn item (collar, tag); 'n/a' if none",
        "mannerism": "the posture / bearing / gesture that sells the character on \
screen (e.g. 'hunched, knotted grip')",
        "defining_feature": "the single visual detail that identifies them instantly",
        "visual_prompt": "a concise text-to-image prompt to render the ACTOR \
TRANSFORMED into the character — a clear character portrait (framing open: headshot \
through full figure as suits the role) for an individual, or a representative shot \
for a group, fusing the actor's features + age + costume + mannerism + \
defining_feature + lighting mood"
      }}
    }}
  ]
}}

Rules:
- Exactly one casting entry per input character, names and kinds matching exactly —
  this includes every animal, bird, and creature, each cast individually.
- For a "group" input, cast it as one entry describing the ensemble and a
  representative member (do not invent individuals the breakdown didn't name).
- The actor's `features` carry the actor (face/build), NOT the role — keep age,
  costume, and weathering out of `features` and in the `character` block.
- `character.visual_prompt` must read as the SAME person from `actor.features`,
  just aged/costumed/styled into the role.
- Invent the actor; do NOT name or imitate a real, identifiable person.
- character.physical_form must be internally consistent and reusable across scenes.
- genre and tone should color the casting (gritty drama vs. heightened fantasy, etc.)
- STORY FIDELITY: Do NOT add physical attributes (face shape, eye colour, hair
  texture, body proportions, skin tone) that the source character description does
  not mention. Unspecified attributes stay generic. Descriptors like 'weathered',
  'gaunt', or 'imposing' are only valid if the story uses them or clearly implies
  them — do not intensify or elaborate beyond the source.

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
