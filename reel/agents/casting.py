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

Also casts LOCATIONS (kind: "location") — the distinct physical settings scenes.py
already named consistently across scenes (its `location` field, one name per real
place regardless of scene count). These carry no `actor` layer, just a
`character.visual_prompt` describing the space itself (the inverse of a character
portrait's isolation rule: SHOW the architecture/decor, but exclude people, action,
and any one scene's specific mood/weather/time-of-day, since it's a background
reference reused across every scene set there). Requires `scenes` be passed to
`cast_characters` — the pipeline runs `casting` after `scenes` for this reason
(previously they ran concurrently).
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
        "visual_prompt": "a concise text-to-image prompt to render ONLY the ACTOR \
TRANSFORMED into the character — a clear character portrait (framing open: headshot \
through full figure as suits the role) for an individual, or a representative shot \
for a group, fusing ONLY the actor's features + age + costume + mannerism + \
defining_feature, ending with the fixed clause 'plain seamless studio backdrop, \
solid neutral grey, no scene, no props, no other people, no location of any kind, \
even lighting'. Even if the character's story context is a specific place (a bar, \
a boat, a battlefield), NONE of that place or its objects may appear here — not \
named, not implied, not in the background. This image is a bare identity \
reference reused as-is across every scene the character appears in, so anything \
scene-shaped baked in here (a setting, a prop, a mood) would bias or contradict \
every one of those otherwise-unrelated scenes."
      }}
    }},
    {{
      "name": "Rusty Anchor Bar (example — LOCATION entries only, see LOCATIONS rule)",
      "kind": "location",
      "character": {{
        "visual_prompt": "a concise text-to-image prompt describing ONLY the \
physical space itself — architecture, layout, materials, fixed decor, its \
characteristic light source — ending with the fixed clause 'empty of people, no \
characters, no figures, no action, evenly lit, no specific time of day'. This is \
a background/set reference reused across every scene set in this place, so it \
must show the place plainly without any scene's specific mood, weather, or \
population baked in."
      }}
    }}
  ]
}}

LOCATIONS: entries with "kind": "location" are physical settings, not people —
they carry no `actor` block at all (omit `actor` entirely for these), just the
`character.visual_prompt` shown in the example above. Unlike a character
portrait, a location reference SHOULD show the place's own defining
architecture/decor — that's the point — but must still exclude people, action,
and any single scene's specific weather/mood/time-of-day, since it is reused
as a background plate for every scene set there.

Rules:
- Exactly one casting entry per input character PLUS exactly one entry per
  distinct entry in LOCATIONS INPUT below (if provided) — names and kinds
  matching exactly. This includes every animal, bird, and creature, each cast
  individually, and every distinct location, each cast once regardless of how
  many scenes share it.
- For a "group" input, cast it as one entry describing the ensemble and a
  representative member (do not invent individuals the breakdown didn't name).
- The actor's `features` carry the actor (face/build), NOT the role — keep age,
  costume, and weathering out of `features` and in the `character` block.
- `character.visual_prompt` must read as the SAME person from `actor.features`,
  just aged/costumed/styled into the role.
- Invent the actor; do NOT name or imitate a real, identifiable person.
- character.physical_form must be internally consistent and reusable across scenes.
- ISOLATION (strict, person/animal/bird/creature/group entries only —
  LOCATIONS invert this, see the LOCATIONS rule above): `character.visual_prompt`
  (like `actor.visual_prompt`) describes ONLY the person/creature and their
  costume — end it with the fixed backdrop clause given above, verbatim. Never
  mention a scene, location, prop, other character, time of day, or mood
  lighting — not even the character's own signature location from the story (a
  bar, a boat, a workshop). This image seeds video identity for every scene the
  character appears in, so anything scene-shaped baked in here would leak into
  and bias every one of those otherwise-unrelated scenes. If you catch yourself
  naming a place or an object that isn't worn on the character's body, cut it.
- genre and tone should color the casting itself (silhouette, costume era, bearing)
  — gritty drama vs. heightened fantasy — never the backdrop/lighting of the render.
- STORY FIDELITY: Do NOT add physical attributes (face shape, eye colour, hair
  texture, body proportions, skin tone) that the source character description does
  not mention. Unspecified attributes stay generic. Descriptors like 'weathered',
  'gaunt', or 'imposing' are only valid if the story uses them or clearly implies
  them — do not intensify or elaborate beyond the source.
- GENDER: use the pronouns and/or gender-marked nouns already present in the
  character breakdown below (e.g. 'she', 'her', 'woman', 'he', 'him', 'man') and
  carry that gender through `actor.features`, `actor.visual_prompt`,
  `character.physical_form`, and `character.visual_prompt` consistently. Never
  default to male or guess a gender the breakdown doesn't establish. If the
  breakdown truly gives no gender cue at all, keep the casting gender-neutral
  rather than assigning one.

CHARACTER BREAKDOWN:
{characters}
{locations_block}"""


def _location_entries(scenes: dict) -> list[dict]:
    """Distinct `location` values across scenes.json, one entry each regardless
    of how many scenes share it. No LLM call — pure dedup, grounded by
    aggregating the sluglines/summaries of every scene set there so the casting
    call has something concrete to work from (scenes.py already ensures the
    same real place always uses the identical `location` string)."""
    by_name: dict[str, list[dict]] = {}
    for sc in scenes.get("scenes", []):
        loc = (sc.get("location") or "").strip()
        if not loc:
            continue
        by_name.setdefault(loc, []).append(sc)

    entries = []
    for name, scs in by_name.items():
        sluglines = sorted({s.get("slugline", "") for s in scs if s.get("slugline")})
        summaries = " ".join(s.get("summary", "") for s in scs if s.get("summary"))
        entries.append({
            "name": name,
            "kind": "location",
            "appearance": "; ".join(sluglines),
            "description": summaries[:400],
        })
    return entries


def cast_characters(
    structure: dict,
    characters: dict,
    profile: str | None = None,
    feedback: str | None = None,
    scenes: dict | None = None,
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
    locations = _location_entries(scenes) if scenes else []
    locations_block = (
        f"\nLOCATIONS INPUT — one casting entry per distinct location (kind: \"location\"):\n"
        f"{json.dumps(locations, ensure_ascii=False, indent=2)}\n"
    ) if locations else ""
    prompt = llm.with_feedback(
        PROMPT.format(
            logline=structure.get("logline", ""),
            genre=structure.get("genre", "drama"),
            tone=structure.get("tone", ""),
            characters=cast_input,
            locations_block=locations_block,
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
