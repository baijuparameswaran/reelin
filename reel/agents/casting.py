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

Also casts recurring PROPS (kind: "prop") — a physical object from scenes.py's
`props` field (its rule 10) that appears in 2+ distinct scenes anywhere in the
story (`_prop_entries`; a single-scene prop isn't cast — no repetition to keep
consistent). Same rationale as locations: Veo has no memory across separately
generated clips, so a named prop with no locked identity can render as a
visibly different object every time it appears — a very descriptive,
consistent `character.visual_prompt` (material, color/finish, size, condition,
distinguishing marks), reused verbatim, is the text-only substitute for that
missing cross-generation memory. Like a location, a prop carries no `actor`
layer, and — unlike a location — a prop's own reference gets an actual
rendered image too via the same kind-agnostic `pipeline._render_casting_images`
(no code change needed there), which then feeds identically into every panel's
Veo Context section via `pipeline._panel_context`'s casting_lookup resolution.
Note the deliberate scope boundary: a cast prop's rendered PNG is NOT wired
into Veo's `reference_images`/seed-image mechanism the way a character's
portrait is (that mechanism is reserved, per the multi-reference-image
investigation noted elsewhere in this project, for the in-frame character
identity anchor) — only the prop's TEXT description reaches the render prompt.
"""
from __future__ import annotations

import json

from .. import llm
from ..revision_merge import merge_by_key

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
      "name": "LOCATION NAME (must exactly match a name in LOCATIONS INPUT below \
— this is a placeholder illustrating the shape, not a real place to include)",
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
    }},
    {{
      "name": "PROP NAME (must exactly match a name in PROPS INPUT below — this \
is a placeholder illustrating the shape, not a real object to include)",
      "kind": "prop",
      "character": {{
        "visual_prompt": "a VERY DESCRIPTIVE text-to-image prompt naming the \
object's exact material, color/finish, size/scale, condition or wear, and any \
distinguishing mark or engraving — specific enough that this exact object reads \
as the SAME object every time it's mentioned again, ending with the fixed clause \
'plain neutral background, no hands, no scene, no other objects, even studio \
lighting'. This is a bare identity reference reused verbatim across every scene \
the prop appears in, so anything scene-shaped baked in here (who's holding it, \
where it is, what's happening) would leak into and bias every one of those \
otherwise-unrelated scenes."
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
as a background plate for every scene set there. If a LOCATIONS INPUT entry
below has a `recurring_props` list, treat it as CANDIDATES, not a mandate:
weave in only the ones that read as a FIXED, permanent part of the space
itself (a mounted trophy, a built-in bar, a standing lamp) — leave out
anything that sounds like it's tied to one scene's specific action or is
carried/held by a character (that stays scene-specific, handled elsewhere,
not baked into this reused reference).

PROPS: entries with "kind": "prop" (see PROPS INPUT below, if provided) are
individual objects, not people or places — they carry no `actor` block at all
(omit `actor` entirely), just the `character.visual_prompt` shown in the
example above. This is the SAME isolation discipline as a character portrait
(strict, see the ISOLATION rule below — it applies to props too, not just
person/animal/bird/creature/group entries) but the payoff is different: the
description must be VERY DESCRIPTIVE and SPECIFIC — exact material, color,
size, condition, engravings or wear marks — precise enough that the object
renders identically every time it recurs across separate scenes and separate
video-generation calls, which have no memory of each other. A vague
description ("an old watch") defeats the entire purpose of casting it; be as
concrete as a prop master's own spec sheet.

Rules:
- Exactly one casting entry per input character PLUS exactly one entry per
  distinct entry in LOCATIONS INPUT below (if provided) PLUS exactly one entry
  per distinct entry in PROPS INPUT below (if provided) — names and kinds
  matching exactly. This includes every animal, bird, and creature, each cast
  individually, every distinct location, each cast once regardless of how
  many scenes share it, and every distinct recurring prop, each cast once
  regardless of how many scenes it appears in.
- For a "group" input, cast it as one entry describing the ensemble and a
  representative member (do not invent individuals the breakdown didn't name).
- The actor's `features` carry the actor (face/build), NOT the role — keep age,
  costume, and weathering out of `features` and in the `character` block.
- `character.visual_prompt` must read as the SAME person from `actor.features`,
  just aged/costumed/styled into the role.
- Invent the actor; do NOT name or imitate a real, identifiable person. This
  "invent" only concerns WHO plays the role — a fictional, non-real-world
  person — it does NOT relax the STORY FIDELITY rule below on WHICH physical
  attributes that invented person has: invent the person, ground their
  attributes in the source.
- character.physical_form must be internally consistent and reusable across
  scenes, but — like `character.visual_prompt` — stays context-free: no
  scene, location, prop, or lighting mention, just the person/creature and
  their costume/mannerism. It differs from `visual_prompt` only in being a
  descriptive field rather than an image-generation prompt; the ISOLATION
  rule below applies equally to both.
- ISOLATION (strict, person/animal/bird/creature/group AND prop entries —
  LOCATIONS invert this, see the LOCATIONS rule above): `character.visual_prompt`
  (like `actor.visual_prompt`) describes ONLY the person/creature and their
  costume — or, for a prop, ONLY the object itself — end it with the fixed
  backdrop clause given above (its own version for props, see the PROPS
  example), verbatim. Never mention a scene, location, another prop, a
  character, time of day, or mood lighting — not even the character's own
  signature location from the story (a bar, a boat, a workshop), and for a
  prop, not who holds it or where it normally sits. This image seeds video
  identity for every scene the character (or prop) appears in, so anything
  scene-shaped baked in here would leak into and bias every one of those
  otherwise-unrelated scenes. If you catch yourself naming a place or another
  object that isn't worn on the character's body (or, for a prop, isn't the
  prop itself), cut it.
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
{locations_block}{props_block}
Before you respond, re-check against the breakdown above (long input pushes
early rules out of recent context — re-verify against what you just read, not
just what you remember from the rules list):
- ISOLATION: every person/animal/bird/creature/group/prop `character.visual_prompt`
  (and `physical_form`) names ONLY the body/object and costume, ends with the fixed
  backdrop clause verbatim, and contains no scene/location/other-prop/other-character/
  lighting/time-of-day word — not even the character's own signature location, and
  for a prop, not who holds it or where it normally sits.
- STORY FIDELITY: no physical attribute (face shape, eye colour, hair texture,
  body proportions, skin tone) beyond what the breakdown above actually states
  or clearly implies.
- GENDER: pronouns/gender-marked nouns match the breakdown above exactly, held
  consistent across `actor.features`, `actor.visual_prompt`, `physical_form`,
  and `character.visual_prompt` — never defaulted or guessed.
- PROPS: every prop's `visual_prompt` is specific enough (material, color, size,
  condition, marks) to render as the SAME object every time it recurs — not a
  generic description.
"""


def _location_entries(scenes: dict) -> list[dict]:
    """Distinct `location` values across scenes.json, one entry each regardless
    of how many scenes share it. No LLM call — pure dedup, grounded by
    aggregating the sluglines/summaries of every scene set there so the casting
    call has something concrete to work from (scenes.py already ensures the
    same real place always uses the identical `location` string).

    Also aggregates every scene's `props` (scenes.py rule 10 — source-grounded
    physical objects) into a `recurring_props` list: an object that shows up
    in MORE THAN ONE scene set at this location is plausibly a fixed part of
    the place itself (a bar's brass mirror, a lighthouse's lamp) rather than
    something tied to one scene's specific action — a single-scene prop (a
    letter someone is holding) is deliberately excluded here, since baking a
    transient/action prop into the location's own reused reference image
    would wrongly force it into every OTHER scene at that place too. This is
    a heuristic (recurrence across scenes), not certainty — the PROMPT below
    still tells the model to only keep genuinely fixed/architectural items
    from this list, not just echo it wholesale."""
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
        prop_counts: dict[str, int] = {}
        for sc in scs:
            for p in sc.get("props") or []:
                p = (p or "").strip()
                if p:
                    prop_counts[p] = prop_counts.get(p, 0) + 1
        recurring = sorted((p for p, n in prop_counts.items() if n > 1 or len(scs) == 1),
                           key=lambda p: -prop_counts[p])
        entry = {
            "name": name,
            "kind": "location",
            "appearance": "; ".join(sluglines),
            "description": summaries[:400],
        }
        if recurring:
            entry["recurring_props"] = recurring
        entries.append(entry)
    return entries


def _prop_entries(scenes: dict) -> list[dict]:
    """Distinct props worth casting their own locked identity: a prop name
    (exact string match — scenes.py rule 10 asks the model to reuse the same
    string for the same recurring object, the same consistency treatment
    `location` already gets in rule 8; a near-miss like "the watch" vs "a
    tarnished pocket watch" isn't caught here, a known limitation shared with
    `_map_chunks`'s identical source_line-matching approach) that appears in
    2 OR MORE distinct scenes ANYWHERE in the story — not scoped to one
    location, since a portable prop (a character's watch, a letter) can
    travel across locations the way a location-fixed prop (a bar's mirror)
    can't. A prop mentioned in only one scene isn't cast: no repetition
    means no consistency problem to solve, and casting every single-scene
    object would be excessive both in render cost and screen clutter. No LLM
    call — pure aggregation over scenes.json, the same shape as
    `_location_entries`."""
    counts: dict[str, int] = {}
    for sc in scenes.get("scenes", []):
        for p in sc.get("props") or []:
            p = (p or "").strip()
            if p:
                counts[p] = counts.get(p, 0) + 1
    return [{"name": p, "kind": "prop", "scene_count": n}
            for p, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n > 1]


def cast_characters(
    structure: dict,
    characters: dict,
    profile: str | None = None,
    feedback: str | None = None,
    scenes: dict | None = None,
    existing: dict | None = None,
    revise_keys: set | None = None,
) -> dict:
    """`existing` + `revise_keys` (a set of casting `name`s) support a scoped
    revision: the LLM still sees the FULL character/location breakdown (its
    casting-consistency rules need whole-cast context — e.g. distinct actors
    per role), but the caller only trusts the response for the names in
    `revise_keys`; every other name is spliced back in byte-identical from
    `existing["casting"]` via `revision_merge.merge_by_key`. This is what
    keeps an untouched character/location's `visual_prompt` text — and thus
    its `_content_hash` in `pipeline._render_casting_images` — stable, so its
    already-rendered reference image is reused rather than regenerated from
    incidental LLM rewording. A genuinely new name (not in `existing` at all)
    is only ever added if it's ALSO in `revise_keys` (e.g. because
    `artifact_diff.diff_artifact` identified it as a genuinely added entry in
    a direct hand-edit) — never picked up unprompted just because the
    model's full-context response happened to include it (see
    `revision_merge.merge_by_key`'s docstring)."""
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
    props = _prop_entries(scenes) if scenes else []
    props_block = (
        f"\nPROPS INPUT — one casting entry per distinct recurring prop (kind: \"prop\"):\n"
        f"{json.dumps(props, ensure_ascii=False, indent=2)}\n"
    ) if props else ""
    prompt = llm.with_feedback(
        PROMPT.format(
            logline=structure.get("logline", ""),
            genre=structure.get("genre", "drama"),
            tone=structure.get("tone", ""),
            characters=cast_input,
            locations_block=locations_block,
            props_block=props_block,
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    result = llm.safe_json(raw)
    if revise_keys is not None and existing:
        result["casting"] = merge_by_key(
            existing.get("casting", []), result.get("casting", []),
            lambda c: c.get("name"), revise_keys,
        )
    return result
