"""Cinematography agent: shot-by-shot camera plan for every scene.

Plays the Director of Photography role — camera positions, shot types, angles,
movement, and lens choices for each dramatic beat. Processes all scenes in one
prompt for continuity: a tracking shot that bleeds into the next scene, a lens
shift that marks a character's psychological state, a repeated angle that becomes
a motif.

Distinct from the visuals agent (art production — color, props, sets) and the
soundscape agent (background score / sound design). Together the three cover the
core creative crew below the director.

Per scene:
  coverage        — overall shooting strategy (e.g. "intimate, handheld coverage")
  shots           — ordered list of camera set-ups per beat
    shot_number   — position in the coverage sequence
    moment        — the dramatic beat this shot serves
    type          — wide / medium / close-up / extreme-close-up / two-shot /
                    over-shoulder / insert / POV / establishing
    angle         — eye-level / low / high / bird's-eye / Dutch-tilt / worm's-eye
    movement      — static / pan / tilt / dolly-in / dolly-out / tracking /
                    crane-up / crane-down / handheld / Steadicam / whip-pan /
                    zoom-in / zoom-out
    lens          — wide-angle / normal / telephoto / macro / anamorphic
    framing       — compositional note (rule of thirds, centered, negative space…)
    emotional_function — what this shot communicates to the audience
  transition_to_next — cut / dissolve / fade-to-black / match-cut / smash-cut /
                        wipe / jump-cut
"""
from __future__ import annotations

import json

from .. import llm
from ..revision_merge import merge_by_key

SYSTEM = (
    "You are an award-winning Director of Photography. You design the camera "
    "coverage for every scene in a screenplay — shot types, angles, movement, "
    "lens choices, and transitions — with a rigorous eye for visual storytelling "
    "continuity, genre grammar, and emotional impact. "
    "You always respond with valid JSON and nothing else."
)

PROMPT = """\
Design the camera coverage for every scene in the following screenplay outline.

Film details:
- Logline: {logline}
- Genre: {genre}
- Tone: {tone}
- Themes: {themes}

Process all scenes together so that camera continuity, recurring motifs, and \
lens/movement language are coherent across the whole film.

Respond with JSON in exactly this shape:
{{
  "cinematography_style": "one sentence — the overall camera philosophy for this film",
  "dominant_movement": "the primary camera movement language (e.g. 'handheld and restless')",
  "scenes": [
    {{
      "scene_number": 1,
      "coverage": "overall shooting strategy for this scene",
      "shots": [
        {{
          "shot_number": 1,
          "moment": "the dramatic beat or action this shot covers",
          "type": "wide shot | medium shot | close-up | extreme close-up | two-shot | \
over-the-shoulder | insert | POV shot | establishing shot",
          "angle": "eye-level | low angle | high angle | bird's eye view | Dutch tilt | worms eye",
          "movement": "static | pan | tilt | dolly in | dolly out | tracking | \
crane up | crane down | handheld | Steadicam | aerial view | zoom in | zoom out",
          "lens": "wide-angle lens | normal lens | telephoto lens | macro lens | anamorphic lens",
          "framing": "specific compositional note",
          "emotional_function": "what this shot communicates to the audience"
        }}
      ],
      "transition_to_next": "cut | dissolve | fade-to-black | match-cut | \
smash-cut | wipe | jump-cut (empty string for final scene)"
    }}
  ]
}}

Rules:
- Each scene should have at least 2 shots; aim for a realistic coverage plan{duration_rule}
- Shot types and movement should reflect the genre \
(thriller → tight, handheld; drama → measured, Steadicam; horror → Dutch tilts, \
low angles; romance → soft telephoto, slow dolly)
- Motifs (a recurring angle, a recurring lens choice) should develop across scenes
- transition_to_next should be empty string for the final scene
- Scenes sharing the same `location` may reuse establishing/wide shots that read \
the same fixed space (that place has a locked, rendered layout) — vary angle, \
movement, and lens to keep coverage distinct, but don't imply a different \
architecture or layout than an earlier scene at the same place

SCENE LIST:
{scenes}
"""


def plan_cinematography(
    structure: dict,
    scenes: dict,
    profile: str | None = None,
    feedback: str | None = None,
    existing: dict | None = None,
    revise_keys: set | None = None,
    shots_guidance: str = "",
) -> dict:
    """`existing` + `revise_keys` (a set of `scene_number`s) support a scoped
    revision — see `reel.agents.soundscape.design_soundscape`'s docstring for
    the shared rationale. This artifact's list field is `"scenes"`, keyed by
    `scene_number`. A revised scene's `shots` list is replaced wholesale
    (not independently merge-spliced per `shot_number`) — no downstream
    consumer needs shot-level granularity the way video needs panel-level
    granularity, so the extra merge complexity has no payoff yet.

    `shots_guidance` (optional, from `reel.duration_budget.suggest_shots_per_scene`)
    is a budget hint tying shot coverage to the pipeline's target total
    runtime — engine-independent (it's about story/shot COUNT, not any video
    backend's own duration capability); empty string is a no-op (today's
    unguided behavior, "aim for a realistic coverage plan")."""
    profile = profile or llm.agent_profile("cinematography")
    scene_list = json.dumps(
        [
            {k: s[k] for k in ("number", "slugline", "location", "summary", "purpose",
                                "source_line", "chunk_indices")
             if k in s}
            for s in scenes.get("scenes", [])
        ],
        ensure_ascii=False,
        indent=2,
    )
    prompt = llm.with_feedback(
        PROMPT.format(
            logline=structure.get("logline", ""),
            genre=structure.get("genre", "drama"),
            tone=structure.get("tone", ""),
            themes=", ".join(structure.get("themes", [])),
            duration_rule=(f" — {shots_guidance}" if shots_guidance else ""),
            scenes=scene_list,
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    result = llm.safe_json(raw)
    if revise_keys is not None and existing:
        result["scenes"] = merge_by_key(
            existing.get("scenes", []), result.get("scenes", []),
            lambda s: s.get("scene_number"), revise_keys,
        )
    return result
