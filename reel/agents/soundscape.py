"""Soundscape agent: design the audio world for every scene.

Processes all scenes in a single prompt so the LLM can reason about
cross-scene continuity — a recurring ocean bed that dims as a character
enters a car, a building industrial hum that bleeds into the next scene.

Each scene gets:
  ambient_bed     — the continuous audio floor (or silent)
  sound_events    — moment-keyed transient cues within the scene
  score_direction — the background score/music direction (see below)
  transition_to_next — how audio evolves at the scene boundary
  silence         — true when silence itself is the dramatic choice
  emotional_function — why this audio choice serves the story

Genre and tone from the structure agent seed the overall audio palette.

`score_direction` fixes a real, previously-latent bug: `storyboard.py`'s
`_build_scene_board` has always read `bundle.audio.score_direction` into
`audio_overview.score_cue` (the field `pipeline._panel_video_prompt`
actually asserts as Veo's music directive), but this schema never asked
the model to produce it — every scene's score direction has always been
silently empty.
"""
from __future__ import annotations

import json

from .. import llm
from ..revision_merge import merge_by_key

SYSTEM = (
    "You are an award-winning film sound designer and music supervisor, among "
    "the best in the business. You craft the audio world of a screenplay — "
    "ambient beds, diegetic transients, silence — with an ear for emotional "
    "truth, genre convention, and cross-scene continuity. You always respond "
    "with valid JSON and nothing else."
)

PROMPT = """\
Design the soundscape for every scene in the following screenplay outline.

Film details:
- Logline: {logline}
- Genre: {genre}
- Tone: {tone}
- Themes: {themes}

Process all scenes together so that audio continuity across scenes is \
intentional and consistent.

DO NOT:
- give two scenes sharing a `location` different base `ambient_bed`s
- invent a sound_event with no grounding in the scene's summary/purpose
- skip, renumber, or duplicate a `scene_number` relative to the SCENE LIST
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present for every scene, no extra top-level keys, no missing keys:
{{
  "audio_palette": "one sentence describing the overall sonic world of this film",
  "soundscapes": [
    {{
      "scene_number": 1,
      "ambient_bed": "the continuous audio texture filling this scene (empty string if \
truly silent)",
      "sound_events": [
        {{
          "moment": "brief description of the beat within the scene",
          "sound": "specific transient sound at that moment"
        }}
      ],
      "score_direction": "the BACKGROUND SCORE/MUSIC for this scene — instrumentation, \
mood, presence — separate from ambient_bed (environmental sound) and sound_events (SFX); \
empty string for a scene that should carry no score at all",
      "transition_to_next": "how the audio evolves or carries over into the next scene",
      "silence": false,
      "emotional_function": "what this soundscape does for the audience emotionally"
    }}
  ]
}}

Rules:
- silence may be true when silence itself is the dramatic choice
- ambient_bed should be empty string (not null) when silence is true
- sound_events may be an empty list when no transients occur
- transition_to_next should be empty string for the final scene
- genre should influence the sonic palette (thriller → tension drones; \
drama → sparse naturalism; comedy → lighter textures, etc.)
- Scenes sharing the same `location` share the same base ambient_bed (that \
place has a fixed, rendered acoustic character) — only specific sound_events \
should vary between them, not the room tone itself

SCENE LIST:
{scenes}

Before you respond, re-check against the scene list above (long scene lists \
push early rules out of recent context — re-verify against what you just \
read, not just what you remember from the rules list):
- Every scene sharing a `location` with another keeps that place's base \
`ambient_bed` identical — only `sound_events` vary.
- `scene_number` in your output matches the `number` field from the scene list \
above exactly, one output scene per input scene, none skipped or renumbered.
- The response is ONLY the JSON object above — no markdown fences, no \
commentary, no extra top-level keys, every scene has all its schema fields.
"""


def design_soundscape(
    structure: dict,
    scenes: dict,
    profile: str | None = None,
    feedback: str | None = None,
    existing: dict | None = None,
    revise_keys: set | None = None,
) -> dict:
    """`existing` + `revise_keys` (a set of `scene_number`s) support a scoped
    revision: the model still designs the soundscape for the WHOLE scene list
    (cross-scene continuity — e.g. a shared-location ambient bed — needs full
    context), but the caller only trusts the response for the targeted scene
    numbers; every other scene's entry is spliced back in byte-identical from
    `existing["soundscapes"]` (note: this artifact's list field is
    `"soundscapes"`, not `"scenes"` like visuals/cinematography/screenplay)."""
    profile = profile or llm.agent_profile("soundscape")
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
            scenes=scene_list,
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    result = llm.safe_json(raw)
    if revise_keys is not None and existing:
        result["soundscapes"] = merge_by_key(
            existing.get("soundscapes", []), result.get("soundscapes", []),
            lambda s: s.get("scene_number"), revise_keys,
        )
    return result
