"""Soundscape agent: design the audio world for every scene.

Processes all scenes in a single prompt so the LLM can reason about
cross-scene continuity — a recurring ocean bed that dims as a character
enters a car, a building industrial hum that bleeds into the next scene.

Each scene gets:
  ambient_bed     — the continuous audio floor (or silent)
  sound_events    — moment-keyed transient cues within the scene
  transition_to_next — how audio evolves at the scene boundary
  silence         — true when silence itself is the dramatic choice
  emotional_function — why this audio choice serves the story

Genre and tone from the structure agent seed the overall audio palette.
"""
from __future__ import annotations

import json

from .. import llm

SYSTEM = (
    "You are a professional film sound designer and music supervisor. You craft "
    "the audio world of a screenplay — ambient beds, diegetic transients, silence "
    "— with an ear for emotional truth, genre convention, and cross-scene "
    "continuity. You always respond with valid JSON and nothing else."
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

Respond with JSON in exactly this shape:
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

SCENE LIST:
{scenes}
"""


def design_soundscape(
    structure: dict,
    scenes: dict,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("soundscape")
    scene_list = json.dumps(
        [
            {k: s[k] for k in ("number", "slugline", "summary", "purpose",
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
    return llm.safe_json(raw)
