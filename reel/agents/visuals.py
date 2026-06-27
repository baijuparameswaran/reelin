"""Visual design agent: cinematography palette for every scene.

Processes all scenes in one prompt so the LLM can reason about cross-scene
visual continuity — a color that drains as hope fades, a recurring prop that
anchors the story, a filter shift that marks a character's transformation.

Each scene gets:
  color_palette    — dominant hues and their emotional weight
  visual_filter    — lens/grading style (grain, diffusion, saturation level)
  lighting         — quality, direction, and source
  key_props        — props with dramatic or thematic visual function
  visual_moments   — beat-keyed specifics within the scene
  transition_to_next — how the visual language evolves at the scene cut
  emotional_function — what the visual design communicates to the audience

Genre and tone from the structure agent seed the overall visual palette.
"""
from __future__ import annotations

import json

from .. import llm

SYSTEM = (
    "You are a professional film cinematographer and production designer. You "
    "define the visual world of a screenplay — color palettes, lighting, "
    "filters, props — with an eye for emotional resonance, genre convention, "
    "and cross-scene continuity. You always respond with valid JSON and nothing else."
)

PROMPT = """\
Design the visual language for every scene in the following screenplay outline.

Film details:
- Logline: {logline}
- Genre: {genre}
- Tone: {tone}
- Themes: {themes}

Process all scenes together so that visual continuity and motif development \
across scenes is intentional and coherent.

Respond with JSON in exactly this shape:
{{
  "visual_palette": "one sentence describing the overall visual world of this film",
  "color_language": "how color is used emotionally and symbolically across the film",
  "scenes": [
    {{
      "scene_number": 1,
      "color_palette": "dominant hues and their emotional weight in this scene",
      "visual_filter": "lens or grading style (e.g. 'desaturated, cool blue cast, \
soft grain')",
      "lighting": "quality, direction, and source (e.g. 'harsh overhead practical, \
deep shadows')",
      "key_props": [
        {{
          "prop": "name or brief description",
          "function": "dramatic or thematic visual role this prop plays"
        }}
      ],
      "visual_moments": [
        {{
          "moment": "brief description of the beat within the scene",
          "visual": "specific visual detail, composition, or color note at that beat"
        }}
      ],
      "transition_to_next": "how the visual language shifts or carries over into \
the next scene",
      "emotional_function": "what the visual design communicates to the audience"
    }}
  ]
}}

Rules:
- key_props should only list props with genuine visual or thematic weight, not \
set dressing
- visual_moments should capture beats where the image itself carries meaning
- visual_filter may be 'none' if the scene calls for flat naturalism
- transition_to_next should be empty string for the final scene
- genre should influence the visual approach (thriller → high contrast, \
deep shadows; drama → naturalistic light; period → desaturated warmth, etc.)

SCENE LIST:
{scenes}
"""


def design_visuals(
    structure: dict,
    scenes: dict,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("visuals")
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
