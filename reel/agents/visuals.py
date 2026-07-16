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

`key_props` is grounded by scenes.py's own `props` field (a plain,
source-mentioned inventory, rule 10 there) when present in the scene list —
this agent's job is still the creative one (which props actually carry
dramatic/thematic weight, and why), but it no longer has to invent candidate
objects from nothing. `key_props` reaches the actual Veo render prompt via
`storyboard.py`'s `visual_overview.key_props` and `pipeline._panel_context`
(see those modules) — see the 2026-07-10 PROGRESS.md session log entry for
why that wiring was needed (key_props previously never left visuals.json).
"""
from __future__ import annotations

import json

from .. import llm
from ..revision_merge import merge_by_key

SYSTEM = (
    "You are an award-winning film cinematographer and production designer, "
    "among the best working today. You define the visual world of a "
    "screenplay — color palettes, lighting, filters, props — with an eye for "
    "emotional resonance, genre convention, and cross-scene continuity. You "
    "always respond with valid JSON and nothing else."
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

DO NOT:
- give two scenes sharing a `location` different base `color_palette`/`lighting`
- promote plain set dressing into `key_props` — only genuine dramatic/thematic weight
- invent a prop not present in the scene's own `props` field (when given)
- skip, renumber, or duplicate a `scene_number` relative to the SCENE LIST
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present for every scene, no extra top-level keys, no missing keys:
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
set dressing. A scene's `props` field (if present below) is a plain, \
source-grounded inventory of objects actually mentioned in the story — a \
starting point, not the answer: promote the ones with real dramatic or \
thematic weight into `key_props` with a `function`; leave out mere set \
dressing from that list even if it's in `props`
- visual_moments should capture beats where the image itself carries meaning
- visual_filter may be 'none' if the scene calls for flat naturalism
- transition_to_next should be empty string for the final scene
- Scenes sharing the same `location` share the same base color_palette and \
lighting (that place has a fixed, rendered look) — only the mood/visual_filter \
should shift between them (e.g. day vs. night at the same place), not the \
underlying architecture or light sources
- genre should influence the visual approach (thriller → high contrast, \
deep shadows; drama → naturalistic light; period → desaturated warmth, etc.)

SCENE LIST:
{scenes}

Before you respond, re-check against the scene list above (long scene lists \
push early rules out of recent context — re-verify against what you just \
read, not just what you remember from the rules list):
- Every scene sharing a `location` with another keeps that place's base \
`color_palette` and `lighting` identical — only mood/`visual_filter` shifts.
- `scene_number` in your output matches the `number` field from the scene list \
above exactly, one output scene per input scene, none skipped or renumbered.
- The response is ONLY the JSON object above — no markdown fences, no \
commentary, no extra top-level keys, every scene has all its schema fields.
"""


def design_visuals(
    structure: dict,
    scenes: dict,
    profile: str | None = None,
    feedback: str | None = None,
    existing: dict | None = None,
    revise_keys: set | None = None,
) -> dict:
    """`existing` + `revise_keys` (a set of `scene_number`s) support a scoped
    revision — see `reel.agents.soundscape.design_soundscape`'s docstring for
    the shared rationale. This artifact's list field is `"scenes"`, keyed by
    `scene_number`."""
    profile = profile or llm.agent_profile("visuals")
    scene_list = json.dumps(
        [
            {k: s[k] for k in ("number", "slugline", "location", "summary", "purpose",
                                "source_line", "chunk_indices", "props")
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
        result["scenes"] = merge_by_key(
            existing.get("scenes", []), result.get("scenes", []),
            lambda s: s.get("scene_number"), revise_keys,
        )
    return result
