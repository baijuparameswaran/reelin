"""Storyboard agent: a visual image for every moment of every scene.

This is the synthesis stage. It fuses the four creative-crew designs into a
concrete frame for each beat:

  casting        → who is in frame and exactly how they look
  visuals        → color palette, lighting, props (art production)
  cinematography → shot type, angle, movement, lens (camera)
  soundscape     → the score / audio attribute under the frame

The agent receives a pre-merged per-scene design bundle (built below) so it can
compose each frame as a single image with both an emotional and an audio
attribute, plus a text-to-image prompt ready for a render pipeline.
"""
from __future__ import annotations

import json

from .. import llm

SYSTEM = (
    "You are a storyboard artist and visual director. You translate a scene's "
    "casting, art design, camera plan, and score into concrete frames — each a "
    "single composed image carrying an emotional charge and an audio attribute. "
    "You always respond with valid JSON and nothing else."
)

PROMPT = """\
Compose a storyboard for the film below. For each scene, break it into frames \
(one per key moment, following the camera shots where given) and render each \
frame as a single concrete image.

Film details:
- Logline: {logline}
- Genre: {genre}
- Tone: {tone}

Each scene's design bundle gives you the cast (and their locked physical form), \
the art design (color, lighting, props), the camera plan (shot, angle, lens, \
movement), and the audio bed. Fuse them.

Respond with JSON in exactly this shape:
{{
  "storyboard_style": "one sentence describing the overall look of the boards",
  "storyboard": [
    {{
      "scene_number": 1,
      "frames": [
        {{
          "frame": 1,
          "moment": "the beat this frame captures",
          "characters_in_frame": ["NAME", "..."],
          "image": "a single composed visual: who/what is in frame, their look, \
the set and props, color and light, and the camera framing (shot/angle/lens)",
          "emotional_attribute": "the emotion this frame should evoke",
          "audio_attribute": "the score / sound under this frame",
          "image_prompt": "a concise text-to-image prompt to render this frame"
        }}
      ]
    }}
  ]
}}

Rules:
- Derive frames from the camera shots when present; otherwise pick the key beats
- Every image must integrate cast appearance + art design + camera framing
- emotional_attribute and audio_attribute are required on every frame
- Keep names consistent with the cast

SCENE DESIGN BUNDLES:
{bundles}
"""


def _scene_bundles(
    scenes: dict,
    casting: dict,
    soundscape: dict,
    visuals: dict,
    cinematography: dict,
) -> list[dict]:
    """Merge the per-scene designs into compact bundles for the prompt."""
    cast_by_name = {c.get("name", ""): c for c in casting.get("casting", [])}
    sound_by_scene = {s.get("scene_number"): s for s in soundscape.get("soundscapes", [])}
    vis_by_scene = {s.get("scene_number"): s for s in visuals.get("scenes", [])}
    cin_by_scene = {s.get("scene_number"): s for s in cinematography.get("scenes", [])}

    bundles = []
    for scene in scenes.get("scenes", []):
        num = scene.get("number")
        names = scene.get("characters", []) or list(cast_by_name)
        cast = []
        for name in names:
            c = cast_by_name.get(name, {})
            # The on-screen look lives in the character (transformation) block;
            # fall back to the flat legacy schema for older casting.json.
            ch = c.get("character", c)
            cast.append({
                "name": name,
                "kind": c.get("kind", ""),
                "physical_form": ch.get("physical_form", ""),
                "wardrobe": ch.get("costume", ch.get("wardrobe", "")),
                "defining_feature": ch.get("defining_feature", ""),
            })

        vis = vis_by_scene.get(num, {})
        snd = sound_by_scene.get(num, {})
        cin = cin_by_scene.get(num, {})

        bundles.append({
            "scene_number": num,
            "slugline": scene.get("slugline", ""),
            "summary": scene.get("summary", ""),
            "cast": cast,
            "art": {
                "color_palette": vis.get("color_palette", ""),
                "lighting": vis.get("lighting", ""),
                "key_props": [p.get("prop", "") for p in vis.get("key_props", [])],
            },
            "audio": {
                "ambient_bed": snd.get("ambient_bed", ""),
                "silence": snd.get("silence", False),
            },
            "camera": [
                {k: shot.get(k, "") for k in ("moment", "type", "angle", "movement", "lens")}
                for shot in cin.get("shots", [])
            ],
        })
    return bundles


def plan_storyboard(
    structure: dict,
    scenes: dict,
    casting: dict,
    soundscape: dict,
    visuals: dict,
    cinematography: dict,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("storyboard")
    bundles = _scene_bundles(scenes, casting, soundscape, visuals, cinematography)
    prompt = llm.with_feedback(
        PROMPT.format(
            logline=structure.get("logline", ""),
            genre=structure.get("genre", "drama"),
            tone=structure.get("tone", ""),
            bundles=json.dumps(bundles, ensure_ascii=False, indent=2),
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
