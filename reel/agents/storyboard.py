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

Each scene's design bundle gives you EVERYTHING about the scene and you must fuse \
ALL of it into each frame, losing no detail:
- cast: each character's locked physical_form, age, wardrobe, defining_feature, \
mannerism, voice, and a reference_image of their on-screen identity
- art: color_palette, lighting, visual_filter, key_props (prop+function), \
visual_moments, emotional_function
- audio: ambient_bed (or silence), sound_events (moment+sound), emotional_function
- camera: coverage + per-shot type/angle/movement/lens/framing/emotional_function \
and the transition to the next scene
- screenplay_shots: the script's own written shots, action, and attributed \
dialogue / voice-over for this scene

These frames DRIVE VIDEO GENERATION, so each `image_prompt` must be self-contained.

Respond with JSON in exactly this shape:
{{
  "storyboard_style": "one sentence describing the overall look of the boards",
  "storyboard": [
    {{
      "scene_number": 1,
      "frames": [
        {{
          "frame": 1,
          "shot_type": "the camera shot type for this frame (from the coverage)",
          "moment": "the beat this frame captures",
          "characters_in_frame": ["NAME", "..."],
          "image": "a single composed visual: who is in frame and their exact \
locked look (physical_form, wardrobe, defining_feature), the set and key props, \
color/light/filter, and the camera framing (shot/angle/lens/movement)",
          "action": "what physically happens/moves in this frame (for motion)",
          "dialogue": [{{"speaker": "NAME", "line": "spoken line heard in this frame"}}],
          "emotional_attribute": "the emotion this frame should evoke",
          "audio_attribute": "the ambient bed + sound events + any score under this frame",
          "image_prompt": "a complete, self-contained prompt to render this frame as \
VIDEO: subject + locked look, setting + props, color/light/filter, camera \
shot/angle/lens/movement, the motion/action, and the audio + any spoken dialogue \
(so a video model can voice it)"
        }}
      ]
    }}
  ]
}}

Rules:
- Produce ONE frame per camera shot — cover EVERY shot in the coverage, in order
  (do not merge or drop shots); align it with the matching screenplay shot. Only
  when a scene has no camera coverage, use the screenplay_shots (or key beats).
- Each frame MUST fuse cast look + art design + camera framing + audio, and carry
  any dialogue/voice-over from the screenplay for that beat.
- `image_prompt` must be self-sufficient (it is what gets rendered) — bake the
  look, camera, motion, and audio into it.
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
    characters: dict | None = None,
    draft: dict | None = None,
) -> list[dict]:
    """Merge the per-scene designs into rich bundles for the prompt.

    Captures the FULL detail of every upstream artifact — the locked cast look
    (casting) plus voice/mannerisms (characters), the complete art design, the
    complete soundscape, the complete camera coverage, and the screenplay's own
    written shots + attributed dialogue for the scene — because the storyboard is
    what drives video generation and must lose nothing.
    """
    cast_by_name = {c.get("name", ""): c for c in casting.get("casting", [])}
    char_by_name = {c.get("name", ""): c for c in (characters or {}).get("characters", [])}
    sound_by_scene = {s.get("scene_number"): s for s in soundscape.get("soundscapes", [])}
    vis_by_scene = {s.get("scene_number"): s for s in visuals.get("scenes", [])}
    cin_by_scene = {s.get("scene_number"): s for s in cinematography.get("scenes", [])}
    draft_by_scene = {s.get("number", s.get("scene_number")): s for s in (draft or {}).get("scenes", [])}

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
            person = char_by_name.get(name, {})
            cast.append({
                "name": name,
                "kind": c.get("kind", person.get("kind", "")),
                "physical_form": ch.get("physical_form", person.get("appearance", "")),
                "age": ch.get("age", ""),
                "wardrobe": ch.get("costume", ch.get("wardrobe", "")),
                "defining_feature": ch.get("defining_feature", ""),
                "mannerism": ch.get("mannerism", person.get("mannerisms", "")),
                "voice": person.get("voice", ""),
                "visual_prompt": ch.get("visual_prompt", ""),
                "reference_image": ch.get("image_path", ""),
            })

        vis = vis_by_scene.get(num, {})
        snd = sound_by_scene.get(num, {})
        cin = cin_by_scene.get(num, {})
        scr = draft_by_scene.get(num, {})

        bundles.append({
            "scene_number": num,
            "slugline": scene.get("slugline", ""),
            "summary": scene.get("summary", ""),
            "purpose": scene.get("purpose", ""),
            "cast": cast,
            "art": {
                "color_palette": vis.get("color_palette", ""),
                "lighting": vis.get("lighting", ""),
                "visual_filter": vis.get("visual_filter", ""),
                "key_props": [{"prop": p.get("prop", ""), "function": p.get("function", "")}
                              for p in vis.get("key_props", [])],
                "visual_moments": [{"moment": m.get("moment", ""), "visual": m.get("visual", "")}
                                   for m in vis.get("visual_moments", [])],
                "emotional_function": vis.get("emotional_function", ""),
            },
            "audio": {
                "ambient_bed": snd.get("ambient_bed", ""),
                "silence": snd.get("silence", False),
                "sound_events": [{"moment": e.get("moment", ""), "sound": e.get("sound", "")}
                                 for e in snd.get("sound_events", [])],
                "emotional_function": snd.get("emotional_function", ""),
            },
            "camera": {
                "coverage": cin.get("coverage", ""),
                "transition_to_next": cin.get("transition_to_next", ""),
                "shots": [
                    {k: shot.get(k, "") for k in
                     ("shot_number", "moment", "type", "angle", "movement", "lens",
                      "framing", "emotional_function")}
                    for shot in cin.get("shots", [])
                ],
            },
            # The screenplay's own written shots + attributed dialogue for this scene,
            # so storyboard frames align with the script and can carry spoken lines.
            "screenplay_shots": [
                {
                    "shot": sh.get("shot", ""),
                    "shot_type": sh.get("shot_type", ""),
                    "description": sh.get("description", ""),
                    "voiceover": sh.get("voiceover") or None,
                    "dialogue": [{"speaker": d.get("speaker", ""), "modifier": d.get("modifier", ""),
                                  "parenthetical": d.get("parenthetical", ""), "line": d.get("line", "")}
                                 for d in (sh.get("dialogue") or [])],
                    "sound": sh.get("sound", ""),
                }
                for sh in scr.get("shots", [])
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
    characters: dict | None = None,
    draft: dict | None = None,
    genre: dict | str | None = None,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("storyboard")
    bundles = _scene_bundles(scenes, casting, soundscape, visuals, cinematography,
                             characters=characters, draft=draft)
    genre_name = (genre.get("genre") if isinstance(genre, dict) else genre) \
        or structure.get("genre", "drama")
    prompt = llm.with_feedback(
        PROMPT.format(
            logline=structure.get("logline", ""),
            genre=genre_name,
            tone=structure.get("tone", ""),
            bundles=json.dumps(bundles, ensure_ascii=False, indent=2),
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    return llm.safe_json(raw)
