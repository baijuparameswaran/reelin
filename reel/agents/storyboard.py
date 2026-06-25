"""Storyboard agent: a production-ready board for every scene of the film.

This is the synthesis stage. It fuses all upstream artifacts into a structured
storyboard that a director, DP, and VFX team — or a video generation model —
can work from directly:

  structure      → logline, genre, tone
  scenes         → slugline, summary, narrative purpose, characters per scene
  casting        → locked on-screen look per character (physical_form, wardrobe,
                   defining_feature, mannerism) + reference image
  characters     → voice, mannerisms
  visuals        → color palette, lighting, key props, visual filter per scene
  soundscape     → score cue, ambient bed, sound events per scene
  cinematography → shot list with full camera grammar per scene
  screenplay     → attributed dialogue, V.O., written action per shot

Output schema mirrors a real production storyboard:
  scene header   → slugline (int/ext · location · time), purpose, characters,
                   duration estimate
  visual/audio   → color palette, lighting setup, score cue, ambient, key sounds
  panels         → one per camera shot, with shot_type / camera_angle /
                   camera_movement / lens / composition / duration / action /
                   dialogue / sound / emotional_note / transition / image_prompt
"""
from __future__ import annotations

import json
import re

from .. import llm

SYSTEM = (
    "You are a professional storyboard supervisor. You translate a complete scene "
    "design package into a production-ready storyboard that a director, DP, and "
    "video generation pipeline can execute from. Every panel carries the full camera "
    "grammar, locked character look, action, dialogue, and audio — self-sufficient "
    "for rendering. You respond with valid JSON and nothing else."
)

PROMPT = """\
Compose a production storyboard for the film below. Use ALL upstream artifacts \
fused into the scene design bundles — lose nothing.

Film:
- Logline: {logline}
- Genre: {genre}
- Tone: {tone}

STORYBOARD STRUCTURE:

For each scene produce:

1. header — slugline, int_ext (INT/EXT/INT·EXT), location, time_of_day, one-sentence \
narrative purpose, characters present, estimated screen duration (e.g. "1m 45s")

2. visual_overview — color_palette for this scene, lighting_setup (rig or natural \
light description), mood (one line)

3. audio_overview — score_cue (music/score description), ambient (ambient bed), \
key_sounds (list of "moment: sound" strings)

4. panels — ONE panel per camera shot (follow the cinematography coverage in order; \
never merge or drop shots; align each panel with the matching screenplay shot). \
Each panel must include:
   - panel: sequential panel number
   - shot_type: ECU / CU / MCU / MS / FS / WS / ELS / POV / OTS / 2S / INSERT
   - camera_angle: EYE LEVEL / LOW ANGLE / HIGH ANGLE / DUTCH TILT / BIRD'S EYE / WORM'S EYE
   - camera_movement: STATIC / PAN / TILT / DOLLY IN / DOLLY OUT / TRACK LEFT / TRACK RIGHT \
/ CRANE UP / CRANE DOWN / HANDHELD / STEADICAM / ZOOM IN / ZOOM OUT / PUSH IN / PULL OUT
   - lens: focal length e.g. "24mm wide" "50mm normal" "85mm portrait" "135mm telephoto"
   - composition: framing note — who/what is where, depth layers, negative space, leading lines
   - duration: estimated screen time e.g. "3s" "6s" "12s"
   - characters_in_frame: list of character names visible
   - action: physical movement or event (what happens / moves — drives video motion)
   - dialogue: list of {{speaker, line, vo}} — attributed lines from the screenplay \
(vo: true for voice-over); empty list if silent
   - sound: ambient bed + specific sound events audible in this panel
   - emotional_note: the emotion this panel must evoke in the audience
   - transition: CUT TO / DISSOLVE TO / FADE TO BLACK / MATCH CUT / SMASH CUT / \
L-CUT / J-CUT / WIPE
   - image_prompt: a COMPLETE, self-contained prompt for a video generation model — \
include character locked look (physical_form, wardrobe, defining_feature), setting \
and key props, color palette and lighting, camera (shot_type / angle / movement / lens), \
the action, audio atmosphere, and any spoken dialogue — everything the model needs \
to render this panel without any other context

JSON schema (respond with this shape and nothing else):
{{
  "storyboard_style": "one sentence: overall visual language of the boards",
  "storyboard": [
    {{
      "scene_number": 1,
      "header": {{
        "slugline": "INT. LIGHTHOUSE LANTERN ROOM - DUSK",
        "int_ext": "INT",
        "location": "Lighthouse lantern room",
        "time_of_day": "DUSK",
        "purpose": "Edith confronts the choice that will define her",
        "characters": ["EDITH"],
        "duration_estimate": "2m 10s"
      }},
      "visual_overview": {{
        "color_palette": "amber and deep navy; warm lantern glow against cold sea dark",
        "lighting_setup": "practical lantern as key light; blue-grey ambient from windows",
        "mood": "claustrophobic intimacy breaking open into vast dread"
      }},
      "audio_overview": {{
        "score_cue": "solo cello, sustained low drone building through the scene",
        "ambient": "wind against glass, faint ocean below",
        "key_sounds": ["panel 2: the lantern mechanism clicks and stalls",
                       "panel 4: silence as she makes her decision"]
      }},
      "panels": [
        {{
          "panel": 1,
          "shot_type": "WS",
          "camera_angle": "LOW ANGLE",
          "camera_movement": "STATIC",
          "lens": "24mm wide",
          "composition": "Edith small in frame, lantern room towering above, \
ocean visible through curved glass behind her",
          "duration": "5s",
          "characters_in_frame": ["EDITH"],
          "action": "Edith enters the lantern room, stops. Looks up at the mechanism.",
          "dialogue": [],
          "sound": "wind against glass, door creaks shut behind her",
          "emotional_note": "awe mixed with foreboding",
          "transition": "CUT TO",
          "image_prompt": "Wide shot, low angle, static camera, 24mm lens. EDITH \
(50s, weathered face, grey-streaked hair pinned tight, navy-wool keeper's uniform, \
brass-button coat) stands small in the centre of a Victorian lighthouse lantern room. \
Amber lantern glow as key light, cold blue-grey ocean light from curved glass panels. \
She looks up at the Fresnel lens mechanism above. Wind-sound against glass. \
Colour palette: deep navy and warm amber. Cinematic, photorealistic."
        }}
      ]
    }}
  ]
}}

Rules:
- One panel per camera shot — every shot in the coverage, in order (never merge or drop)
- Align each panel with its screenplay shot; bake in the attributed dialogue
- image_prompt is self-contained and render-ready — the character look, setting, \
camera grammar, motion, and audio must ALL be in the prompt
- emotional_note and transition are required on every panel
- Keep character names consistent with the cast

SCENE DESIGN BUNDLES:
{bundles}
"""


def _parse_slugline(slugline: str) -> dict:
    """Extract int_ext, location, and time_of_day from a Fountain slugline."""
    s = slugline.strip().upper()
    time_of_day = ""
    location = s
    if " - " in s:
        parts = s.rsplit(" - ", 1)
        location, time_of_day = parts[0].strip(), parts[1].strip()
    int_ext = ""
    for prefix in ("INT./EXT.", "EXT./INT.", "INT/EXT.", "INT.", "EXT."):
        if location.startswith(prefix):
            int_ext = prefix.rstrip(".")
            location = location[len(prefix):].strip()
            break
    return {"int_ext": int_ext, "location": location, "time_of_day": time_of_day}


def _scene_bundles(
    scenes: dict,
    casting: dict,
    soundscape: dict,
    visuals: dict,
    cinematography: dict,
    characters: dict | None = None,
    draft: dict | None = None,
) -> list[dict]:
    """Merge per-scene designs into rich bundles for the prompt.

    Captures the FULL detail of every upstream artifact — locked cast look
    (casting) plus voice/mannerisms (characters), complete art design, complete
    soundscape, complete camera coverage, and the screenplay's shots + attributed
    dialogue — because the storyboard drives video generation and must lose nothing.
    """
    cast_by_name = {c.get("name", ""): c for c in casting.get("casting", [])}
    char_by_name = {c.get("name", ""): c for c in (characters or {}).get("characters", [])}
    sound_by_scene = {s.get("scene_number"): s for s in soundscape.get("soundscapes", [])}
    vis_by_scene = {s.get("scene_number"): s for s in visuals.get("scenes", [])}
    cin_by_scene = {s.get("scene_number"): s for s in cinematography.get("scenes", [])}
    draft_by_scene = {s.get("number", s.get("scene_number")): s
                      for s in (draft or {}).get("scenes", [])}

    bundles = []
    for scene in scenes.get("scenes", []):
        num = scene.get("number")
        slugline = scene.get("slugline", "")
        slugline_parsed = _parse_slugline(slugline)
        char_names = scene.get("characters", []) or list(cast_by_name)

        cast = []
        for name in char_names:
            c = cast_by_name.get(name, {})
            ch = c.get("character", c)          # character block (locked on-screen look)
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
            "slugline": slugline,
            "slugline_parsed": slugline_parsed,
            "summary": scene.get("summary", ""),
            "purpose": scene.get("purpose", ""),
            "characters_in_scene": char_names,
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
                "score_direction": snd.get("score_direction", ""),
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
            "screenplay_shots": [
                {
                    "shot": sh.get("shot", ""),
                    "shot_type": sh.get("shot_type", ""),
                    "description": sh.get("description", ""),
                    "voiceover": sh.get("voiceover") or None,
                    "dialogue": [
                        {"speaker": d.get("speaker", ""), "modifier": d.get("modifier", ""),
                         "parenthetical": d.get("parenthetical", ""), "line": d.get("line", "")}
                        for d in (sh.get("dialogue") or [])
                    ],
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
