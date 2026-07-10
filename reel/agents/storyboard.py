"""Storyboard agent: a production-ready board for every scene of the film.

This is the fusion stage. It merges all upstream artifacts into a structured
storyboard that a director, DP, and VFX team — or a video generation model —
can work from directly:

  structure      → logline, genre, tone
  scenes         → slugline, summary, narrative purpose, characters + location per scene
  casting        → locked on-screen look per character (physical_form, wardrobe,
                   defining_feature, mannerism) + reference image; also locked
                   locations (kind: "location", no actor layer) + reference image,
                   keyed by scenes' `location` field
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

BUILD PATH — deterministic by default, LLM only when there's an actual creative
judgment call to make. Every field above already has an authoritative source
in an upstream artifact (that's the whole point of a FUSION stage — it isn't
supposed to make new creative decisions, just faithfully combine ones already
made). `_build_scene_board` constructs the entire scene deterministically —
`_scene_bundles` (pure Python, no LLM) already merges everything scene-by-scene
keyed off `scene_number`; `_align_shots` pairs cinematography's shot list
(authoritative for panel count/order — the "coverage plan") with screenplay's
shot list (should already track it 1:1, since screenplay.py's own prompt
instructs it to derive shots from the camera coverage; falls back to
positional pairing when they don't). A deterministic merge can't drift,
hallucinate, or drop a field the way an LLM sometimes did in practice (see
PROGRESS.md's session log — dropped voiceover lines, unused visual_filter/
sound_events/key_props, invented transitions). The ONE thing genuinely
invented rather than sourced is `characters_in_frame` for close-shot panels,
which is a heuristic (dialogue speaker, when the shot type is a close-up
family) since no artifact actually specifies who's visually in a given shot.

`feedback` is the one path that still calls the LLM (`_llm_generate_scene`,
the prior full generation prompt) — a directed creative note ("add more
close-ups", "make it darker") needs actual judgment to apply, which the
deterministic builder can't do. Everything else — a fresh run, a scoped
`revise_keys` revision with no `feedback` — is fully deterministic.

NOTE on image_prompt: even where it's still generated (the deterministic
fallback, or the LLM feedback path), it's not what's actually submitted to
Veo for real renders — pipeline._render_scene_frames reconstructs the prompt
from structured fields (this panel's own shot_type/camera_angle/camera_movement/
lens, casting.json, the scene's visual_overview) into the five-PART formula —
Cinematography+Subject+Action+Context+Style&Ambiance, per Google's Veo 3.1
prompting guide (see pipeline._five_part_veo_prompt) — whenever casting context
is available. image_prompt here is used verbatim only as a fallback when that
context is absent (e.g. a standalone gen-video prompt with no casting/scene data).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import llm
from ..revision_merge import merge_by_key
from .ingest import scene_source_context

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
{story_block}
STORYBOARD STRUCTURE:

For each scene produce:

1. header — slugline, int_ext (INT/EXT/INT·EXT), location, time_of_day, one-sentence \
narrative purpose, characters present, estimated screen duration (e.g. "1m 45s")

2. visual_overview — color_palette for this scene, lighting_setup (rig or natural \
light description), visual_filter (lens/grading style, e.g. "desaturated, cool blue \
cast, soft grain"), mood (one line), key_props (list of prop NAMES only, copied from \
bundle.art.key_props — this is what actually reaches the render prompt, see MERGE \
SOURCE below)

3. audio_overview — score_cue (music/score description), ambient (ambient bed), \
key_sounds (list of "moment: sound" strings)

4. panels — ONE panel per camera shot (follow the cinematography coverage in order; \
never merge or drop shots; align each panel with the matching screenplay shot). \
Each panel must include:
   - panel: sequential panel number
   - shot_type: ECU / CU / MCU / MS / FS / WS / ELS / POV / OTS / 2S / INSERT
   - camera_angle: EYE LEVEL / LOW ANGLE / HIGH ANGLE / DUTCH TILT / BIRD'S EYE VIEW / WORMS EYE / AERIAL VIEW / TOP-DOWN
   - camera_movement: STATIC / PAN / TILT / DOLLY IN / DOLLY OUT / TRACKING / \
CRANE UP / CRANE DOWN / HANDHELD / STEADICAM / AERIAL VIEW / ZOOM IN / ZOOM OUT
   - lens: focal length e.g. "24mm wide-angle lens" "50mm normal lens" "85mm portrait lens" "135mm telephoto lens"
   - composition: framing note — who/what is where, depth layers, negative space, leading lines
   - duration: estimated screen time e.g. "3s" "6s" "12s"
   - characters_in_frame: list of character names visible
   - action: physical movement or event (what happens / moves — drives video motion)
   - dialogue: list of {{speaker, line, vo}} — copy VERBATIM from the matching \
screenplay_shot.dialogue; do not paraphrase, trim, or invent lines; \
(vo: true for voice-over); empty list if silent
   - sound: TWO PARTS — (a) ambient bed first: the environment's soundscape described \
naturally ("wind howling outside, distant ocean waves"); (b) then SFX: explicitly \
described action sounds ("a door slams, glass shatters") — separate the two with " | " \
so the render layer can apply Veo's ambient-noise vs SFX distinction correctly
   - emotional_note: the emotion this panel must evoke in the audience
   - transition: CUT TO / DISSOLVE TO / FADE TO BLACK / MATCH CUT / SMASH CUT / \
L-CUT / J-CUT / WIPE
   - image_prompt: a COMPLETE, self-contained Veo video generation prompt following \
the five-element guide order — (1) Subject: character(s) with full locked look \
(physical_form, wardrobe, defining_feature); (2) Action: exactly what moves or \
happens in the shot; (3) Style: one or more Veo style keywords — "cinematic", \
"photorealistic", "film noir", "documentary", "cinéma vérité", etc.; \
(4) Camera & Composition: shot type in natural language ("wide shot", "close-up", \
"extreme close-up", "POV shot", "over-the-shoulder shot", "two-shot", "medium shot"), \
camera angle using Veo vocabulary ("eye-level", "low angle", "high angle", \
"bird's eye view", "worms eye", "aerial view", "top-down shot", "Dutch tilt"), \
movement ("static", "dolly in", "dolly out", "tracking", "handheld", "aerial view", \
"panning", "crane up", "zoom in"), \
and lens (e.g. "24mm wide-angle lens", "85mm portrait lens", "135mm telephoto lens"); \
(5) Focus & Ambiance: focus term + color and lighting mood — MUST MATCH THIS \
PANEL'S OWN shot_type, not a default: "portrait, shallow focus" for CU/ECU shots \
(Veo guide: enhances facial detail), "deep focus" for WS/ELS (environmental \
clarity), "macro lens" for INSERT shots. Every panel in a scene re-decides this \
independently from its own shot_type — do not copy the focus term from a \
different panel's example or from the previous panel; a close-up next to a wide \
shot should NOT share the same focus term. Describe lighting and color as mood \
("warm golden hour glow", "cold blue-grey shadows", "neon eerie glow", "harsh \
midday sun", "soft diffused overcast light"). Do NOT include dialogue or sound effects in image_prompt — \
those live in the panel's dialogue and sound fields and are added to the Veo prompt \
separately

MERGE SOURCE — where each output field's ground truth already lives in the scene
bundle below (an earlier stage already decided this; ground your output in it
rather than reinventing it):
- visual_overview.color_palette / lighting_setup / mood  <- bundle.art.color_palette / \
.lighting / .emotional_function
- visual_overview.visual_filter                          <- bundle.art.visual_filter, \
used close to verbatim
- audio_overview.ambient / score_cue                      <- bundle.audio.ambient_bed / \
.score_direction — UNLESS bundle.audio.silence is true, in which case leave both \
empty/minimal: silence there is a deliberate choice, not a gap to fill
- audio_overview.key_sounds                               <- one "panel N: <sound>" \
entry per item in bundle.audio.sound_events, not invented from scratch
- panel.action                                            <- grounded in the matching \
bundle.screenplay_shots.description (the SAME event the screenplay already wrote — \
add cinematic specificity, but do not describe a different event)
- panel.dialogue                                          <- copied verbatim from the \
matching bundle.screenplay_shots.dialogue, PLUS a {{"speaker", "line", "vo": true}} \
entry for bundle.screenplay_shots.voiceover whenever a shot has one — voiceover is a \
SEPARATE field from dialogue; never drop it or substitute your own narration
- panel.composition / image_prompt                        <- incorporate \
bundle.art.visual_moments (beat-keyed visual specifics) and bundle.art.key_props \
where they apply to this panel's beat, rather than inventing unrelated detail
- visual_overview.key_props                               <- prop NAMES only \
(drop the .function commentary) from bundle.art.key_props, copied as a plain list — \
this is what actually reaches the Veo render prompt for every panel in the scene, \
not just the panel where the prop is most relevant, so list every key prop that \
belongs anywhere in this scene
- the LAST panel's transition                             <- bundle.camera.transition_to_next \
when present, used as-is rather than an invented scene-ending transition

JSON schema (respond with this shape and nothing else):
{{
  "storyboard_style": "one sentence: overall visual language of the boards",
  "storyboard": [
    {{
      "scene_number": 1,
      "header": {{
        "slugline": "EXT. LOCATION NAME - TIME OF DAY",
        "int_ext": "EXT",
        "location": "Location name from the scene bundle's `location.name` field \
if present (use it verbatim — do not rephrase), otherwise from the slugline",
        "time_of_day": "DAY",
        "purpose": "One sentence narrative purpose of this scene",
        "characters": ["CHARACTER_A"],
        "duration_estimate": "1m 30s"
      }},
      "visual_overview": {{
        "color_palette": "describe the dominant colors and contrast for this scene",
        "lighting_setup": "describe the light source(s) and quality",
        "visual_filter": "lens/grading style from the scene bundle's art.visual_filter",
        "mood": "one line capturing the emotional atmosphere",
        "key_props": ["prop name from bundle.art.key_props, copied verbatim", "..."]
      }},
      "audio_overview": {{
        "score_cue": "describe the music/score for this scene",
        "ambient": "describe the environmental soundscape",
        "key_sounds": ["panel N: describe the key sound event",
                       "panel N: describe another key sound event"]
      }},
      "panels": [
        {{
          "panel": 1,
          "shot_type": "WS",
          "camera_angle": "EYE LEVEL",
          "camera_movement": "STATIC",
          "lens": "24mm wide",
          "composition": "describe who/what is where in the frame, depth layers, leading lines",
          "duration": "5s",
          "characters_in_frame": ["CHARACTER_A"],
          "action": "Describe exactly what moves or happens in this shot.",
          "dialogue": [],
          "sound": "describe ambient bed and any specific sound events audible here",
          "emotional_note": "the emotion this panel must evoke in the audience",
          "transition": "CUT TO",
          "image_prompt": "CHARACTER_A (physical description: age, build, hair, wardrobe, \
defining feature) performs the action in the location. Cinematic, photorealistic. \
Wide shot, eye-level, static camera, 24mm wide-angle lens. Deep focus. \
Describe lighting and color palette as mood."
        }},
        {{
          "panel": 2,
          "shot_type": "CU",
          "camera_angle": "EYE LEVEL",
          "camera_movement": "STATIC",
          "lens": "50mm normal",
          "composition": "describe framing tighter on the subject's face/reaction",
          "duration": "4s",
          "characters_in_frame": ["CHARACTER_A"],
          "action": "Describe the closer, more intimate beat this shot covers.",
          "dialogue": [],
          "sound": "describe ambient bed and any specific sound events audible here",
          "emotional_note": "the emotion this panel must evoke in the audience",
          "transition": "CUT TO",
          "image_prompt": "CHARACTER_A (physical description: age, build, hair, wardrobe, \
defining feature) performs the action in the location. Cinematic, photorealistic. \
Close-up, eye-level, static camera, 50mm normal lens. Portrait, shallow focus. \
Describe lighting and color palette as mood."
        }}
      ]
    }}
  ]
}}

Rules:
- FIDELITY FIRST: every panel's action and events must be grounded in the source \
material above. Do not invent scenes, add character motivations, or introduce \
relationships not present in the story.
- Dialogue is LOCKED: copy each line verbatim from `screenplay_shots.dialogue` \
into the matching panel's `dialogue` list. Never paraphrase, merge, or add lines. \
`screenplay_shots.voiceover` is a SEPARATE field, not part of `.dialogue` — when a \
shot has one, copy it in too as its own `dialogue` entry with `vo: true`; do not \
drop it just because it isn't in the `.dialogue` list.
- One panel per camera shot — every shot in the coverage, in order (never merge or drop)
- Align each panel with its screenplay shot; bake in the verbatim attributed dialogue
- image_prompt is self-contained and render-ready — the character look, setting, \
camera grammar, motion, and audio must ALL be in the prompt
- CHARACTER LOOK IN image_prompt: build it from `cast[].physical_form`, `wardrobe`, \
`defining_feature`, and `mannerism` — the STRUCTURED fields. Do NOT copy \
`cast[].visual_prompt` into a panel's image_prompt: that field is a portrait \
prompt for an isolated identity-reference render (it ends with its OWN fixed \
backdrop clause — "plain seamless studio backdrop, solid neutral grey, no scene, \
no props, no location" — written for a blank-background photo, not this scene). \
Echoing any part of that clause into a panel's image_prompt would tell Veo to \
render the character against a blank studio backdrop instead of in the actual \
scene, contradicting every other part of the prompt.
- If the scene bundle includes a `moodboard_tile`, use its `visual_reference` text \
as a tonal and composition anchor when writing `image_prompt`s for that scene's panels
- If the scene bundle includes a `location` block, its `visual_prompt` IS meant to be \
used directly (unlike `cast[].visual_prompt` above) — it describes the actual \
physical space with no isolation clause to strip, and is the LOCKED render \
reference for this scene's setting (an actual rendered image backs it). Describe \
the environment in every panel's `image_prompt` consistently with it. Do not \
contradict it (a different layout, different fixed decor) and do not re-describe \
it fully in every panel — establish it in the first panel's composition, then \
reference it briefly in later panels of the same scene.
- SELF-CONTAINED VS. BRIEF (not a conflict — different scope): "self-contained" \
above means every panel's image_prompt independently carries the CHARACTER LOOK, \
CAMERA GRAMMAR, and ACTION needed to render that panel alone, with nothing \
implied from a neighboring panel. It does NOT mean re-describing the LOCATION's \
full architecture/decor in every panel — that one element follows the brief-\
after-first-panel rule just above. A later panel's image_prompt is still fully \
self-contained for rendering purposes even with just a short location anchor \
phrase (e.g. "in the same bar" rather than restating every fixture).
- emotional_note and transition are required on every panel
- Keep character names consistent with the cast
- Avoid unnecessary repetition across panels: each panel's `image_prompt` and
  `emotional_note` must read as visually/dramatically distinct from every other
  panel in this scene — vary the framing, focal detail, or moment described —
  never restate the same composition or description twice just because the
  shots cover the same characters/setting

SCENE DESIGN BUNDLES:
{bundles}

Before you respond, re-check against the scene design bundle above (long \
bundles push early rules out of recent context — re-verify against what you \
just read, not just what you remember from the rules list):
- Every `dialogue` line is copied verbatim from `screenplay_shots.dialogue`, \
plus a separate `vo: true` entry for any `screenplay_shots.voiceover` — neither \
paraphrased nor dropped.
- One panel per camera shot in `camera.shots`, in the same order, none merged \
or dropped.
- Each panel's `image_prompt` re-decides Focus & Ambiance from THAT panel's own \
shot_type (see SELF-CONTAINED VS. BRIEF above) — not copied from a neighboring \
panel.
- `scene_number` in your output matches the bundle's own `scene_number` exactly.
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
    moodboard: dict | None = None,
    source: dict | None = None,
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
    # Moodboard tiles are text-only visual cues (one per scene, in order).
    # They travel as text through the storyboard so image generation is not required
    # at this stage; the storyboard uses the label + image_prompt as a reference brief.
    mood_tiles = list((moodboard or {}).get("tiles") or [])

    # Locations are cast alongside characters (kind: "location") but have no
    # actor layer and must never be treated as a character — excluded from the
    # "no characters listed" fallback below so a location doesn't get described
    # with person-shaped fields (wardrobe, mannerism, ...) it doesn't have.
    person_names = [n for n, c in cast_by_name.items() if c.get("kind") != "location"]

    bundles = []
    for i, scene in enumerate(scenes.get("scenes", [])):
        num = scene.get("number")
        slugline = scene.get("slugline", "")
        slugline_parsed = _parse_slugline(slugline)
        char_names = scene.get("characters", []) or person_names

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

        # Locked location reference (kind: "location" casting entry, keyed by
        # scenes.py's `location` field — same name across every scene set there).
        loc_name = (scene.get("location") or "").strip()
        loc_entry = cast_by_name.get(loc_name, {}) if loc_name else {}
        loc_ch = loc_entry.get("character", loc_entry)
        location = ({
            "name": loc_name,
            "visual_prompt": loc_ch.get("visual_prompt", ""),
            "reference_image": loc_ch.get("image_path", ""),
        } if loc_name else None)

        vis = vis_by_scene.get(num, {})
        snd = sound_by_scene.get(num, {})
        cin = cin_by_scene.get(num, {})
        scr = draft_by_scene.get(num, {})
        mood_tile = mood_tiles[i] if i < len(mood_tiles) else {}

        bundles.append({
            "scene_number": num,
            "slugline": slugline,
            "slugline_parsed": slugline_parsed,
            "summary": scene.get("summary", ""),
            "purpose": scene.get("purpose", ""),
            "characters_in_scene": char_names,
            "cast": cast,
            "location": location,
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
            # Moodboard tile for this scene: text-only visual reference brief.
            "moodboard_tile": {
                "label": mood_tile.get("label", ""),
                "visual_reference": mood_tile.get("image_prompt", ""),
            } if mood_tile else None,
            # Per-scene source context: prefers the scene's own precise
            # source_excerpt (see scenes.py's _attach_source_excerpts), else the
            # chunk(s) of the story that this scene corresponds to. Used in the
            # per-scene storyboard LLM call so the model sees the specific
            # passage rather than a global truncated head.
            "_source_context": scene_source_context(
                source or {}, scene.get("chunk_indices"),
                source_excerpt=scene.get("source_excerpt")) if source else "",
        })
    return bundles


def _collect_casting_images(casting: dict, out: Path | None) -> list[Path]:
    """Gather unique casting reference images from output/casting/*.png.

    Returns paths in character order. Missing files are silently skipped so the
    call degrades gracefully when images haven't been rendered yet.
    """
    if out is None:
        return []
    imgs: list[Path] = []
    seen: set[str] = set()
    for c in casting.get("casting", []):
        ch = c.get("character", c)
        path_str = ch.get("image_path", "")
        if path_str and path_str not in seen:
            seen.add(path_str)
            p = Path(path_str) if Path(path_str).is_absolute() else out / path_str
            if p.exists():
                imgs.append(p)
    return imgs


def _story_block(context_text: str) -> str:
    """Source context for one scene — fidelity anchor in the storyboard prompt."""
    text = (context_text or "").strip()
    if not text:
        return ""
    return (
        "\nSOURCE MATERIAL for this scene (primary fidelity anchor — every panel "
        "must reflect what actually happens here; do not invent events or embellish "
        "beyond what the source supports):\n" + text + "\n"
    )


# ── deterministic scene-board construction (no LLM) ───────────────────────────
# See the module docstring's "BUILD PATH" section for why this is the default.

_CLOSE_SHOT_TYPES = {"CU", "ECU", "MCU", "OTS", "POV"}

_SHOT_BASE_SECONDS = {
    "ECU": 2, "CU": 3, "MCU": 3, "MS": 4, "FS": 5, "WS": 5,
    "ELS": 6, "POV": 4, "OTS": 4, "2S": 4, "INSERT": 2,
}


def _align_shots(cam_shots: list[dict], scr_shots: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each cinematography shot (authoritative for panel count/order —
    it's the coverage plan every panel is required to follow) with its
    corresponding screenplay shot. Primary: match by shot number — screenplay.py's
    own prompt already instructs it to derive its shots from the camera
    coverage, so the two lists should carry the same numbering in the common
    case. Falls back to positional pairing when a cinematography shot has no
    `shot_number` or the numbering doesn't line up (the two lists genuinely
    have different counts) — a best-effort pairing rather than losing that
    beat's dialogue/action entirely."""
    scr_by_num = {sh["shot"]: sh for sh in scr_shots if isinstance(sh.get("shot"), int)}
    pairs = []
    for i, cam in enumerate(cam_shots):
        num = cam.get("shot_number")
        scr = scr_by_num.get(num) if isinstance(num, int) else None
        if scr is None and i < len(scr_shots):
            scr = scr_shots[i]
        pairs.append((cam, scr or {}))
    return pairs


def _panel_dialogue(scr: dict) -> list[dict]:
    """`screenplay_shots.dialogue` copied as-is, plus a `vo: true` entry for
    `.voiceover` — a separate field from `.dialogue` that's easy to miss
    (see the module docstring; an LLM missed it in practice)."""
    dialogue = [
        {"speaker": d.get("speaker", ""), "line": d.get("line", ""), "vo": False}
        for d in (scr.get("dialogue") or []) if d.get("line")
    ]
    vo = scr.get("voiceover")
    if isinstance(vo, dict) and vo.get("line"):
        dialogue.append({"speaker": vo.get("speaker") or "NARRATOR", "line": vo["line"], "vo": True})
    return dialogue


def _panel_characters_in_frame(cam: dict, dialogue: list[dict], scene_characters: list) -> list:
    """HEURISTIC, not a citation — no artifact actually specifies who's
    visually in a given shot. For a close-shot-family panel (CU/ECU/MCU/OTS/
    POV) with dialogue, assume it's on whoever's speaking (a close-up
    typically follows the speaker or their reaction). Otherwise — a wide/
    establishing/full shot, or a close shot with no dialogue — default to
    everyone present in the scene, since narrower shots are the exception,
    not the rule."""
    shot_type = (cam.get("type") or "").upper()
    if shot_type in _CLOSE_SHOT_TYPES:
        speakers, seen = [], set()
        for d in dialogue:
            sp = d.get("speaker")
            if sp and sp not in seen:
                seen.add(sp)
                speakers.append(sp)
        if speakers:
            return speakers
    return list(scene_characters)


def _panel_sound(cam: dict, scr: dict, audio: dict) -> str:
    """Ambient bed (empty when `audio.silence` is true — a deliberate choice,
    not a gap to fill) plus SFX: the matching screenplay shot's own `sound`
    field if it has one, else the `audio.sound_events` entry whose `moment`
    text overlaps this shot's `moment` — both sourced, never invented."""
    ambient = "" if audio.get("silence") else audio.get("ambient_bed", "")
    sfx = scr.get("sound", "")
    if not sfx:
        moment = (cam.get("moment") or "").lower()
        for ev in audio.get("sound_events", []):
            ev_moment = (ev.get("moment") or "").lower()
            if ev_moment and moment and (ev_moment in moment or moment in ev_moment):
                sfx = ev.get("sound", "")
                break
    return " | ".join(p for p in (ambient, sfx) if p)


def _estimate_duration(shot_type: str, dialogue: list[dict]) -> str:
    """No artifact provides per-shot screen time, so this is a heuristic, not
    a citation: a base duration by shot type (wider/establishing shots read
    longer on screen, close-ups/inserts shorter), extended to cover any
    dialogue in the shot at a natural speaking pace (~2.5 words/sec) plus a
    short buffer, whichever is larger."""
    base = _SHOT_BASE_SECONDS.get((shot_type or "").upper(), 4)
    words = sum(len((d.get("line") or "").split()) for d in dialogue)
    spoken = round(words / 2.5) + 1 if words else 0
    return f"{max(base, spoken)}s"


def _format_total_duration(panels: list[dict]) -> str:
    total = 0
    for p in panels:
        m = re.match(r"(\d+)", p.get("duration") or "")
        if m:
            total += int(m.group(1))
    if total < 60:
        return f"{total}s"
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"


def _fallback_image_prompt(cam: dict, action: str, characters_in_frame: list,
                           cast_lookup: dict, location_desc: str, art: dict) -> str:
    """A serviceable but intentionally simple fallback — the real Veo prompt
    for an actual render is reconstructed from structured fields by
    pipeline._five_part_veo_prompt, not read from here (see the module
    docstring); this only matters for a caller with no casting/scene context
    (e.g. a bare `gen-video` prompt)."""
    subjects = []
    for name in characters_in_frame:
        form = (cast_lookup.get(name) or {}).get("physical_form", "")
        subjects.append(f"{name} ({form})" if form else name)
    subject = " and ".join(subjects) or "the scene"
    bits = [f"{subject} {action}".strip(), "Cinematic, photorealistic"]
    cam_bits = ", ".join(b for b in (
        cam.get("type", ""), (cam.get("angle") or "").lower(),
        (cam.get("movement") or "").lower(), cam.get("lens", ""),
    ) if b)
    if cam_bits:
        bits.append(cam_bits)
    if location_desc:
        bits.append(location_desc)
    mood = ", ".join(b for b in (art.get("color_palette", ""), art.get("lighting", "")) if b)
    if mood:
        bits.append(mood)
    return " ".join(b.rstrip(".") + "." for b in bits if b)


def _build_panel(panel_num: int, cam: dict, scr: dict, bundle: dict, is_last: bool) -> dict:
    art = bundle.get("art", {})
    audio = bundle.get("audio", {})
    cast_lookup = {c["name"]: c for c in bundle.get("cast", [])}
    location_desc = (bundle.get("location") or {}).get("visual_prompt", "")

    dialogue = _panel_dialogue(scr)
    characters_in_frame = _panel_characters_in_frame(cam, dialogue, bundle.get("characters_in_scene", []))
    action = scr.get("description") or cam.get("moment") or ""
    transition = (bundle.get("camera", {}).get("transition_to_next") if is_last else "") or "CUT TO"

    return {
        "panel": panel_num,
        "shot_type": cam.get("type", ""),
        "camera_angle": cam.get("angle", ""),
        "camera_movement": cam.get("movement", ""),
        "lens": cam.get("lens", ""),
        "composition": cam.get("framing", ""),
        "duration": _estimate_duration(cam.get("type", ""), dialogue),
        "characters_in_frame": characters_in_frame,
        "action": action,
        "dialogue": dialogue,
        "sound": _panel_sound(cam, scr, audio),
        "emotional_note": (cam.get("emotional_function") or art.get("emotional_function")
                           or audio.get("emotional_function") or ""),
        "transition": transition,
        "image_prompt": _fallback_image_prompt(cam, action, characters_in_frame, cast_lookup,
                                               location_desc, art),
    }


def _build_scene_board(bundle: dict) -> dict:
    """Deterministically build one scene's full storyboard entry — header,
    visual_overview, audio_overview, panels — from its bundle. See the
    module docstring's "BUILD PATH" section."""
    slug_parsed = bundle.get("slugline_parsed", {})
    location = bundle.get("location") or {}
    art = bundle.get("art", {})
    audio = bundle.get("audio", {})

    cam_shots = bundle.get("camera", {}).get("shots", [])
    scr_shots = bundle.get("screenplay_shots", [])
    pairs = _align_shots(cam_shots, scr_shots)
    panels = [
        _build_panel(i + 1, cam, scr, bundle, is_last=(i == len(pairs) - 1))
        for i, (cam, scr) in enumerate(pairs)
    ]

    header = {
        "slugline": bundle.get("slugline", ""),
        "int_ext": slug_parsed.get("int_ext", ""),
        "location": location.get("name") or slug_parsed.get("location", ""),
        "time_of_day": slug_parsed.get("time_of_day", ""),
        "purpose": bundle.get("purpose", ""),
        "characters": bundle.get("characters_in_scene", []),
        "duration_estimate": _format_total_duration(panels),
    }
    visual_overview = {
        "color_palette": art.get("color_palette", ""),
        "lighting_setup": art.get("lighting", ""),
        "visual_filter": art.get("visual_filter", ""),
        "mood": art.get("emotional_function", ""),
        # Copied straight from bundle.art.key_props (visuals.json), not
        # invented here — this is what actually lets a prop with real
        # dramatic weight reach the Veo render prompt's Context section
        # (pipeline._panel_context); previously key_props existed only in
        # visuals.json and never reached a rendered frame at all.
        "key_props": [p.get("prop", "") for p in art.get("key_props", []) if p.get("prop")],
    }
    audio_overview = {
        "score_cue": audio.get("score_direction", ""),
        "ambient": "" if audio.get("silence") else audio.get("ambient_bed", ""),
        "key_sounds": [
            f"{e.get('moment', '')}: {e.get('sound', '')}".strip(": ")
            for e in audio.get("sound_events", []) if e.get("sound")
        ],
    }

    return {
        "scene_number": bundle.get("scene_number"),
        "header": header,
        "visual_overview": visual_overview,
        "audio_overview": audio_overview,
        "panels": panels,
    }


def _deterministic_storyboard_style(visuals: dict, cinematography: dict, moodboard: dict | None,
                                    genre_name: str, tone: str) -> str:
    """`storyboard_style` is a whole-board judgment, not a per-scene one — no
    single artifact owns it, but several already carry a film-wide summary
    that's a strict superset of what an LLM would otherwise have to invent
    fresh: prefer visuals'/cinematography's own top-level style fields, then
    the moodboard's overall aesthetic, then fall back to genre + tone."""
    bits = [visuals.get("visual_palette", ""), cinematography.get("cinematography_style", "")]
    style = " ".join(b for b in bits if b)
    if style:
        return style
    if moodboard and moodboard.get("overall_aesthetic"):
        return moodboard["overall_aesthetic"]
    return f"{genre_name} — {tone}".strip(" —") or "Cinematic, photorealistic."


def _llm_generate_scene(bundle: dict, logline: str, genre_name: str, tone: str,
                        source_text: str, feedback: str, profile: str) -> tuple[dict | None, str]:
    """The one remaining LLM path — used only when `feedback` (a directed
    creative note) is given, since the deterministic builder above can't
    interpret free text. Same PROMPT/SYSTEM as the original full-generation
    design. Returns (scene_doc_or_None, storyboard_style)."""
    scene_ctx = bundle.pop("_source_context", "") or source_text
    story_blk = _story_block(scene_ctx)
    single_bundle = json.dumps([bundle], ensure_ascii=False, indent=2)
    prompt = llm.with_feedback(
        PROMPT.format(logline=logline, genre=genre_name, tone=tone,
                      story_block=story_blk, bundles=single_bundle),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    result = llm.safe_json(raw)
    returned = result.get("storyboard") or []
    return (returned[0] if returned else None), result.get("storyboard_style", "")


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
    moodboard: dict | None = None,
    source_text: str = "",        # kept for backward compat with old callers
    source: dict | None = None,   # preferred: full source dict with chunks
    profile: str | None = None,
    feedback: str | None = None,
    out: str | Path | None = None,
    existing: dict | None = None,
    revise_keys: set | None = None,
) -> dict:
    """Build the storyboard scene by scene — deterministically by default (see
    the module docstring's "BUILD PATH" section), falling back to the LLM
    (`_llm_generate_scene`) only when `feedback` is given, since applying a
    directed creative note needs actual judgment the deterministic builder
    can't provide. Either way this is scene-by-scene: the source context
    used by the LLM fallback is the specific chunk(s) mapped to that scene
    (from scenes[*].chunk_indices), so the model sees the relevant passage
    rather than a generic head truncation.

    `existing` + `revise_keys` (a set of `scene_number`s) support a scoped
    revision: restricts WHICH scenes' bundles get (re)built at all — for the
    deterministic path this just means fewer scenes to build; for the LLM
    fallback it means fewer LLM calls, same as before. The freshly-built
    scenes are merge-spliced into `existing["storyboard"]` via
    `revision_merge.merge_by_key`; `storyboard_style` (a whole-board
    judgment, not a per-scene one) is kept from `existing` rather than
    adopted from a single revised scene. Panel-level (as opposed to
    whole-scene) storyboard TEXT revision isn't supported here — only whole
    scenes can be targeted; panel-level revision of the rendered VIDEO is a
    separate, purely-visual concern handled by `pipeline.rerender_panels`."""
    profile = profile or llm.agent_profile("storyboard")
    bundles = _scene_bundles(scenes, casting, soundscape, visuals, cinematography,
                             characters=characters, draft=draft, moodboard=moodboard,
                             source=source)
    genre_name = (genre.get("genre") if isinstance(genre, dict) else genre) \
        or structure.get("genre", "drama")
    logline = structure.get("logline", "")
    tone = structure.get("tone", "")

    scoped = revise_keys is not None and existing
    if scoped:
        revise_keys = set(revise_keys)
        bundles = [b for b in bundles if b.get("scene_number") in revise_keys]

    board_scenes = []
    dropped: list[dict] = []

    if feedback:
        # A directed creative note needs actual model judgment to apply —
        # the deterministic builder below can't interpret free text.
        storyboard_style = existing.get("storyboard_style", "") if scoped else ""
        for bundle in bundles:
            expected_num = bundle.get("scene_number")
            scene_doc, style = _llm_generate_scene(bundle, logline, genre_name, tone,
                                                   source_text, feedback, profile)
            if style and not storyboard_style:
                storyboard_style = style
            # Each call is sent exactly ONE scene's bundle, so the response
            # should come back as exactly one storyboard entry for it — but
            # nothing actually enforces that on the model's side. Reconcile
            # deterministically rather than trust the echo, the same way
            # _five_part_veo_prompt reconstructs the Veo prompt from
            # structured fields instead of the model's free text: a scene
            # the model dropped entirely (empty/failed response) is recorded
            # in `dropped_scenes` instead of silently vanishing from the
            # board (mirrors scenes.py's own `dropped_scenes`); a
            # scene_number the model got wrong or omitted is corrected to
            # the one we KNOW is right, since we sent exactly one scene per
            # call — there's no ambiguity to resolve, only bookkeeping to
            # trust ourselves over the model for.
            if scene_doc is None:
                dropped.append({"scene_number": expected_num, "reason": "model returned no storyboard entry"})
                continue
            if scene_doc.get("scene_number") != expected_num:
                scene_doc["scene_number"] = expected_num
            board_scenes.append(scene_doc)
    else:
        storyboard_style = existing.get("storyboard_style", "") if scoped else \
            _deterministic_storyboard_style(visuals, cinematography, moodboard, genre_name, tone)
        board_scenes = [_build_scene_board(bundle) for bundle in bundles]

    # Keep the final scene list in the SAME order scenes.json established,
    # regardless of any incidental reordering (deterministic build already
    # preserves it; this guards the LLM-feedback path too).
    board_scenes.sort(key=lambda s: (s.get("scene_number") is None, s.get("scene_number")))

    if scoped:
        final_storyboard = merge_by_key(existing.get("storyboard", []), board_scenes,
                                        lambda s: s.get("scene_number"), revise_keys)
        return {
            "storyboard_style": storyboard_style,
            "storyboard": final_storyboard,
            "dropped_scenes": dropped,
        }

    return {
        "storyboard_style": storyboard_style,
        "storyboard": board_scenes,
        "dropped_scenes": dropped,
    }
