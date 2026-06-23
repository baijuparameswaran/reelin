"""Moodboard agent: the film's visual-tone bible, set once for the whole movie.

A moodboard is the pre-production reference that aligns every department on one
aesthetic — the color story, lighting mood, textures, atmosphere, and visual
influences the film should evoke. It sits one level ABOVE the per-scene `visuals`
stage (which implements the look scene by scene) and, like `genre`, is a single
cross-cutting reference rather than per-scene data.

In the pipeline it runs right after structure (it needs the logline/genre/tone/
themes) and then STEERS the downstream creative stages — casting, soundscape,
visuals, cinematography, storyboard — via the shared creative-direction hook
(`llm.set_direction`), so they all compose toward the same look without any of
those agents needing to change.

Runs on the OPEN models (Ollama) via `reel.models.text`, per the project policy
that Gemini is used only for image + video generation.
"""
from __future__ import annotations

import json

from .. import models

SYSTEM = (
    "You are a film production designer and visual director assembling a "
    "moodboard. From a story's structure and genre you define ONE coherent "
    "aesthetic — color, light, texture, atmosphere, influences — that every "
    "department will work toward. You always respond with valid JSON and nothing "
    "else."
)

PROMPT = """\
{revision_note}Define the MOODBOARD — the film-wide visual-tone bible — for the
adaptation below. This is one coherent aesthetic the whole movie works toward, not
per-scene detail.

Film:
- Logline: {logline}
- Genre: {genre}
- Tone: {tone}
- Themes: {themes}
{genre_block}{story_block}
Respond with JSON in exactly this shape:
{{
  "overall_aesthetic": "1-2 sentence statement of the film's visual vision",
  "color_story": "how the palette is used and evolves across the film",
  "palette": ["dominant color (name or hex)", "accent", "..."],
  "lighting_mood": "the film's signature lighting (quality, direction, contrast)",
  "texture_materials": ["material / texture the film leans on", "..."],
  "atmosphere_keywords": ["evocative mood word", "..."],
  "visual_influences": ["a film, photographer, painter, or movement it evokes", "..."],
  "wardrobe_mood": "the overall costume/character-look direction",
  "sound_mood": "the sonic character that matches this look",
  "do": ["aesthetic choice to embrace", "..."],
  "avoid": ["aesthetic choice to avoid (off-tone)", "..."],
  "tiles": [
    {{"label": "what this reference frame represents",
      "image_prompt": "a concise prompt to render this moodboard reference tile"}}
  ]
}}

Make it specific to THIS story and genre, internally consistent, and usable as a
brief for casting, art, lighting, camera, and sound. Provide exactly {tiles} \
reference tile(s) — these correspond to the scenes being rendered.
"""


def design_moodboard(structure: dict, source_text: str = "", genre: dict | str | None = None,
                     max_scenes: int | None = None,
                     profile: str | None = None, feedback: str | None = None) -> dict:
    """Produce the film-wide moodboard from the structure (and genre/story).

    The aesthetic fields are film-wide, but the render-ready `tiles` are limited to
    `max_scenes` — the same cap that bounds how many scenes get rendered by default —
    so the moodboard's rendering scope matches the scenes actually being rendered.
    Runs on the open models via the model abstraction."""
    n_tiles = max_scenes if (isinstance(max_scenes, int) and max_scenes > 0) else 4
    themes = structure.get("themes", [])
    genre_name = (genre.get("genre") if isinstance(genre, dict) else genre) \
        or structure.get("genre", "drama")
    genre_block = ""
    if isinstance(genre, dict) and genre:
        gb = {k: genre.get(k) for k in ("subgenre", "tone", "conventions",
                                        "visual_language", "sound_language") if genre.get(k)}
        if gb:
            genre_block = f"- Genre direction: {json.dumps(gb, ensure_ascii=False)}\n"
    story_block = f"\nStory (for texture):\n{source_text[:2500]}\n" if source_text else ""
    revision_note = (f"REVISION REQUEST (apply throughout):\n{feedback}\n\n" if feedback else "")
    prompt = PROMPT.format(
        revision_note=revision_note,
        logline=structure.get("logline", ""),
        genre=genre_name,
        tone=structure.get("tone", ""),
        themes=", ".join(themes) if isinstance(themes, list) else str(themes),
        genre_block=genre_block,
        story_block=story_block,
        tiles=n_tiles,
    )
    raw = models.text(prompt, system=SYSTEM,
                      profile=profile or models.agent_profile("moodboard"),
                      as_json=True)
    board = models.safe_json(raw)
    # Hard-cap the tiles to the render scope, in case the model over-produces.
    if isinstance(board.get("tiles"), list):
        board["tiles"] = board["tiles"][:n_tiles]
    return board


def guidance(moodboard: dict | None) -> str:
    """Compact aesthetic directive to steer every creative stage toward the board."""
    if not moodboard or not moodboard.get("overall_aesthetic"):
        return ""
    bits = [f"MOODBOARD: {moodboard['overall_aesthetic']}"]
    if moodboard.get("color_story"):
        bits.append(f"Color: {moodboard['color_story']}.")
    if moodboard.get("palette"):
        bits.append("Palette: " + ", ".join(str(c) for c in moodboard["palette"][:6]) + ".")
    if moodboard.get("lighting_mood"):
        bits.append(f"Light: {moodboard['lighting_mood']}.")
    if moodboard.get("texture_materials"):
        bits.append("Textures: " + ", ".join(str(t) for t in moodboard["texture_materials"][:5]) + ".")
    if moodboard.get("atmosphere_keywords"):
        bits.append("Atmosphere: " + ", ".join(str(a) for a in moodboard["atmosphere_keywords"][:6]) + ".")
    if moodboard.get("sound_mood"):
        bits.append(f"Sound: {moodboard['sound_mood']}.")
    if moodboard.get("avoid"):
        bits.append("Avoid: " + "; ".join(str(a) for a in moodboard["avoid"][:3]) + ".")
    return ("Hold the whole film to this moodboard. " + " ".join(bits)).strip()
