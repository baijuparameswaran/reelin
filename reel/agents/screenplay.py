"""Screenplay agent: draft Fountain-formatted pages from the scene list.

Fountain (https://fountain.io) is plain-text screenplay markup, so the output is
human-readable, diff-able, and importable into most screenwriting tools.

On CPU-only hardware, drafting every scene is slow, so by default we draft the
first `max_scenes` scenes fully — enough to prove the end-to-end slice. Raise it
(or run the quality profile on a faster box) to draft the whole script.
"""
from __future__ import annotations

import json
from datetime import date

from .. import llm

SYSTEM = (
    "You are a professional screenwriter. You write in Fountain screenplay "
    "format: scene headings in CAPS (INT./EXT. LOCATION - TIME), tight action "
    "lines in present tense, and character cues in CAPS above dialogue. Output "
    "only the screenplay text for the requested scene — no commentary."
)

PROMPT = """\
{revision_note}Write the screenplay for ONE scene in Fountain format.

Story logline: {logline}
Tone: {tone}

Characters in this scene:
{characters}

Scene to write:
- Slugline: {slugline}
- What happens: {summary}
- Dramatic purpose: {purpose}
{soundscape_block}{visuals_block}{cinema_block}
Write vivid but economical action and natural dialogue. Begin with the slugline.
"""


def _soundscape_lookup(soundscape: dict) -> dict[int, dict]:
    return {s.get("scene_number"): s for s in soundscape.get("soundscapes", [])}


def _scene_soundscape_block(scene_number: int, lookup: dict[int, dict]) -> str:
    s = lookup.get(scene_number)
    if not s:
        return ""
    lines = ["Soundscape for this scene:"]
    if s.get("silence"):
        lines.append("- Silence (intentional — no ambient audio)")
    elif s.get("ambient_bed"):
        lines.append(f"- Ambient: {s['ambient_bed']}")
    for ev in s.get("sound_events", []):
        lines.append(f"- Audio cue at '{ev.get('moment', '')}': {ev.get('sound', '')}")
    if s.get("emotional_function"):
        lines.append(f"- Emotional function: {s['emotional_function']}")
    return "\n".join(lines) + "\n"


def _visuals_lookup(visuals: dict) -> dict[int, dict]:
    return {s.get("scene_number"): s for s in visuals.get("scenes", [])}


def _scene_visuals_block(scene_number: int, lookup: dict[int, dict]) -> str:
    v = lookup.get(scene_number)
    if not v:
        return ""
    lines = ["Visual design for this scene:"]
    if v.get("color_palette"):
        lines.append(f"- Color: {v['color_palette']}")
    if v.get("visual_filter"):
        lines.append(f"- Filter/grade: {v['visual_filter']}")
    if v.get("lighting"):
        lines.append(f"- Lighting: {v['lighting']}")
    for p in v.get("key_props", []):
        lines.append(f"- Prop '{p.get('prop', '')}': {p.get('function', '')}")
    for m in v.get("visual_moments", []):
        lines.append(f"- Visual at '{m.get('moment', '')}': {m.get('visual', '')}")
    if v.get("emotional_function"):
        lines.append(f"- Emotional function: {v['emotional_function']}")
    return "\n".join(lines) + "\n"


def _cinema_lookup(cinematography: dict) -> dict[int, dict]:
    return {s.get("scene_number"): s for s in cinematography.get("scenes", [])}


def _scene_cinema_block(scene_number: int, lookup: dict[int, dict]) -> str:
    c = lookup.get(scene_number)
    if not c:
        return ""
    lines = ["Camera coverage for this scene:"]
    if c.get("coverage"):
        lines.append(f"- Coverage: {c['coverage']}")
    for shot in c.get("shots", []):
        num = shot.get("shot_number", "?")
        moment = shot.get("moment", "")
        label = f"  Shot {num}" + (f" ({moment})" if moment else "")
        parts = [shot.get("type", ""), shot.get("angle", ""), shot.get("movement", "")]
        detail = ", ".join(p for p in parts if p)
        lines.append(f"{label}: {detail}")
        if shot.get("lens"):
            lines[-1] += f" — {shot['lens']}"
        if shot.get("framing"):
            lines.append(f"    Framing: {shot['framing']}")
    if c.get("transition_to_next"):
        lines.append(f"- Transition: {c['transition_to_next']}")
    return "\n".join(lines) + "\n"


def _char_lookup(characters: dict) -> dict[str, dict]:
    return {c.get("name", ""): c for c in characters.get("characters", [])}


def _scene_char_brief(names: list[str], lookup: dict[str, dict]) -> str:
    rows = []
    for name in names or lookup.keys():
        c = lookup.get(name, {"name": name})
        kind = c.get("kind", "")
        tag = f" [{kind}]" if kind and kind != "person" else ""
        parts = [f"- {c.get('name', name)}{tag}: {c.get('description', '')}"]
        if c.get("appearance"):
            parts.append(f"  Appearance: {c['appearance']}")
        if c.get("voice"):
            parts.append(f"  Voice: {c['voice']}")
        if c.get("mannerisms"):
            parts.append(f"  Mannerisms: {c['mannerisms']}")
        rows.append("\n".join(p for p in parts if p.strip()))
    return "\n".join(rows) or "- (none extracted)"


def draft_screenplay(
    source: dict,
    structure: dict,
    characters: dict,
    scenes: dict,
    soundscape: dict | None = None,
    visuals: dict | None = None,
    cinematography: dict | None = None,
    max_scenes: int = 3,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("screenplay")
    logline = structure.get("logline", source["title"])
    tone = structure.get("tone", "")
    char_lookup = _char_lookup(characters)
    sound_lookup = _soundscape_lookup(soundscape or {})
    vis_lookup = _visuals_lookup(visuals or {})
    cinema_lookup = _cinema_lookup(cinematography or {})
    revision_note = (
        f"REVISION REQUEST (apply throughout all scenes):\n{feedback}\n\n"
        if feedback else ""
    )

    scene_list = scenes.get("scenes", [])
    drafted = []
    for scene in scene_list[:max_scenes]:
        scene_num = scene.get("number")
        scene_chars = scene.get("characters", [])
        prompt = PROMPT.format(
            revision_note=revision_note,
            logline=logline,
            tone=tone,
            characters=_scene_char_brief(scene_chars, char_lookup),
            slugline=scene.get("slugline", "INT. LOCATION - DAY"),
            summary=scene.get("summary", ""),
            purpose=scene.get("purpose", ""),
            soundscape_block=_scene_soundscape_block(scene_num, sound_lookup),
            visuals_block=_scene_visuals_block(scene_num, vis_lookup),
            cinema_block=_scene_cinema_block(scene_num, cinema_lookup),
        )
        text = llm.generate(prompt, profile=profile, system=SYSTEM)
        drafted.append({"number": scene_num, "fountain": text})

    return {
        "drafted_count": len(drafted),
        "total_scenes": len(scene_list),
        "scenes": drafted,
    }


def to_fountain(source: dict, structure: dict, draft: dict) -> str:
    """Assemble drafted scenes into a single Fountain document with title page."""
    title = source.get("title", "Untitled")
    logline = structure.get("logline", "")
    parts = [
        f"Title: {title}",
        "Credit: adapted by",
        "Author: reel (AI draft)",
        f"Draft date: {date.today().isoformat()}",
        "",
        f"= {logline}" if logline else "",
        "",
        "====",
        "",
    ]
    for s in draft.get("scenes", []):
        parts.append(s["fountain"].strip())
        parts.append("")
    if draft.get("drafted_count", 0) < draft.get("total_scenes", 0):
        parts.append(
            f"[[ Draft covers {draft['drafted_count']} of "
            f"{draft['total_scenes']} scenes. Raise --max-scenes to continue. ]]"
        )
    return "\n".join(parts).strip() + "\n"
