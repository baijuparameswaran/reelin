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
    "You are a professional screenwriter and shot-list director. You break a "
    "scene into clearly numbered SHOTS, write tight present-tense action, and "
    "attribute every spoken line to a named speaker. You mark off-screen voices "
    "(O.S.), off-screen narration / interior monologue as voice-over (V.O.), and "
    "use the camera coverage provided. You always respond with valid JSON and "
    "nothing else."
)

PROMPT = """\
{revision_note}Break ONE scene into a numbered shot list with fully attributed
dialogue. Use the camera coverage below to decide the shots.

Story logline: {logline}
Tone: {tone}
{story_block}
Characters in this scene (use these EXACT names as speakers):
{characters}
{casting_block}{prior_scenes_block}
Scene to write:
- Slugline: {slugline}
- What happens: {summary}
- Dramatic purpose: {purpose}
{soundscape_block}{visuals_block}{cinema_block}
Respond with JSON in exactly this shape:
{{
  "scene_number": {scene_number},
  "slugline": "{slugline}",
  "shots": [
    {{
      "shot": 1,
      "shot_type": "WIDE | MEDIUM | CLOSE-UP | OTS | POV | INSERT | TWO-SHOT | ...",
      "description": "the action / what the camera sees in this shot, present tense",
      "voiceover": {{"speaker": "NAME", "line": "narration or interior monologue heard over the shot"}},
      "dialogue": [
        {{"speaker": "NAME (exact)", "modifier": "" or "O.S." or "CONT'D",
          "parenthetical": "optional acting/delivery note, no parens", "line": "the spoken line"}}
      ],
      "sound": "key audio under this shot (from the soundscape)"
    }}
  ]
}}

Rules:
- Stay faithful to the source material — do not invent events, relationships, or
  dialogue not supported by the story excerpt above.
- Derive shots from the camera coverage when given; otherwise pick the key beats.
- EVERY dialogue line MUST have a `speaker` that matches a character name above.
- Do NOT repeat or contradict anything already established in prior scenes.
- Use `voiceover` only when narration / interior monologue genuinely serves the
  scene (e.g. reflection over action); set it to null when not needed.
- Use modifier "O.S." for a speaker heard but not seen; "V.O." voice is the
  `voiceover` field. Keep `parenthetical` short or "".
- Keep action economical and shootable; no camera directions inside `description`
  beyond what the shot_type implies.
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


def _casting_lookup(casting: dict) -> dict[str, dict]:
    return {c.get("name", ""): c for c in (casting or {}).get("casting", [])}


def _scene_casting_brief(names: list[str], lookup: dict[str, dict]) -> str:
    """The LOCKED on-screen look per character (casting), so action descriptions
    stay true to what's actually rendered for video."""
    if not lookup:
        return ""
    rows = ["Locked on-screen look (keep action true to this):"]
    for name in names or lookup.keys():
        c = lookup.get(name)
        if not c:
            continue
        ch = c.get("character", c)
        bits = [ch.get("physical_form", ""), ch.get("age", ""), ch.get("costume", ch.get("wardrobe", "")),
                ch.get("defining_feature", ""), ch.get("mannerism", "")]
        detail = "; ".join(b for b in bits if b)
        rows.append(f"- {name}: {detail}" if detail else f"- {name}")
    return ("\n".join(rows) + "\n") if len(rows) > 1 else ""


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


def _story_block(source_text: str, max_chars: int = 2500) -> str:
    """Truncated source excerpt so the agent stays anchored to the actual story."""
    text = (source_text or "").strip()
    if not text:
        return ""
    excerpt = text[:max_chars]
    if len(text) > max_chars:
        excerpt += "\n[… excerpt truncated]"
    return f"Source material (stay faithful — adapt from this, do not invent):\n{excerpt}\n"


def _prior_scenes_block(drafted: list[dict]) -> str:
    """Compact summary of already-drafted scenes for cross-scene continuity."""
    if not drafted:
        return ""
    lines = ["Previously drafted scenes (do NOT repeat, contradict, or recap):"]
    for d in drafted:
        slug = d.get("slugline", f"Scene {d.get('number', '?')}")
        shots = d.get("shots", [])
        # One-line summary: first shot description + any dialogue speakers seen
        first_desc = shots[0].get("description", "") if shots else ""
        speakers = sorted({
            dl.get("speaker", "") for s in shots
            for dl in (s.get("dialogue") or []) if dl.get("speaker")
        })
        detail = first_desc[:80] + ("…" if len(first_desc) > 80 else "")
        speaker_note = f" [speakers: {', '.join(speakers)}]" if speakers else ""
        lines.append(f"  - {slug}: {detail}{speaker_note}")
    return "\n".join(lines) + "\n"


def draft_screenplay(
    source: dict,
    structure: dict,
    characters: dict,
    scenes: dict,
    soundscape: dict | None = None,
    visuals: dict | None = None,
    cinematography: dict | None = None,
    casting: dict | None = None,
    max_scenes: int = 3,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("screenplay")
    logline = structure.get("logline", source["title"])
    tone = structure.get("tone", "")
    source_text = source.get("text", "")
    char_lookup = _char_lookup(characters)
    casting_lookup = _casting_lookup(casting or {})
    sound_lookup = _soundscape_lookup(soundscape or {})
    vis_lookup = _visuals_lookup(visuals or {})
    cinema_lookup = _cinema_lookup(cinematography or {})
    revision_note = (
        f"REVISION REQUEST (apply throughout all scenes):\n{feedback}\n\n"
        if feedback else ""
    )
    story_blk = _story_block(source_text)

    scene_list = scenes.get("scenes", [])
    drafted = []
    for scene in scene_list[:max_scenes]:
        scene_num = scene.get("number")
        scene_chars = scene.get("characters", [])
        slugline = scene.get("slugline", "INT. LOCATION - DAY")
        prompt = PROMPT.format(
            revision_note=revision_note,
            logline=logline,
            tone=tone,
            story_block=story_blk,
            characters=_scene_char_brief(scene_chars, char_lookup),
            casting_block=_scene_casting_brief(scene_chars, casting_lookup),
            prior_scenes_block=_prior_scenes_block(drafted),
            slugline=slugline,
            scene_number=scene_num if scene_num is not None else 0,
            summary=scene.get("summary", ""),
            purpose=scene.get("purpose", ""),
            soundscape_block=_scene_soundscape_block(scene_num, sound_lookup),
            visuals_block=_scene_visuals_block(scene_num, vis_lookup),
            cinema_block=_scene_cinema_block(scene_num, cinema_lookup),
        )
        raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
        scene_doc = llm.safe_json(raw)
        scene_doc.setdefault("scene_number", scene_num)
        scene_doc.setdefault("slugline", slugline)
        scene_doc.setdefault("shots", [])
        scene_doc["number"] = scene_num
        scene_doc["fountain"] = scene_to_fountain(scene_doc)  # rendered view
        drafted.append(scene_doc)

    return {
        "drafted_count": len(drafted),
        "total_scenes": len(scene_list),
        "scenes": drafted,
    }


def _cue(speaker: str, modifier: str = "") -> str:
    name = (speaker or "").upper().strip()
    mod = (modifier or "").strip().strip("()")
    return f"{name} ({mod})" if mod else name


def scene_to_fountain(scene: dict) -> str:
    """Render one structured scene (shots + attributed dialogue + V.O.) as Fountain."""
    out = [scene.get("slugline", "INT. LOCATION - DAY").upper(), ""]
    for shot in scene.get("shots", []):
        num = shot.get("shot", "")
        stype = (shot.get("shot_type") or "").upper()
        header = f"SHOT {num}" + (f" — {stype}" if stype else "")
        out.append(f"!{header}")                      # '!' forces an action/shot line
        if shot.get("description"):
            out.append(shot["description"].strip())
        out.append("")
        vo = shot.get("voiceover")
        if isinstance(vo, dict) and vo.get("line"):
            out.append(_cue(vo.get("speaker", "NARRATOR"), "V.O."))
            out.append(vo["line"].strip())
            out.append("")
        for d in shot.get("dialogue", []) or []:
            if not d.get("line"):
                continue
            out.append(_cue(d.get("speaker", ""), d.get("modifier", "")))
            if d.get("parenthetical"):
                out.append(f"({d['parenthetical'].strip().strip('()')})")
            out.append(d["line"].strip())
            out.append("")
    return "\n".join(out).strip()


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
