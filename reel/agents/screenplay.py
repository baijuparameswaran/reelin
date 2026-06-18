"""Screenplay agent: draft Fountain-formatted pages from the scene list.

Fountain (https://fountain.io) is plain-text screenplay markup, so the output is
human-readable, diff-able, and importable into most screenwriting tools.

On CPU-only hardware, drafting every scene is slow, so by default we draft the
first `max_scenes` scenes fully — enough to prove the end-to-end slice. Raise it
(or run the quality profile on a faster box) to draft the whole script.
"""
from __future__ import annotations

import json

from .. import llm

SYSTEM = (
    "You are a professional screenwriter. You write in Fountain screenplay "
    "format: scene headings in CAPS (INT./EXT. LOCATION - TIME), tight action "
    "lines in present tense, and character cues in CAPS above dialogue. Output "
    "only the screenplay text for the requested scene — no commentary."
)

PROMPT = """\
Write the screenplay for ONE scene in Fountain format.

Story logline: {logline}
Tone: {tone}

Characters who may appear:
{characters}

Scene to write:
- Slugline: {slugline}
- What happens: {summary}
- Dramatic purpose: {purpose}
- Featured characters: {scene_chars}

Write vivid but economical action and natural dialogue. Begin with the slugline.
"""


def _char_brief(characters: dict) -> str:
    rows = []
    for c in characters.get("characters", [])[:8]:
        rows.append(f"- {c.get('name','?')}: {c.get('description','')}".strip())
    return "\n".join(rows) or "- (none extracted)"


def draft_screenplay(
    source: dict,
    structure: dict,
    characters: dict,
    scenes: dict,
    max_scenes: int = 3,
    profile: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("screenplay")
    logline = structure.get("logline", source["title"])
    tone = structure.get("tone", "")
    char_brief = _char_brief(characters)

    scene_list = scenes.get("scenes", [])
    drafted = []
    for scene in scene_list[:max_scenes]:
        prompt = PROMPT.format(
            logline=logline,
            tone=tone,
            characters=char_brief,
            slugline=scene.get("slugline", "INT. LOCATION - DAY"),
            summary=scene.get("summary", ""),
            purpose=scene.get("purpose", ""),
            scene_chars=", ".join(scene.get("characters", [])) or "as needed",
        )
        text = llm.generate(prompt, profile=profile, system=SYSTEM)
        drafted.append({"number": scene.get("number"), "fountain": text})

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
        f"Draft date: {_today()}",
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


def _today() -> str:
    from datetime import date
    return date.today().isoformat()
