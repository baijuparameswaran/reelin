"""Scene agent: segment the story into a numbered scene list.

The source text is the ONLY authority — every scene must correspond to an actual
event in the source. Structural beats are a secondary ordering scaffold only.
"""
from __future__ import annotations

import json

from .. import llm
from ..llm import MAX_CHARS
from .ingest import chunk_text, CHUNK_SIZE

SYSTEM = (
    "You are a screenwriter breaking a story into filmable scenes. Each scene "
    "happens in one location and continuous time. You always respond with valid "
    "JSON and nothing else."
)

PROMPT = """\
Break the story below into filmable scenes. Aim for {target}, but use FEWER scenes
if the story doesn't have enough distinct events — never invent scenes to hit a count.

STRICT RULES:
1. SOURCE TEXT IS THE ONLY AUTHORITY. Every scene must correspond to an actual
   event, location, or moment explicitly present in the source text below.
2. No invented scenes, characters, plot points, or locations. If the source doesn't
   describe it, it cannot appear in the scene list.
3. `source_line` is mandatory: copy a SHORT verbatim phrase (5-15 words) from the
   source text that anchors this scene. If you cannot find a matching phrase, the
   scene does not belong in the list.
4. `summary` must describe only what the source text says — no embellishment.
5. Use the EXACT character names that appear in the source text.
6. The structural beats below are a secondary ordering hint only. Where they conflict
   with the source text, the source text wins.

Respond with JSON in exactly this shape (no extra keys, no commentary):
{{
  "scenes": [
    {{
      "number": 1,
      "slugline": "INT./EXT. LOCATION - DAY/NIGHT",
      "source_line": "short verbatim phrase from the source text that this scene covers",
      "summary": "one or two sentences of what actually happens in the source",
      "characters": ["EXACT NAME as in source", "..."],
      "purpose": "why this scene exists dramatically"
    }}
  ]
}}

SOURCE MATERIAL — primary fidelity anchor (title: {title}):
\"\"\"
{text}
\"\"\"

STRUCTURAL BEATS (secondary scaffold — ordering/emphasis only, not a replacement
for what the source actually says):
{beats}
"""


def _validate(scenes: list[dict], source_text: str) -> list[dict]:
    """Strip scenes whose source_line is not found in the source text."""
    valid = []
    for sc in scenes:
        sl = (sc.get("source_line") or "").strip()
        if sl and len(sl) >= 5:
            if sl[:8].lower() not in source_text.lower():
                continue      # phrase not in source — likely hallucinated
        valid.append(sc)
    return valid


def _map_chunks(scenes: list[dict], source: dict) -> list[dict]:
    """Add chunk_indices to every scene by matching its source_line against chunks.

    Each scene gets the indices of chunks that contain its source_line, plus the
    immediately following chunk for boundary coverage. Falls back to [0] when no
    match is found (short story where the whole text is chunk 0).
    """
    chunks: list[dict] = source.get("chunks") or []
    if not chunks:
        for sc in scenes:
            sc["chunk_indices"] = [0]
        return scenes

    for sc in scenes:
        sl = (sc.get("source_line") or "").strip().lower()
        probe = sl[:8] if sl else ""
        indices: set[int] = set()
        if probe:
            for ch in chunks:
                if probe in ch["text"].lower():
                    indices.add(ch["index"])
                    # Include the next chunk for scenes near a chunk boundary.
                    if ch["index"] + 1 < len(chunks):
                        indices.add(ch["index"] + 1)
        sc["chunk_indices"] = sorted(indices) or [0]
    return scenes


def segment_scenes(
    source: dict,
    structure: dict,
    target: str = "8-14 scenes",
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    profile = profile or llm.agent_profile("scenes")
    beats = json.dumps(structure.get("three_act", {}), ensure_ascii=False, indent=2)
    source_text = source["text"][:MAX_CHARS]
    prompt = llm.with_feedback(
        PROMPT.format(
            target=target,
            beats=beats,
            title=source["title"],
            text=source_text,
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    result = llm.safe_json(raw)
    scenes = result.get("scenes") or []
    scenes = _validate(scenes, source_text)
    scenes = _map_chunks(scenes, source)
    result["scenes"] = scenes
    return result
