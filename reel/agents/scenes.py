"""Scene agent: segment the story into a numbered scene list.

The source text is the ONLY authority — every scene must correspond to an actual
event in the source. Structural beats are a secondary ordering scaffold only.

Each scene also carries a `location` — the plain name of its physical setting,
identical across every scene set in the same place (independent of the DAY/NIGHT
slugline formatting). This locks the scene→location association explicitly
rather than leaving it to be fuzzy-matched from slugline text later, and is the
seed for the casting agent's location-casting step (see `reel/agents/casting.py`
`_location_entries`) — a location need not recur across scenes to be worth
naming precisely; even a one-scene setting benefits from a consistent reference
across that scene's own panels.
"""
from __future__ import annotations

import json
import re

from .. import llm
from ..llm import MAX_CHARS
from ..revision_merge import merge_by_key
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
7. No unnecessary repeats: never split one event into two overlapping scenes, and
   never list two scenes whose `summary`/`source_line` cover substantially the same
   moment. Each scene must earn its place with something the others don't already
   cover.
8. `location` is the plain NAME of the physical setting (e.g. "Rusty Anchor Bar",
   "Lumen Field", "Antwerp café") — NOT the full slugline formatting (no "INT./EXT."
   or "- DAY/NIGHT"). If two or more scenes are set in the same real place, they
   MUST use the exact identical `location` string, even if their sluglines differ
   (e.g. one is DAY and another is NIGHT at the same place) — this name is the
   anchor a later stage uses to keep that location visually consistent, so
   inconsistent naming of the same place defeats the purpose. A location need not
   recur to deserve its own name; name it precisely either way.

Respond with JSON in exactly this shape (no extra keys, no commentary):
{{
  "scenes": [
    {{
      "number": 1,
      "slugline": "INT./EXT. LOCATION - DAY/NIGHT",
      "location": "plain name of the physical setting, identical across every scene set there",
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


def _validate(scenes: list[dict], source_text: str) -> tuple[list[dict], list[dict]]:
    """Split into (valid, dropped) by whether source_line is found in the source
    text. Dropped scenes are returned (not silently discarded) so the caller can
    surface them — a scene with a hallucinated source_line is still worth showing
    the operator, e.g. to explain a gap in the scene numbering at the gate.

    Whitespace is collapsed before matching: the ingested source text keeps its
    original hard line-wraps (e.g. "Marcel\\nhad watched"), but a model-written
    source_line naturally uses normal spacing ("Marcel had watched") — an
    unnormalized check flags a real, verbatim quote as hallucinated whenever its
    prefix happens to straddle a line break."""
    norm_source = re.sub(r"\s+", " ", source_text).lower()
    valid, dropped = [], []
    for sc in scenes:
        sl = (sc.get("source_line") or "").strip()
        norm_sl = re.sub(r"\s+", " ", sl)
        if norm_sl and len(norm_sl) >= 5 and norm_sl[:8].lower() not in norm_source:
            dropped.append(dict(sc, drop_reason="source_line not found in source text"))
            continue
        valid.append(sc)
    return valid, dropped


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
    existing: dict | None = None,
    revise_keys: set | None = None,
) -> dict:
    """`existing` + `revise_keys` (a set of scene `number`s) support a scoped
    revision: the model still sees the FULL source text and re-segments the
    whole story (scene boundaries need full-story awareness to stay
    consistent), but the caller only trusts its output for the numbers in
    `revise_keys` — every other scene number is spliced back in
    byte-identical from `existing["scenes"]`. Scene insertion/deletion/
    reordering is out of scope for a scoped revision in v1 (see
    `reel.artifact_diff`'s `allow_add=False, allow_remove=False` for
    "scenes") — a revision that changes the scene COUNT should go through
    `reel.artifact_diff.diff_artifact`'s drastic path (a full, non-scoped
    regeneration) instead of `revise_keys`."""
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
    scenes, dropped = _validate(scenes, source_text)
    scenes = _map_chunks(scenes, source)
    if revise_keys is not None and existing:
        scenes = merge_by_key(existing.get("scenes", []), scenes,
                              lambda s: s.get("number"), revise_keys)
    result["scenes"] = scenes
    result["dropped_scenes"] = dropped
    return result
