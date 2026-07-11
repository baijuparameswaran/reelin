"""Scene agent: segment the story into a numbered scene list.

The source text is the ONLY authority — every scene must correspond to an actual
event in the source. Structural beats are a secondary ordering scaffold only.

This is the single most consequential stage for downstream cost: every scene
becomes a separate rendered video clip (or several, one per shot) further down
the pipeline, so the prompt actively steers toward the FEWEST scenes that can
still tell the story faithfully — merging continuous beats that share a
location and continuous time into one scene, splitting only on a real
location/time/purpose change (see STRICT RULE 9, "MINIMIZE SCENE COUNT",
below). This is a cost/pacing bias, not a fidelity relaxation: the existing
"never invent scenes to hit a count" / "no unnecessary repeats" rules already
prevent the opposite failure mode (dropping real events to save a scene).

Each scene also carries a `location` — the plain name of its physical setting,
identical across every scene set in the same place (independent of the DAY/NIGHT
slugline formatting). This locks the scene→location association explicitly
rather than leaving it to be fuzzy-matched from slugline text later, and is the
seed for the casting agent's location-casting step (see `reel/agents/casting.py`
`_location_entries`) — a location need not recur across scenes to be worth
naming precisely; even a one-scene setting benefits from a consistent reference
across that scene's own panels.

Character names get the same anchoring treatment: when `characters` (from
characters.json) is passed in, every scene's `characters` list is required to
reuse those exact settled names (`_canonical_names_block` in the prompt,
`_reconcile_character_names` as a deterministic fallback) — otherwise a
character the source text only ever describes in prose, never names outright
(e.g. "a beautiful young woman"), can get independently re-derived slightly
differently by the `characters` agent ("Young Woman") and this one ("Woman"),
which silently breaks every later name-keyed lookup (casting image rendering,
storyboard/screenplay character briefs) for that person.

Each scene also captures the actual PORTION of the story it covers, not just
the short `source_line` anchor (5-15 words, used only for hallucination-
checking and chunk-matching): `_attach_source_excerpts` deterministically (no
LLM, no hallucination risk) partitions the source text into one contiguous
span per scene, from that scene's `source_line` position to the next scene's,
and attaches it as `source_excerpt` plus a `word_count` metadatum. This is the
authoritative per-scene text every downstream per-scene agent should ground
against — `ingest.scene_source_context` prefers it over the older, coarser
chunk-based join (which only approximates scene boundaries at ~3000-char
chunk granularity and can pull in neighboring scenes' text) whenever it's
present.

Each scene also carries `props` — notable physical objects the source text
explicitly mentions for that scene (rule 10), source-grounded the same way
`characters` is. This is the earliest, most-authoritative point in the
pipeline a prop can be identified, and it seeds two downstream uses: (1)
`casting.py`'s `_location_entries` aggregates every scene's `props` per
`location` so a location's rendered reference can incorporate genuinely
fixed/recurring decor (not every scene's props — see that function's
docstring for the fixed-vs-transient distinction); (2) `visuals.py`'s
per-scene art design gets them as grounding context for its own `key_props`
(which, unlike this field, is a creative judgment of DRAMATIC weight, not a
plain inventory — see `visuals.py`'s docstring). Without a source-grounded
starting point, both of those downstream steps had no better option than
inventing props from scratch, which had no fidelity anchor at all.
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
Break the story below into filmable scenes. Aim for {target}. Never invent
scenes to hit a count — but every scene below becomes a separate rendered
video clip downstream, so also never split what can be told as ONE continuous
scene; prefer the FEWEST scenes that still tell the story faithfully (see
STRICT RULE 9 below).
{structure_note}
STRICT RULES:
1. SOURCE TEXT IS THE ONLY AUTHORITY. Every scene must correspond to an actual
   event, location, or moment explicitly present in the source text below.
2. No invented scenes, characters, plot points, or locations. If the source doesn't
   describe it, it cannot appear in the scene list.
3. `source_line` is mandatory: copy a SHORT verbatim phrase (5-15 words) from the
   source text that anchors this scene. If you cannot find a matching phrase, the
   scene does not belong in the list.
4. `summary` must describe only what the source text says — no embellishment.
5. Use the EXACT character names that appear in the source text — or, when a
   CANONICAL CHARACTER NAMES list is provided below, the exact matching name
   from THAT list (it already settled on one consistent name per character;
   never shorten, rephrase, or re-derive your own variant of it).
6. The structural beats below are a secondary ordering hint only. Where they conflict
   with the source text, the source text wins.
7. No unnecessary repeats: never split one event into two overlapping scenes, and
   never list two scenes whose `summary`/`source_line` cover substantially the same
   moment. Each scene must earn its place with something the others don't already
   cover.
8. `location` is the plain NAME of the physical setting (e.g. "the harbor tavern",
   "a wheat field", "a corner café" — generic illustrations of the FORMAT only;
   use the source's own name for the place when it has one) — NOT the full
   slugline formatting (no "INT./EXT."
   or "- DAY/NIGHT"). If two or more scenes are set in the same real place, they
   MUST use the exact identical `location` string, even if their sluglines differ
   (e.g. one is DAY and another is NIGHT at the same place) — this name is the
   anchor a later stage uses to keep that location visually consistent, so
   inconsistent naming of the same place defeats the purpose. A location need not
   recur to deserve its own name; name it precisely either way.
9. MINIMIZE SCENE COUNT: each scene here becomes a separate rendered video clip
   (often several, one per camera shot) further down the pipeline — more scenes
   directly means more render time and cost. Merge consecutive beats that share
   the same `location` and continuous, uninterrupted time into ONE scene rather
   than splitting them across several, as long as the combined action can still
   read clearly through continuous coverage. Only start a NEW scene when the
   location changes, there's a real time jump (a scene break, not just the next
   sentence), or the dramatic purpose genuinely shifts — not for every new beat,
   line of dialogue, or minor action within an otherwise continuous moment. This
   does not license dropping or compressing away real events (rules 1-2 above
   still apply in full) — it only means telling everything that happens in one
   place at one time as a single scene instead of several redundant ones.
10. `props` lists notable PHYSICAL OBJECTS explicitly present or mentioned in
    the source text for this scene (e.g. "a brass diving bell", "a tarnished
    pocket watch", "an unopened letter" — generic illustrations of the level
    of specificity wanted, not suggestions to include) — grounded the same
    way `characters`
    is: only objects the source text actually mentions, never invented set
    dressing. Empty list if the source names nothing worth rendering. A prop
    that's a fixed, recurring part of the location itself (always there,
    regardless of scene) vs. one a character carries or that's specific to
    this scene's action are BOTH valid — don't filter either out; a later
    stage decides which is which.

Respond with JSON in exactly this shape (no extra keys, no commentary):
{{
  "scenes": [
    {{
      "number": 1,
      "slugline": "INT./EXT. LOCATION - DAY/NIGHT",
      "location": "plain name of the physical setting, identical across every scene set there",
      "source_line": "short verbatim phrase from the source text that this scene covers",
      "summary": "one or two sentences of what actually happens in the source",
      "characters": ["EXACT NAME as in source, or the matching CANONICAL name below", "..."],
      "purpose": "why this scene exists dramatically",
      "props": ["notable physical object explicitly present or mentioned in the source for this scene", "..."]
    }}
  ]
}}
{character_names_block}
SOURCE MATERIAL — primary fidelity anchor (title: {title}):
\"\"\"
{text}
\"\"\"

STRUCTURAL BEATS (secondary scaffold — ordering/emphasis only, not a replacement
for what the source actually says):
{beats}

Before you respond, re-check against the source material above (long source
text pushes early rules out of recent context — re-verify against what you
just read, not just what you remember from the rules list):
- Every `source_line` is an actual short verbatim quote FROM THE SOURCE TEXT above,
  not paraphrased and not from the structural beats.
- Every name in `characters` matches the CANONICAL CHARACTER NAMES list above
  exactly, if one was given.
- Every `location` string is byte-identical across every scene set in that
  same place.
- No two adjacent scenes share the same `location` AND continuous time without
  a real reason they're split (MINIMIZE SCENE COUNT, rule 9) — if you find one,
  merge them before responding.
- Every entry in `props` is an object the source material above actually names
  — not one you inferred would look good on screen.
"""


def _canonical_names_block(characters: dict | None) -> str:
    """Prompt block listing characters.json's settled names, when available —
    without it, a character only ever described in prose (never given a
    proper name in the source, e.g. "a beautiful young woman") gets
    independently re-derived by both the `characters` agent and this one,
    with no shared anchor forcing them to agree (unlike `location`, which
    already has this exact treatment — see the module docstring). Degrades
    to an empty string when no characters are available (e.g. a standalone
    `stage scenes` run with no characters.json checkpoint yet), which is a
    strict no-op — the surrounding PROMPT text/rule 5 already fall back to
    "use the exact name from the source text" in that case."""
    names = [c.get("name", "") for c in (characters or {}).get("characters", []) if c.get("name")]
    if not names:
        return ""
    bullet = "\n".join(f"- {n}" for n in names)
    return (
        "\nCANONICAL CHARACTER NAMES — every later stage (casting, storyboard, "
        "video) keys off these exact strings. When populating a scene's "
        '"characters" list, use one of these exact names for anyone who '
        'appears — never a shortened, rephrased, or differently-capitalized '
        'variant (e.g. if the canonical name is "Young Woman", do NOT write '
        '"Woman", "The Woman", or "young woman"). If someone appears who '
        "isn't in this list, name them using the exact wording from the "
        "source text instead, as usual.\n"
        f"{bullet}\n"
    )


def _structure_alignment_note(existing: dict | None, revise_keys: set | None) -> str:
    """Scoped-revision-only prompt block: states the EXISTING scene count and
    exactly which scene number(s) this round is scoped to, so the model
    aligns to the previous run's structure by default rather than silently
    re-segmenting the whole story to a different count. Additions/removals
    remain possible — `revision_merge.merge_by_key` only actually accepts a
    new number if it's in `revise_keys`, and deletion is handled entirely by
    the caller (`cli._strip_removed_scenes`) — but per the operator's own
    framing, that should only ever happen as an explicit, intended exception,
    never as an incidental side effect of the model re-segmenting freely.
    Empty (no-op) for a fresh, non-scoped run."""
    if revise_keys is None or not existing:
        return ""
    count = len(existing.get("scenes", []) or [])
    if not count:
        return ""
    return (
        f"\nSTRUCTURE ALIGNMENT — SCOPED REVISION: the story currently has "
        f"{count} scene(s). This round only revises scene number(s) "
        f"{sorted(revise_keys, key=str)} — every other scene must stay "
        f"exactly as it was (same count, same numbering, same order). Do NOT "
        f"add or remove a scene as a side effect of re-segmenting the story; "
        f"only introduce a new scene number or drop an existing one if the "
        f"revision note above explicitly calls for it as a deliberate "
        f"exception, not as an incidental recount.\n"
    )


def _revision_reminder_note(prior_scene_count: int | None) -> str:
    """Distinct from `_structure_alignment_note` above: that one only fires
    for a SCOPED regen (`existing`+`revise_keys` both given), where the
    scene count is enforced to stay stable. This one fires for a DRASTIC,
    fully UNSCOPED regen following a revision (e.g. `cli._revise_source`'s
    fallback when a source-text edit is judged to genuinely imply an
    added/removed scene) — where the count IS expected to legitimately
    change, so `_structure_alignment_note`'s "keep the same count" framing
    would be actively wrong here.

    But "the count may change" is not the same as "fragment freely" — rule
    9 (MINIMIZE SCENE COUNT) already applies to every segmentation, fresh or
    not, yet a full regen triggered by a text EDIT carries a specific risk
    the generic rule doesn't call out: the edited prose's own paragraph
    breaks can bias the model toward treating each rewritten beat as its
    own scene, even when location and time both stay continuous — exactly
    the failure this note exists to head off. Empty (no-op) for a genuinely
    fresh, non-revision segmentation (`prior_scene_count` is only ever
    passed by the reel.cli revise flow, never a first-time pipeline run)."""
    if not prior_scene_count:
        return ""
    return (
        f"\nNOTE: this is a REVISION regenerating the scene breakdown for an "
        f"edited story, not a first-time segmentation — the previous version "
        f"told this story in {prior_scene_count} scene(s). MINIMIZE SCENE "
        f"COUNT (rule 9) still applies IN FULL here: new or rewritten prose "
        f"is not itself a reason for more scenes, and a paragraph break in "
        f"the edited text is not automatically a scene break. Only introduce "
        f"a new scene where the edit genuinely changes location, causes a "
        f"real time jump, or shifts the dramatic purpose — merge everything "
        f"else exactly as rule 9 already requires, the same as you would for "
        f"a fresh segmentation.\n"
    )


def _reconcile_character_names(scenes: list[dict], characters: dict | None) -> list[dict]:
    """Deterministic safety net on top of the prompt-level instruction above:
    prompt instructions alone aren't always reliably followed (this codebase
    has hit that class of drift before — see PROGRESS.md's session log), so
    this catches the common case where a scene names someone with a string
    that isn't an exact canonical name but whose words are a strict subset
    of exactly ONE canonical name's words (case-insensitive) — e.g. "Woman"
    vs. the canonical "Young Woman" — and rewrites it to the canonical form.
    Ambiguous (matches more than one canonical name) or unrelated names are
    left untouched rather than guessed."""
    names = [c.get("name", "") for c in (characters or {}).get("characters", []) if c.get("name")]
    if not names:
        return scenes
    canonical_words = {n: set(n.lower().split()) for n in names}
    for sc in scenes:
        fixed = []
        for name in sc.get("characters") or []:
            if name in names:
                fixed.append(name)
                continue
            name_words = set(name.lower().split())
            candidates = [n for n, words in canonical_words.items()
                         if name_words and name_words <= words]
            fixed.append(candidates[0] if len(candidates) == 1 else name)
        sc["characters"] = fixed
    return scenes


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


def _attach_source_excerpts(scenes: list[dict], source_text: str) -> list[dict]:
    """Deterministically capture the actual PORTION of the story each scene
    covers — not just the short `source_line` anchor (5-15 words, only used
    for hallucination-checking and chunk-matching) but the real span of prose
    between this scene's anchor and the next one's. No LLM involved: it's a
    straight substring extraction from `source_line` positions `_validate`
    has already confirmed exist in the source text, so there's no
    hallucination risk the way there would be if the model were asked to
    quote a long passage verbatim.

    Scenes are ordered by where their `source_line` actually occurs in the
    text (not by their `number`, in case the model's numbering and the
    source's real order ever disagree) so each excerpt is a genuine
    contiguous, non-overlapping partition of the story — the last scene's
    excerpt runs to the end of the text. Known limitation, shared with
    `_map_chunks`'s identical `source_line`-matching approach: if a phrase
    genuinely repeats verbatim elsewhere in the story, `str.find` locates
    its FIRST occurrence, which may not be the one this scene actually
    covers — an inherent limit of anchoring on short quoted text, not
    something this function can resolve on its own.

    `word_count` rides along as a cheap, directly useful metadatum for
    downstream duration/pacing estimation (`reel.duration_budget`) — a real
    signal from the actual source material, not just storyboard.py's
    per-shot-type heuristic.

    A scene whose `source_line` can't be located (shouldn't happen after
    `_validate` already dropped those) gets an empty excerpt rather than
    raising."""
    norm_source = re.sub(r"\s+", " ", source_text)
    lower_source = norm_source.lower()

    positioned = []
    for sc in scenes:
        sl = re.sub(r"\s+", " ", (sc.get("source_line") or "").strip())
        pos = lower_source.find(sl.lower()) if sl else -1
        positioned.append((pos if pos >= 0 else None, sc))

    located = sorted((p for p in positioned if p[0] is not None), key=lambda p: p[0])
    for i, (pos, sc) in enumerate(located):
        end = located[i + 1][0] if i + 1 < len(located) else len(norm_source)
        excerpt = norm_source[pos:end].strip()
        sc["source_excerpt"] = excerpt
        sc["word_count"] = len(excerpt.split())
    for pos, sc in positioned:
        if pos is None:
            sc.setdefault("source_excerpt", "")
            sc.setdefault("word_count", 0)

    return scenes


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
    target: str = "as few scenes as the story can be told in — often 3-6 for a short story",
    profile: str | None = None,
    feedback: str | None = None,
    existing: dict | None = None,
    revise_keys: set | None = None,
    characters: dict | None = None,
    prior_scene_count: int | None = None,
) -> dict:
    """`existing` + `revise_keys` (a set of scene `number`s) support a scoped
    revision: the model still sees the FULL source text and re-segments the
    whole story (scene boundaries need full-story awareness to stay
    consistent), but the caller only trusts its output for the numbers in
    `revise_keys` — every other scene number is spliced back in
    byte-identical from `existing["scenes"]`. A genuinely NEW scene number
    (not present in `existing["scenes"]` at all) is appended by
    `merge_by_key`, then the merged list is re-sorted by `number` below
    so an inserted scene lands in its correct narrative position rather
    than always at the end of the array — several downstream consumers
    (e.g. `storyboard._scene_bundles`) iterate `scenes["scenes"]` in LIST
    order, not re-sorted by number themselves. Scene DELETION is handled
    by the caller (`cli._revise_one`'s "scenes" branch + `cli.
    _strip_removed_scenes`), not here — this function only ever adds to or
    replaces `existing["scenes"]`, consistent with `revision_merge.
    merge_by_key` never deleting a key on its own.

    `characters` (optional, from characters.json — the `characters` stage
    already runs before `scenes` in the pipeline, see pipeline.py's "2/10
    structure ‖ characters" then "3/10 scenes" ordering): when given, its
    settled names are fed into the prompt as the canonical names every scene
    must reuse (see `_canonical_names_block`), and `_reconcile_character_names`
    deterministically corrects the common near-miss (e.g. "Woman" vs the
    canonical "Young Woman") the LLM sometimes still produces despite that
    instruction — see both functions' docstrings for why this needed both a
    prompt fix AND a deterministic fallback.

    `prior_scene_count` (only ever passed by `reel.cli`'s `revise` flow, for
    a DRASTIC — fully unscoped — regen following a source-text edit judged
    to genuinely imply an added/removed scene) feeds `_revision_reminder_note`
    instead of `_structure_alignment_note`: the count is expected to change
    here, but rule 9's discipline still fully applies — see that function's
    docstring for the specific risk (edited prose's paragraph breaks biasing
    the model toward over-fragmenting) it exists to head off. Mutually
    exclusive with the scoped `existing`+`revise_keys` case in practice — a
    call is either scoped (existing+revise_keys) or a fresh/drastic regen
    (prior_scene_count), never both."""
    profile = profile or llm.agent_profile("scenes")
    beats = json.dumps(structure.get("three_act", {}), ensure_ascii=False, indent=2)
    source_text = source["text"][:MAX_CHARS]
    structure_note = (_structure_alignment_note(existing, revise_keys)
                      or _revision_reminder_note(prior_scene_count))
    prompt = llm.with_feedback(
        PROMPT.format(
            target=target,
            beats=beats,
            title=source["title"],
            text=source_text,
            character_names_block=_canonical_names_block(characters),
            structure_note=structure_note,
        ),
        feedback,
    )
    raw = llm.generate(prompt, profile=profile, system=SYSTEM, as_json=True)
    result = llm.safe_json(raw)
    scenes = result.get("scenes") or []
    scenes, dropped = _validate(scenes, source_text)
    scenes = _attach_source_excerpts(scenes, source_text)
    scenes = _map_chunks(scenes, source)
    scenes = _reconcile_character_names(scenes, characters)
    if revise_keys is not None and existing:
        scenes = merge_by_key(existing.get("scenes", []), scenes,
                              lambda s: s.get("number"), revise_keys)
        # A genuinely new scene number is appended by merge_by_key — re-sort
        # so it lands in its correct narrative position, not always last.
        # A single numeric key (not a tuple) avoids None-vs-None comparison
        # errors if more than one entry somehow lacks a number.
        scenes.sort(key=lambda s: s.get("number")
                    if isinstance(s.get("number"), int) else float("inf"))
    result["scenes"] = scenes
    result["dropped_scenes"] = dropped
    return result
