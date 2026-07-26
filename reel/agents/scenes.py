"""Scene agent: segment the story into a numbered scene list.

The source text is the ONLY authority — every scene must correspond to an actual
event in the source. Structural beats are a secondary ordering scaffold only.

This is the single most consequential stage for how completely the film
captures the story: every scene becomes a separate rendered video clip (or
several, one per shot) further down the pipeline, so how the story gets
segmented here directly determines what a viewer actually sees. The prompt
steers toward a DIRECTOR'S-EYE segmentation (see STRICT RULE 9, "CAPTURE THE
STORY FULLY", below) — give a distinct scene to every beat that carries its
own dramatic or visual weight, rather than compressing the story to hit a
lower scene count. This deliberately replaces an earlier version of this
rule that steered toward the FEWEST scenes as a cost-minimization bias —
removed per direct instruction, since it was under-serving the story. The
existing "never invent scenes to hit a count" / "no unnecessary repeats"
rules still prevent the opposite failure mode (inventing or duplicating
scenes that aren't really there).

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

Each scene finally carries `emotional_beat` + `expression` (rule 11,
DIRECTOR'S INTERPRETIVE EXPANSION) — the director's read of what the moment
must make the audience FEEL and how that reads visibly on a face/body. This
is the ONE place this stage is licensed to go beyond the literal text: an
emotional or expressional moment the prose only implies (the held look, the
decision not to speak, the beat where grief lands) is treated as filmable
material, and may even earn its own scene, where an implied EVENT never
could. That boundary — interpretation of HOW a moment is felt and performed,
never new FACTS (no invented events, characters, locations, props, lines, or
outcomes; rules 1-2 still stand unchanged) — is the same line
`veo_prompt.panel_action`'s "director's freedom" layer already draws at
render time, applied here at the point the scene list itself is built. Two
deliberate consequences: rule 4 keeps `summary` clean and source-checkable
so the interpretive layer stays quarantined in its own named fields (fidelity
grading still has an unembellished record to judge against), and rule 7 was
tightened so "two beats from one passage" can't degrade into two scenes
restating one moment. Both fields are propagated to the stages that can
actually act on them — `cinematography` (into each shot's
`emotional_function`, which reaches the rendered Veo prompt via
`storyboard`'s panel `emotional_note`), `visuals`, `soundscape`, and
`screenplay` — rather than stopping in scenes.json, which is the failure
mode `visuals.key_props` and panel `emotional_note` both previously had (see
PROGRESS.md's 2026-07-10 and 2026-07-23 entries).
"""
from __future__ import annotations

import json
import re

from .. import llm
from ..revision_merge import merge_by_key
from .ingest import chunk_text, CHUNK_SIZE

SYSTEM = (
    "You are one of the best screenwriters working today, breaking a story "
    "into filmable scenes. Each scene happens in one location and continuous "
    "time. You always respond with valid JSON and nothing else."
)

PROMPT = """\
Break the story below into filmable scenes the way a great director breaks
down a shooting script: {target}. Give every beat that carries its own
dramatic, visual, or EMOTIONAL weight its own scene, so the film actually
does the story justice (see STRICT RULES 9 and 11 below). Read the story for
the emotional and expressional life inside each line — including what the
prose implies but never spells out — and give that life both its own scenes
and its own explicit direction (RULE 11). The only limits on scene count are
real ones: never invent a scene the source doesn't support, and never split a
single continuous beat into two.
{structure_note}
STRICT RULES:
1. SOURCE TEXT IS THE ONLY AUTHORITY FOR WHAT HAPPENS. Every scene must
   correspond to an actual event, location, or moment present in the source
   text below — including a moment the source carries implicitly, in what a
   character does, feels, withholds, or leaves unsaid (rule 11). What the
   source does not support, you may not add.
2. No invented characters, plot points, locations, props, or outcomes. If the
   source doesn't describe it, it cannot appear in the scene list. A scene that
   stages an EVENT the source never has is an invented scene; a scene that
   surfaces an emotional or expressional moment the source genuinely implies is
   not (rule 11) — that distinction is the whole line between direction and
   invention, so hold it precisely.
3. `source_line` is mandatory: copy a SHORT verbatim phrase (5-15 words) from the
   source text that anchors this scene. If you cannot find a matching phrase, the
   scene does not belong in the list. When two scenes draw on the same sentence or
   passage (rule 11 makes that legitimate), each must still anchor on its own
   distinct phrase wherever the text offers one — don't repeat an identical
   `source_line` across scenes.
4. `summary` must describe only what the source text says — no embellishment.
   Keep your interpretive reading OUT of `summary` and IN `emotional_beat` /
   `expression`, where it belongs: `summary` stays the plain, source-checkable
   record of what happens, so the two layers never get confused for each other.
5. Use the EXACT character names that appear in the source text — or, when a
   CANONICAL CHARACTER NAMES list is provided below, the exact matching name
   from THAT list (it already settled on one consistent name per character;
   never shorten, rephrase, or re-derive your own variant of it).
6. The structural beats below are a secondary ordering hint only. Where they conflict
   with the source text, the source text wins.
7. No unnecessary repeats: never split one event into two overlapping scenes, and
   never list two scenes whose `summary`/`source_line` cover substantially the same
   moment. Each scene must earn its place with something the others don't already
   cover. Two scenes drawn from the SAME passage are legitimate only when each
   turns on a genuinely different emotional or expressional beat (the dread
   before a door opens, then the recognition after it does) — restating one beat
   in different words is still a repeat, and a differently-worded
   `emotional_beat` alone never turns one moment into two scenes.
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
9. CAPTURE THE STORY FULLY — DIRECTOR'S EYE: think like a director breaking
   the story into a shot list, not like someone trying to keep the scene
   count down. Give a DISTINCT scene to every beat that carries its own
   dramatic or visual weight — a turning point, a shift in who's present or
   what they want, a change in emotional register, a meaningful pause or
   reaction — even when it shares a `location` and roughly continuous time
   with what comes before or after it. Only fold two moments into ONE scene
   when they are genuinely the SAME continuous beat with nothing
   dramatically distinct happening between them (e.g. a character simply
   crossing a room mid-conversation is not its own scene; a character
   deciding something and acting on it usually is). If a scene-count target
   is given above, treat it as a soft, secondary guide — never merge or
   compress distinct beats just to land inside it; what the story itself
   needs always wins over any count.
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
11. DIRECTOR'S INTERPRETIVE EXPANSION — THE EMOTIONAL LAYER. Read the source
    the way the best directors read a story: every line carries emotional and
    expressional life the prose states only in passing, or leaves entirely
    between the lines — the held look before an answer, the flinch of
    recognition, the moment someone decides not to speak, the beat where grief
    finally lands, the small physical tell that gives away what a character
    won't say. Surface that layer generously and precisely; it is what makes an
    adaptation feel directed rather than transcribed. Two things follow:
    (a) an implicit emotional or expressional MOMENT may earn its OWN scene
    even where the source never marks it as a separate event — if the prose
    supports a character feeling or showing it at that point in the story,
    it is filmable material;
    (b) EVERY scene carries `emotional_beat` and `expression`: what the moment
    must make the audience feel, and how that reads visibly on a face, in a
    body, in behaviour. Be elaborate and specific in both — a performable
    read, not a label ("the practiced stillness of someone who has already
    decided to lie, eyes steady a half-beat too long" — not "he is nervous").
    This is the one place in this stage where your own directorial reading of
    the story is actively wanted.
    THE LIMIT — this licenses interpretation of HOW a moment is felt and
    performed, never new FACTS. You still may not add an event, character,
    location, prop, spoken line, or outcome the source doesn't contain
    (rules 1-2 stand): an inferred EMOTION is grounded in the source, an
    inferred PLOT POINT is not.

DO NOT:
- invent an EVENT, character, location, prop, or outcome the source text
  doesn't actually contain (rules 1-2 above) — note that surfacing an
  emotional or expressional moment the source implies is direction, not
  invention (rule 11), and the two must not be confused in either direction
- write a generic, unperformable `emotional_beat`/`expression` ("she is sad",
  "it is tense") when the passage supports a specific read (rule 11b)
- let interpretation leak into `summary` instead of `emotional_beat`/
  `expression` (rule 4)
- merge two genuinely distinct dramatic or emotional beats into one scene just
  to keep the count down, or split one continuous beat into two just to
  inflate it
- leave `source_line` paraphrased instead of a real verbatim quote
- produce duplicate or non-sequential `number` values
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present for every scene, no extra top-level keys, no missing keys:
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
      "props": ["notable physical object explicitly present or mentioned in the source for this scene", "..."],
      "emotional_beat": "the director's read: what this moment must make the audience FEEL, specifically — the emotional turn inside it, including what the source only implies",
      "expression": "how that reads on screen: the specific faces, bodies, gestures, held silences, and physical tells that make it visible and performable"
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
- Every distinct dramatic beat in the source material above has its own
  scene (CAPTURE THE STORY FULLY, rule 9) — check for two different turning
  points or emotional shifts hiding inside one merged scene just because
  they share a location, and split them if you find one.
- Every distinct EMOTIONAL or EXPRESSIONAL moment the source material above
  supports — stated outright or carried in subtext — has been surfaced rather
  than passed over (DIRECTOR'S INTERPRETIVE EXPANSION, rule 11), and every
  scene's `emotional_beat`/`expression` is a specific, performable read of
  that scene rather than a generic label.
- Nothing in `emotional_beat`/`expression` asserts a new EVENT, spoken line,
  character, location, prop, or outcome the source material above doesn't
  contain — interpretation of how a moment feels and plays, never new facts
  (rule 11's LIMIT).
- Every entry in `props` is an object the source material above actually names
  — not one you inferred would look good on screen.
- The response is ONLY the JSON object above — no markdown fences, no
  commentary, no extra top-level keys, every scene has all 10 schema fields.
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

    Purely informational — states the prior count as context, nothing more.
    An earlier version of this note enforced "MINIMIZE SCENE COUNT" here
    specifically, on the theory that an edited passage's own paragraph
    breaks could bias the model toward over-fragmenting; now that rule 9 is
    "capture the story fully" rather than "minimize," that concern doesn't
    apply — a rewritten passage calling for more distinct beats than before
    is a correct outcome, not a failure mode to guard against. Empty (no-op)
    for a genuinely fresh, non-revision segmentation (`prior_scene_count` is
    only ever passed by the reel.cli revise flow, never a first-time
    pipeline run)."""
    if not prior_scene_count:
        return ""
    return (
        f"\nNOTE: this is a REVISION regenerating the scene breakdown for an "
        f"edited story, not a first-time segmentation — the previous version "
        f"told this story in {prior_scene_count} scene(s). That count is "
        f"informational only, not a target: segment the edited text the same "
        f"way you would a fresh story, rule 9 (CAPTURE THE STORY FULLY) in "
        f"full — a rewritten passage may now call for more distinct scenes "
        f"than before, or fewer, purely based on what the story now needs, "
        f"not on matching the old count.\n"
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
    positions = [p for p, _ in located]
    for i, (pos, sc) in enumerate(located):
        # The next STRICTLY GREATER position, not simply the next entry in the
        # list: rule 11 (DIRECTOR'S INTERPRETIVE EXPANSION) makes it legitimate
        # for two scenes to draw distinct emotional beats out of the SAME
        # passage, and two scenes whose `source_line` resolves to the same
        # offset would otherwise hand the earlier one an empty excerpt (and a
        # `word_count` of 0, silently breaking duration estimation for it).
        # Colliding scenes share that passage's span instead.
        nxt = next((p for p in positions[i + 1:] if p > pos), None)
        end = nxt if nxt is not None else len(norm_source)
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
    # Coverage-first and deliberately NUMBERLESS. This used to be overridden
    # by `duration_budget.suggest_scene_target`, which interpolated a literal
    # range ("roughly 2-4 scenes" at the 45s default) into the sentence below
    # — the most salient spot in the prompt — where it beat every coverage
    # rule that followed. Callers no longer pass `target` at all; see
    # `reel.duration_budget`'s note for why a runtime budget bounds rendering
    # rather than design.
    target: str = "as many scenes as the story's own beats call for, "
                  "never a number decided in advance and never for brevity",
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
    # Budget is per-PROFILE, not a global constant: on the hosted frontier
    # tier the local 12,000-char cap would truncate the story to roughly
    # its first 2,000 words while the model has room for a whole novel.
    source_text = source["text"][:llm.max_chars(profile)]
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
