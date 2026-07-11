"""Deterministic diffing between two versions of a stage's JSON artifact.

No LLM, no dependency on any other `reel` module. Used by the revision flow
(`reel/cli.py`'s `revise` command) to figure out exactly which scene numbers /
character-or-location names / panel numbers a hand-edit actually touched, so
only those need to flow into a downstream re-run instead of regenerating
everything.

Every artifact's revisable content is a JSON-object list keyed by some stable
field (scene `number`, `scene_number`, or casting/character `name`) — that key
is the only thing treated as a stable identity across a revision. A `removed`
key on an artifact that disallows removal (`allow_remove=False`) is reported
as `drastic`, and the caller should fall back to a full downstream re-run
rather than attempt a scoped one.

`scenes.json` itself (the SOURCE of truth for the scene list) allows both
add and remove directly — `cli._revise_one`'s "scenes" branch propagates an
addition (a new scene number) through the normal `revise_keys` scoped-
regen path, and a removal by stripping the deleted scene number out of
every OTHER scene-keyed artifact (`cli._strip_removed_scenes`, reusing
`fidelity.strip_orphan_scenes` — the exact same primitive `pipeline.
run_group` already uses to self-heal scene-structure alignment). Every
OTHER scene-keyed artifact (soundscape/visuals/cinematography/screenplay/
storyboard) still disallows add/remove for ITS OWN diff — they're not
independently edited to add/remove scenes, they're kept in sync with
scenes.json's actual set by that same propagation, so a direct hand-edit to
one of them that doesn't match scenes.json's current scene set remains
`drastic` (unrelated to the scenes.json-driven add/remove path above)."""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field


@dataclass
class KeyDiff:
    added: list = field(default_factory=list)      # keys present only in `new`
    removed: list = field(default_factory=list)     # keys present only in `old`
    changed: list = field(default_factory=list)     # keys in both, value differs
    unchanged: list = field(default_factory=list)   # keys in both, value identical
    drastic: bool = False
    drastic_reason: str = ""


def _deep_eq(a, b) -> bool:
    """Order-insensitive structural equality — tolerant of dict key reordering
    a hand-edit through $EDITOR can introduce without the operator intending
    any actual content change."""
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == \
        json.dumps(b, sort_keys=True, ensure_ascii=False)


def diff_keyed_list(old_list: list[dict], new_list: list[dict], key_fn,
                     *, allow_add: bool = True, allow_remove: bool = False) -> KeyDiff:
    """Generic keyed diff over two lists of dicts.

    `allow_remove=False` (the default for every scene-keyed artifact EXCEPT
    `scenes` itself — see `ARTIFACT_SHAPES`) means any key present in `old`
    but missing from `new` sets `drastic=True` for THIS artifact's own diff —
    the caller (`cli._revise_one`) instead propagates a `scenes.json`
    deletion by directly stripping the orphaned key from these artifacts
    (`cli._strip_removed_scenes`), not by editing them here. `allow_add`
    stays True for name-keyed artifacts (characters/casting) where a
    genuinely new name is the normal, expected case, not a drastic one, and
    for `scenes` itself, where a genuinely new scene number is likewise
    normal (see `revision_merge.merge_by_key`'s append-new-keys behavior;
    `scenes.segment_scenes` re-sorts the merged list by `number` afterward
    so an appended scene lands in its correct narrative position rather than
    always at the end).
    """
    old_by_key = {key_fn(e): e for e in old_list}
    new_by_key = {key_fn(e): e for e in new_list}
    old_keys, new_keys = set(old_by_key), set(new_by_key)

    removed = sorted(old_keys - new_keys, key=str)
    added = sorted(new_keys - old_keys, key=str)
    changed, unchanged = [], []
    for k in sorted(old_keys & new_keys, key=str):
        if _deep_eq(old_by_key[k], new_by_key[k]):
            unchanged.append(k)
        else:
            changed.append(k)

    drastic, reason = False, ""
    if removed and not allow_remove:
        drastic = True
        reason = f"key(s) removed: {removed!r} — deletion/reordering isn't supported in v1"
    elif added and not allow_add:
        drastic = True
        reason = f"key(s) added: {added!r} — insertion isn't supported in v1"

    return KeyDiff(added=added, removed=removed, changed=changed,
                   unchanged=unchanged, drastic=drastic, drastic_reason=reason)


# (list_field, key_fn, allow_add, allow_remove, nested)
# `nested` is (nested_list_field, nested_key_fn) for artifacts with a keyed
# list one level down (storyboard panels inside scenes, cinematography shots
# inside scenes) — None for flat keyed lists.
ARTIFACT_SHAPES: dict[str, tuple] = {
    "scenes":         ("scenes",      lambda s: s.get("number"),        True,  True,  None),
    "characters":     ("characters",  lambda c: c.get("name"),          True,  True,  None),
    "casting":        ("casting",     lambda c: c.get("name"),          True,  True,  None),
    "soundscape":     ("soundscapes", lambda s: s.get("scene_number"),  False, False, None),
    "visuals":        ("scenes",      lambda s: s.get("scene_number"),  False, False, None),
    "cinematography": ("scenes",      lambda s: s.get("scene_number"),  False, False,
                        ("shots", lambda s: s.get("shot_number"))),
    "screenplay":     ("scenes",      lambda s: s.get("scene_number"),  False, False, None),
    "storyboard":     ("storyboard",  lambda s: s.get("scene_number"),  False, False,
                        ("panels", lambda p: p.get("panel"))),
}

# Which ARTIFACT_SHAPES entries are keyed by scene_number vs. by name — a
# single shared definition so the revision CLI's cross-type key translation
# (cli.py's _translate_revise_keys) and the fidelity agent's scene-alignment
# check (fidelity.check_scene_alignment) don't each maintain their own copy.
SCENE_KEYED_ARTIFACTS = {"scenes", "soundscape", "visuals", "cinematography", "screenplay", "storyboard"}
NAME_KEYED_ARTIFACTS = {"characters", "casting"}

# Whole-file artifacts with no natural keyed-list shape — a flat dict of
# fields that every downstream agent's prompt draws on directly (logline,
# genre, palette, ...). An edit here always counts as drastic (full
# downstream regen), never fine-grained-diffed.
WHOLE_FILE_ARTIFACTS = {"structure", "moodboard", "source"}


def diff_artifact(name: str, old: dict, new: dict) -> KeyDiff:
    """Diff the top-level keyed list of artifact `name` (scene number /
    character-or-location name / etc, per ARTIFACT_SHAPES). Raises KeyError
    for an artifact with no shape entry — check `WHOLE_FILE_ARTIFACTS` first;
    those have no keyed-list structure to diff at all."""
    if name in WHOLE_FILE_ARTIFACTS:
        raise KeyError(f"{name!r} is a whole-file artifact (see WHOLE_FILE_ARTIFACTS) "
                       "— it has no keyed list to diff; treat any edit as drastic")
    if name not in ARTIFACT_SHAPES:
        raise KeyError(f"no diff shape registered for artifact {name!r}")
    list_field, key_fn, allow_add, allow_remove, _nested = ARTIFACT_SHAPES[name]
    old_list = old.get(list_field, []) or []
    new_list = new.get(list_field, []) or []
    return diff_keyed_list(old_list, new_list, key_fn,
                           allow_add=allow_add, allow_remove=allow_remove)


def diff_nested(name: str, old: dict, new: dict) -> dict:
    """For artifacts with a nested keyed list (storyboard panels inside
    scenes, cinematography shots inside scenes): returns
    {outer_key: KeyDiff-over-inner} for every outer key present in BOTH old
    and new. Outer keys that were themselves added/removed are already
    reported by `diff_artifact` — this only inspects survivors, since a panel
    diff is meaningless for a scene that doesn't exist in both versions.
    Returns {} for an artifact with no nested shape."""
    if name in WHOLE_FILE_ARTIFACTS or name not in ARTIFACT_SHAPES:
        return {}
    list_field, outer_key_fn, _allow_add, _allow_remove, nested = ARTIFACT_SHAPES[name]
    if nested is None:
        return {}
    nested_field, nested_key_fn = nested
    old_by_key = {outer_key_fn(e): e for e in (old.get(list_field, []) or [])}
    new_by_key = {outer_key_fn(e): e for e in (new.get(list_field, []) or [])}
    result = {}
    for k in old_by_key.keys() & new_by_key.keys():
        old_inner = old_by_key[k].get(nested_field, []) or []
        new_inner = new_by_key[k].get(nested_field, []) or []
        result[k] = diff_keyed_list(old_inner, new_inner, nested_key_fn,
                                    allow_add=True, allow_remove=False)
    return result


def diff_source_text(old_source: dict, new_source: dict) -> dict:
    """Cheap "did the raw story text actually change at all" gate for a
    hand-edit of `source.json`'s `"text"` field — used by `cli._revise_source`
    before spending any effort (deterministic or LLM) on figuring out WHERE
    it changed. Deliberately says nothing about scope: `unified_source_diff`/
    `candidate_changed_scenes` below (deterministic) plus
    `reel.agents.revision.identify_source_text_changes` (LLM-confirmed) are
    what narrow a source-text edit down to the specific scene numbers it
    actually affects, rather than treating every edit as drastic."""
    changed = (old_source.get("text", "") != new_source.get("text", ""))
    return {
        "changed": changed,
        "drastic": changed,
        "reason": "source text changed" if changed else "",
    }


def _split_paragraphs(text: str) -> list[str]:
    """Split raw story text into paragraph-like units for diffing:
    blank-line-separated paragraphs when present, else single lines, else
    sentence-ish chunks as a last resort — so even a single-block story
    (no line breaks at all) still yields a meaningful diff granularity
    instead of one giant "the whole thing changed" unit."""
    normalized = text.replace("\r\n", "\n")
    paras = [p for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if len(paras) >= 2:
        return paras
    lines = [ln for ln in normalized.split("\n") if ln.strip()]
    if len(lines) >= 2:
        return lines
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", normalized.strip()) if s.strip()]
    return sentences or ([normalized] if normalized.strip() else [])


def unified_source_diff(old_text: str, new_text: str, *, context: int = 1) -> str:
    """A compact, paragraph-level unified diff between two versions of the
    raw story text — deliberately NOT "send the whole story twice", which
    wastes context and gives a model no structural hint about where to
    even look. `-`-prefixed lines were removed, `+`-prefixed were added,
    unprefixed lines are unchanged context (kept `context` paragraphs deep
    on either side of a change so the model can tell what surrounds it).
    File-diff header lines (`---`/`+++`/`@@`) are stripped since they're
    meaningless for prose with no real filenames or line numbers."""
    old_paras = _split_paragraphs(old_text)
    new_paras = _split_paragraphs(new_text)
    raw = difflib.unified_diff(old_paras, new_paras, lineterm="", n=context)
    body = [ln for ln in raw if not ln.startswith(("---", "+++", "@@"))]
    return "\n".join(body).strip()


def candidate_changed_scenes(old_text: str, new_text: str, scenes: dict) -> list[int]:
    """Deterministic (no LLM) first pass: a scene whose stored
    `source_excerpt` (falling back to `source_line` for a checkpoint that
    predates that field) is no longer found verbatim — whitespace-
    normalized — anywhere in `new_text` is a CANDIDATE for being affected
    by the edit. Not a certainty either way: the surrounding prose may have
    only been re-wrapped or trivially reworded (a false positive the LLM
    confirmation pass in `reel.agents.revision.identify_source_text_changes`
    can rule back out), and a genuinely different scene's short `source_line`
    could in principle still coincidentally match elsewhere (unlikely for a
    real quote, and a pre-existing limitation this function shares with
    `reel.agents.scenes._map_chunks`'s identical matching approach — not
    solved here). Returns scene numbers sorted ascending."""
    norm_new = re.sub(r"\s+", " ", new_text).lower()
    candidates = []
    for sc in scenes.get("scenes", []):
        anchor = (sc.get("source_excerpt") or sc.get("source_line") or "").strip()
        if not anchor:
            continue
        norm_anchor = re.sub(r"\s+", " ", anchor).lower()
        if norm_anchor and norm_anchor not in norm_new:
            num = sc.get("number")
            if isinstance(num, int):
                candidates.append(num)
    return sorted(candidates)
