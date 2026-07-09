"""Deterministic diffing between two versions of a stage's JSON artifact.

No LLM, no dependency on any other `reel` module. Used by the revision flow
(`reel/cli.py`'s `revise` command) to figure out exactly which scene numbers /
character-or-location names / panel numbers a hand-edit actually touched, so
only those need to flow into a downstream re-run instead of regenerating
everything.

Every artifact's revisable content is a JSON-object list keyed by some stable
field (scene `number`, `scene_number`, or casting/character `name`) — that key
is the only thing treated as a stable identity across a revision. Positions
that are purely sequential (scene numbering, panel numbering) are ASSUMED
stable across a revision in v1 — no scene insertion/deletion/reordering is
supported; a `removed` key on an artifact that disallows removal is reported
as `drastic`, and the caller should fall back to a full downstream re-run
rather than attempt a scoped one.
"""
from __future__ import annotations

import json
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

    `allow_remove=False` (the default, and the v1 scope assumption for every
    scene-keyed artifact) means any key present in `old` but missing from
    `new` sets `drastic=True` — a deletion/reorder isn't a "this key changed"
    edit, it invalidates the positional assumptions the rest of the revision
    flow relies on. `allow_add` stays True for name-keyed artifacts
    (characters/casting) where a genuinely new name is the normal, expected
    case, not a drastic one.
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
    "scenes":         ("scenes",      lambda s: s.get("number"),        False, False, None),
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
    """Special case for a hand-edit of the raw ingested story text
    (`source.json`'s `"text"` field). Word-level diffing raw prose into
    "which scene's source_line moved" is a genuinely hard problem (every
    downstream per-scene agent trusts `chunk_indices`/`source_line` offsets
    that a text edit can shift) — out of scope for v1. Always reports drastic
    when the text differs, so the caller falls back to a full downstream
    regen rather than attempting anything fine-grained."""
    changed = (old_source.get("text", "") != new_source.get("text", ""))
    return {
        "changed": changed,
        "drastic": changed,
        "reason": "source text changed — chunk_indices/source_line offsets throughout "
                  "the pipeline are no longer trustworthy; falling back to a full "
                  "downstream regen" if changed else "",
    }
