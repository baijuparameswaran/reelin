"""The one shared merge primitive every scoped-revision agent call uses.

Every per-scene/per-name agent (`casting.py`, `scenes.py`, `soundscape.py`,
`visuals.py`, `cinematography.py`, `screenplay.py`, `storyboard.py`,
`characters.py`) accepts an `existing` artifact + a `revise_keys` set: it
still gives the LLM full-story context for coherence, but the CALLER only
trusts the LLM's output for the targeted keys — everything else is spliced
back in byte-identical from `existing`. That splice logic is identical across
every one of those call sites, so it lives here once rather than being
reimplemented eight times and silently drifting.

Two levels of splicing happen: `merge_by_key` decides WHICH ENTRIES (by
`key_fn`) are trusted from the model's response at all; `merge_fields` then
decides, WITHIN each of those trusted entries, which INDIVIDUAL FIELDS
actually changed — a targeted entry being "in scope" doesn't mean every
field on it should be blindly overwritten, since the model saw full-story
context and can (and does, in practice) reword fields that weren't the
actual point of the edit. Only a field whose value is genuinely different
from the existing one is accepted; every other field keeps its OLD value
verbatim, for the same `_content_hash`-stability reason whole-entry
preservation exists in the first place (an untouched field's re-render
should never spuriously invalidate a downstream hash).
"""
from __future__ import annotations

import json


def _values_equal(a, b) -> bool:
    """Structural equality, tolerant of dict-key reordering (a model's JSON
    field order isn't semantically meaningful) but NOT of list-element
    reordering (list order often IS semantically meaningful — dialogue
    order, shot order, scene order, ...). Same approach as
    `artifact_diff._deep_eq` (kept as a separate small implementation here,
    not imported, since this module is deliberately dependency-free — see
    the module docstring — and `artifact_diff.py` already has no reverse
    dependency on it)."""
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == \
        json.dumps(b, sort_keys=True, ensure_ascii=False)


def merge_fields(old_entry: dict, new_entry: dict, *, label: str = "") -> dict:
    """Field-by-field splice of `new_entry` onto `old_entry`: for every
    field the model's response actually supplies, accept the new value
    IF AND ONLY IF it's structurally different from the existing one;
    otherwise keep the OLD value verbatim (same object — re-serializing
    that field is guaranteed byte-identical). Applied even to an entry
    `merge_by_key` has already decided IS a scoped target — being in scope
    means the entry is CANDIDATE for changes, not that every field on it
    should be trusted wholesale; a field the model reworded without any
    actual meaning change (punctuation, phrasing, incidental rewording of
    a field that had nothing to do with the actual edit) shouldn't be
    allowed to drift.

    A field present in `new_entry` but absent from `old_entry` (a
    genuinely new field) is taken as-is — nothing old to compare against.
    A field present in `old_entry` but absent from `new_entry` (the model
    simply didn't echo it back in its response) keeps the OLD value — an
    omission isn't a deletion, the same "preserve, don't lose data"
    philosophy `merge_by_key` already applies when the model omits an
    entire targeted KEY, extended here to a single FIELD within one.

    Deliberately ONE level deep — compares each top-level field of the
    entry as a whole value (a list or nested dict field is compared (and,
    if different, replaced) in its entirety, not recursed into further).
    Going deeper would need per-artifact-type knowledge of which nested
    structures are independently meaningful (e.g. storyboard's `panels`,
    casting's `character`/`actor` blocks) that this generic primitive
    doesn't have — panel/shot-level granularity already has its own
    dedicated mechanism (`artifact_diff.diff_nested` + `cli.py`'s
    `panel_targets`) for the one case that needs it.

    `label` (optional) — when given, prints a one-line summary of which
    fields actually changed (or that none did) for this entry, purely for
    revise-flow transparency; every caller of `merge_by_key` only reaches
    this during a scoped revision (never a fresh pipeline run), so this
    never prints outside that context."""
    merged = dict(old_entry)
    changed: list[str] = []
    for k, new_v in new_entry.items():
        if k not in old_entry or not _values_equal(old_entry[k], new_v):
            merged[k] = new_v
            changed.append(k)
    if label:
        if changed:
            print(f"[revision]   {label!r}: field(s) actually changed: {sorted(changed)}",
                 flush=True)
        else:
            print(f"[revision]   {label!r}: no field differed from the model's response "
                 "— kept entirely as-is", flush=True)
    return merged


def merge_by_key(existing_list: list[dict], new_partial_list: list[dict],
                 key_fn, keys_to_replace) -> list[dict]:
    """Splice `new_partial_list` entries (keyed by `key_fn`) into
    `existing_list` for exactly `keys_to_replace`; every other key is carried
    over from `existing_list` UNCHANGED (same dict object, so re-serializing
    is guaranteed byte-identical), preserving `existing_list`'s order. A
    requested key the model didn't actually return a replacement for keeps
    its `existing` value (a loud warning, not a silent drop — a local model
    occasionally omits an item; better to preserve than lose data).

    `existing_list`'s own structure — its set of keys, in its own order — is
    the enforced baseline: ANY key in `new_partial_list` that isn't in
    `keys_to_replace` is ignored, including a genuinely NEW key (not present
    in `existing_list` at all). Addition is only ever an intentional
    EXCEPTION the caller opted into by including that key in
    `keys_to_replace` (e.g. `artifact_diff.diff_artifact`'s own `added` set,
    for a scenes.json hand-edit that genuinely inserted a scene) — never an
    incidental side effect of a scoped call giving the model full-story
    context and it deciding, unprompted, to invent or rename an extra entry.
    A new key that IS in `keys_to_replace` is appended, in the order the
    model returned it, after every existing entry (scene-numbered artifacts
    additionally re-sort afterward — see `scenes.segment_scenes` — since
    narrative position, not response order, is what matters there).
    Deletion is never done here at all, by design — this primitive only
    ever adds or replaces a key; see `cli._strip_removed_scenes` for the
    dedicated, explicit mechanism an operator-requested deletion goes
    through instead.

    A key that IS replaced doesn't take the model's entry wholesale — it's
    spliced through `merge_fields` (see that function's docstring) first,
    so only the fields that actually differ from the existing entry are
    accepted; every other field on that same entry keeps its old value."""
    keys_to_replace = set(keys_to_replace)
    by_key_new = {key_fn(e): e for e in new_partial_list}
    out: list[dict] = []
    seen = set()
    for e in existing_list:
        k = key_fn(e)
        seen.add(k)
        if k in keys_to_replace:
            if k in by_key_new:
                out.append(merge_fields(e, by_key_new[k], label=str(k)))
            else:
                print(f"[revision] ⚠ key {k!r} was in revise_keys but the "
                     "model didn't return it — keeping the existing entry", flush=True)
                out.append(e)
        else:
            out.append(e)
    for e in new_partial_list:
        k = key_fn(e)
        if k in seen:
            continue
        if k in keys_to_replace:
            out.append(e)
        else:
            print(f"[revision] ⚠ discarding unrequested new key {k!r} the model returned "
                 "— not in revise_keys, so not an intended addition", flush=True)
    return out
