"""The one shared merge primitive every scoped-revision agent call uses.

Every per-scene/per-name agent (`casting.py`, `scenes.py`, `soundscape.py`,
`visuals.py`, `cinematography.py`, `screenplay.py`, `storyboard.py`,
`characters.py`) accepts an `existing` artifact + a `revise_keys` set: it
still gives the LLM full-story context for coherence, but the CALLER only
trusts the LLM's output for the targeted keys — everything else is spliced
back in byte-identical from `existing`. That splice logic is identical across
every one of those call sites, so it lives here once rather than being
reimplemented eight times and silently drifting.
"""
from __future__ import annotations


def merge_by_key(existing_list: list[dict], new_partial_list: list[dict],
                 key_fn, keys_to_replace) -> list[dict]:
    """Splice `new_partial_list` entries (keyed by `key_fn`) into
    `existing_list` for exactly `keys_to_replace`; every other key is carried
    over from `existing_list` UNCHANGED (same dict object, so re-serializing
    is guaranteed byte-identical), preserving `existing_list`'s order. A
    requested key the model didn't actually return a replacement for keeps
    its `existing` value (a loud warning, not a silent drop — a local model
    occasionally omits an item; better to preserve than lose data). Keys in
    `new_partial_list` that aren't in `keys_to_replace` are ignored (defends
    against a chatty/over-eager LLM call that revised more than asked).
    Genuinely new keys (not present in `existing_list` at all) are appended,
    in the order the model returned them, after every existing entry."""
    keys_to_replace = set(keys_to_replace)
    by_key_new = {key_fn(e): e for e in new_partial_list}
    out: list[dict] = []
    seen = set()
    for e in existing_list:
        k = key_fn(e)
        seen.add(k)
        if k in keys_to_replace:
            if k in by_key_new:
                out.append(by_key_new[k])
            else:
                print(f"[revision] ⚠ key {k!r} was in revise_keys but the "
                     "model didn't return it — keeping the existing entry", flush=True)
                out.append(e)
        else:
            out.append(e)
    for e in new_partial_list:
        k = key_fn(e)
        if k not in seen:
            out.append(e)
    return out
