"""Pure, deterministic panel-grouping logic for multi-segment Veo prompts.

Detects contiguous runs of storyboard panels WITHIN one scene that share the
same in-frame character set, so `pipeline.py` can render the whole run as a
SINGLE Veo call using Google's documented timestamp-bracketed multi-segment
technique (`[00:00-00:02] ... [00:02-00:04] ...`) instead of one call per
panel — cuts API calls/cost and lets a fixed Subject/Context be stated once
while Cinematography/Action vary shot to shot within one continuous
generation.

Deliberately scoped to WITHIN one scene only — never crosses a
`scene_number` boundary. `location` is a single field per scene (never
per-panel), so within one scene it's already guaranteed constant; nothing
here needs to check it. Merging across scene boundaries when consecutive
scenes share a location is a real, larger follow-on (would need to touch
`pipeline.py`'s per-scene hard-cut reset and `_stitch_scene`'s
one-video-per-scene structure) and is explicitly out of scope here — see
PROGRESS.md.
"""
from __future__ import annotations

from . import duration_budget


def char_set_changed(fr: dict, prev_char_key) -> bool:
    """True if THIS panel is a character "shot boundary" relative to
    `prev_char_key` — the immediately preceding panel's in-frame character
    SET (a `frozenset`, or `None` for a scene's first panel, meaning
    "nothing to compare against yet", which itself counts as a boundary).
    A panel with no listed characters at all is never a boundary (nothing
    to compare) — it's treated as a continuation of whatever came before.

    Relocated verbatim from `pipeline._char_set_changed` (this module is the
    single source of truth now — `pipeline.py` imports it back under that
    name so every existing call site keeps working unchanged) so
    `group_panels` below shares the exact same "same cast" predicate every
    other continuity/reference-image decision already uses, instead of a
    second, potentially-drifting reimplementation.

    Shared by `pipeline._resolve_panel_references` (decides whether to use
    multi-character `reference_images` instead of a single seed) AND by
    every caller that decides whether Veo's `continuity_mode: extend` is
    even eligible for this panel — extending the PREVIOUS clip only makes
    sense when this panel is showing the same subject(s) that clip was."""
    names = fr.get("characters_in_frame") or []
    if not names:
        return False
    return prev_char_key is None or frozenset(names) != prev_char_key


def _panel_duration(fr: dict) -> float:
    """This panel's own estimated screen time in seconds, falling back to
    the same planning-stage assumption `duration_budget.py` uses elsewhere
    when the panel's own `duration` field is missing/unparseable."""
    return (duration_budget.parse_duration_seconds(fr.get("duration"))
            or duration_budget.ASSUMED_SECONDS_PER_SHOT)


def group_panels(panels: list[dict], *, max_seconds: int = 8) -> list[list[dict]]:
    """Partition one scene's panel list into contiguous runs suitable for a
    single multi-segment Veo call: same in-frame cast (per `char_set_changed`)
    AND a summed duration that fits within `max_seconds` (Veo's own max valid
    clip length — see `i2v.VEO_MAX_DURATION`).

    A group of size 1 is the common/degenerate case (no same-cast run, or
    grouping produced nothing beneficial) — `pipeline.py` treats that
    identically to today's per-panel render, so this function returning all
    singletons for some/every scene is always safe.

    Hard-caps DURING accumulation, not after: before adding a panel to the
    open group, checks whether doing so would push the running duration sum
    over `max_seconds` — if so, closes the current group (even though the
    cast hasn't changed) and starts a new one with that panel. This avoids
    silently compressing several panels' authored screen time to fit a
    rounded-down bucket, which a "sum then round" approach would risk.

    Never crosses panel-list boundaries beyond what's given — callers must
    pass ONE scene's own panels (never a whole board flattened across
    scenes); see this module's docstring for why scene boundaries are out
    of scope here."""
    if not panels:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_seconds = 0.0
    prev_char_key = None
    for fr in panels:
        names = list(dict.fromkeys(fr.get("characters_in_frame") or []))
        this_seconds = _panel_duration(fr)
        is_boundary = char_set_changed(fr, prev_char_key)
        overflow = bool(current) and (current_seconds + this_seconds > max_seconds)
        if current and (is_boundary or overflow):
            groups.append(current)
            current = []
            current_seconds = 0.0
        current.append(fr)
        current_seconds += this_seconds
        # Propagate the char key forward exactly like
        # `pipeline._resolve_panel_references` does: a panel with no
        # characters rides along on whatever key is already open, rather
        # than resetting tracking to "no cast".
        if names:
            prev_char_key = frozenset(names)
    if current:
        groups.append(current)
    return groups


def segment_boundaries(raw_durations: list[float], target_total: float) -> list[tuple[float, float]]:
    """Proportionally rescale `raw_durations` (a group's per-panel estimated
    seconds) so cumulative segment boundaries sum EXACTLY to `target_total`
    (one of Veo's valid 4/6/8s values, or whatever `i2v.forced_group_duration`
    pins it to) — the multi-segment prompt's timestamps must always match
    the actual requested render duration, or the prompt lies about what Veo
    is being asked to produce.

    Guards against a segment rescaling to (near) zero length by flooring
    each at 10% of an even share and borrowing the difference from the
    longer segments, then rounds each boundary to the nearest 0.5s (the
    final boundary is always forced to exactly `target_total`, absorbing any
    rounding remainder)."""
    n = len(raw_durations)
    if n == 0:
        return []
    positive = [d for d in raw_durations if d > 0]
    total_raw = sum(positive) if positive else 0.0
    if total_raw <= 0:
        raw_durations = [1.0] * n
        total_raw = float(n)
    scale = target_total / total_raw
    scaled = [d * scale for d in raw_durations]

    floor = (target_total / n) * 0.1
    short = [max(0.0, floor - s) for s in scaled]
    deficit = sum(short)
    if deficit > 0:
        headroom = sum(s - floor for s in scaled if s > floor)
        if headroom > 0:
            scaled = [floor if s <= floor else s - deficit * (s - floor) / headroom
                     for s in scaled]
        else:
            scaled = [target_total / n] * n

    boundaries: list[tuple[float, float]] = []
    running = 0.0
    for i, s in enumerate(scaled):
        running += s
        end = float(target_total) if i == n - 1 else round(running * 2) / 2
        start = boundaries[-1][1] if boundaries else 0.0
        end = max(end, start)  # never let rounding produce a negative-length segment
        boundaries.append((start, end))
    return boundaries


def format_timestamp(seconds: float) -> str:
    """`MM:SS` formatting for a Veo multi-segment prompt timestamp."""
    total = max(0, int(round(seconds)))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"
