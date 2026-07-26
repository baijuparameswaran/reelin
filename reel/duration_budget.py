"""Deterministic, engine-independent helpers turning a target total video
runtime into planning guidance for scenes/cinematography, and for
formatting/summing the resulting per-scene duration estimates.

Deliberately has NO knowledge of which video backend is configured — Veo is
only one of several supported image-to-video engines (`reel/i2v.py` also
supports diffusers-based LTX/Wan/CogVideoX and a remote comfyui/http
endpoint), so scene/shot COUNT planning must not assume Veo's specific
capabilities. Any backend-specific constraint on the actual rendered clip
length (e.g. Veo 3.1's `duration_seconds` only accepting 4, 6, or 8 — see
`i2v.py`'s `VEO_VALID_DURATIONS`/`_veo_nearest_valid_duration`, applied only
inside `i2v._gen_gemini`) lives with that backend's own code, not here.
"""
from __future__ import annotations

import re

DEFAULT_TARGET_SECONDS = 45

# Planning-stage assumption ONLY — used to convert a target runtime into
# scene-count/shots-per-scene GUIDANCE text fed to the scenes/cinematography
# prompts. Scene/shot counts stay a creative judgment call for those agents;
# this is a budget hint, not forced arithmetic, and not tied to any one
# engine's actual clip-length capability.
ASSUMED_SECONDS_PER_SHOT = 6
ASSUMED_SHOTS_PER_SCENE = 2.5


# NO scene-count guidance is produced here, deliberately. `suggest_scene_target`
# used to turn `target_seconds` into a literal range ("roughly 2-4 scenes" at
# the 45s default) that `scenes.PROMPT` interpolated into the END of its
# opening sentence — the highest-salience position in the prompt, immediately
# after two anti-inflation warnings. A concrete number there reliably beat the
# prose forty lines below telling the model to capture the story fully, so the
# stage was capped by a runtime default rather than by the story's own shape.
#
# It also contradicted this project's own convention that a runtime/quantity
# budget bounds RENDERING, not design: `--max-scenes` restricts only the
# media-generating stages while every design stage processes every scene. The
# runtime target still does real work below — shots-per-scene pacing, and each
# clip's requested duration in i2v — but it no longer decides how many scenes
# a story gets. `scenes.segment_scenes` uses its own coverage-first default
# `target` instead.


def suggest_shots_per_scene(target_seconds: int, scene_count: int) -> str:
    """Guidance text for cinematography.py's prompt: roughly how many shots
    each scene should carry so the film's total shot count — at the ~6s/shot
    planning assumption — lands near `target_seconds`. Empty string when
    there's nothing to compute (`scene_count <= 0`). Floor of 1 shot/scene,
    not 2 — a scene needs only one decisive shot when the beat is that
    simple; this is a soft budget hint, not a coverage-count minimum OR
    maximum — cinematography.py's own coverage rule takes priority over it.

    TBD — the same anchor problem `suggest_scene_target` was removed for, one
    level down; deliberately left in place for now rather than changed
    alongside that removal, so the scene-count fix could be measured on its
    own. What's wrong: this is a CONSERVED TOTAL. `est_total_shots` is pinned
    at `target_seconds / 6` and then DIVIDED by `scene_count`, so more scenes
    mechanically means fewer shots each — at the 45s default that's ~8 shots
    for the whole film regardless, and the scene-count fix (which took the
    bundled sample from 4 scenes to 9) therefore halves this stage's
    per-scene budget from ~2 shots to 1 as a side effect. Two stages now pull
    against each other: scenes is told to cover the story fully, and
    cinematography is told it has a fixed shot pool to spread across whatever
    scenes produced.

    Shape of the fix when it's picked up: state a per-scene FLOOR derived
    from coverage needs rather than a quotient of a fixed pool, and let the
    film's total runtime fall out of that — the same "budget bounds
    RENDERING, not design" direction the scene-count removal took. Worth
    measuring the same way (A/B on the bundled samples) rather than assuming,
    since unlike scene count, shot count does have a real per-clip cost: every
    shot becomes a panel and every panel a Veo call, so uncapping it moves
    real money in a way uncapping scene count did not."""
    if scene_count <= 0:
        return ""
    est_total_shots = max(scene_count, round(target_seconds / ASSUMED_SECONDS_PER_SHOT))
    per_scene = max(1, round(est_total_shots / scene_count))
    return (f"the film has a soft target total runtime of about {target_seconds}s; at "
           f"roughly {ASSUMED_SECONDS_PER_SHOT}s per shot, that's about {est_total_shots} "
           f"shot(s) total across {scene_count} scene(s) — roughly {per_scene} shot(s) per "
           "scene on average IF that's enough for real coverage (some scenes need more, "
           "some need fewer — this is a budget to lean toward, never a reason to under-shoot "
           "a scene that needs more coverage than this average suggests)")


def parse_duration_seconds(text: str | None) -> int:
    """Parse a duration string ("8s", "1m 30s", "2m") back into whole
    seconds. Returns 0 for empty/unparseable input — never raises."""
    text = (text or "").strip()
    if not text:
        return 0
    m = re.match(r"(?:(\d+)m)?\s*(?:(\d+)s)?", text)
    if not m or not (m.group(1) or m.group(2)):
        return 0
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def format_seconds(total: int) -> str:
    """The inverse of `parse_duration_seconds` — "45s" under a minute,
    "1m 30s" / "2m" at or above it. Shared by storyboard.py's per-scene
    `_format_total_duration` and `estimated_total_seconds` below so both
    render the same way."""
    if total < 60:
        return f"{total}s"
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"


def estimated_total_seconds(storyboard: dict) -> int:
    """Sum every scene's own `header.duration_estimate` (already computed by
    storyboard._build_scene_board) into one film-wide estimate, for
    comparing against the target at the gate. Engine-independent — this is
    the PLANNED runtime the storyboard describes, before any backend-specific
    per-clip rounding (e.g. Veo's 4/6/8s constraint) is applied at render time."""
    return sum(
        parse_duration_seconds(scene.get("header", {}).get("duration_estimate"))
        for scene in storyboard.get("storyboard", [])
    )
