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


def suggest_scene_target(target_seconds: int) -> str:
    """A scene-count-range string for scenes.py's own `target` prompt param
    (its existing default is the plain literal "8-14 scenes") — sized so
    roughly `target_seconds` worth of ~6s shots at ~2.5 shots/scene fits,
    with a floor of 1 scene."""
    est_shots = max(1, round(target_seconds / ASSUMED_SECONDS_PER_SHOT))
    est_scenes = max(1, round(est_shots / ASSUMED_SHOTS_PER_SCENE))
    lo, hi = max(1, est_scenes - 1), est_scenes + 1
    return f"{lo}-{hi} scenes (aiming for a total runtime near {target_seconds}s)"


def suggest_shots_per_scene(target_seconds: int, scene_count: int) -> str:
    """Guidance text for cinematography.py's prompt: roughly how many shots
    each scene should carry so the film's total shot count — at the ~6s/shot
    planning assumption — lands near `target_seconds`. Empty string when
    there's nothing to compute (`scene_count <= 0`)."""
    if scene_count <= 0:
        return ""
    est_total_shots = max(scene_count * 2, round(target_seconds / ASSUMED_SECONDS_PER_SHOT))
    per_scene = max(2, round(est_total_shots / scene_count))
    return (f"the film has a target total runtime of about {target_seconds}s; at roughly "
           f"{ASSUMED_SECONDS_PER_SHOT}s per shot, that's about {est_total_shots} shot(s) "
           f"total across {scene_count} scene(s) — aim for about {per_scene} shot(s) per "
           "scene on average (some scenes can carry more, others fewer, as the story needs; "
           "this is a budget to aim near, not a hard per-scene quota)")


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
