"""Minimal Fountain parser + screenplay→storyboard builder for video rendering.

Reads `output/screenplay.fountain` (the draft the screenplay agent writes) and
turns it into a render plan: per scene, an ordered set of **shots** (panels),
each fusing the screenplay's action/dialogue with the scene's **visual
design** (color, light, filter — from visuals.json), **audio** (ambient bed,
sound events, spoken lines — from soundscape.json), and **camera grammar**
(cinematography.json). The board this module builds is SHAPE-COMPATIBLE with
`storyboard.py`'s own deterministic build (`header.location`, `visual_overview`,
`audio_overview`, per-panel `characters_in_frame`/`camera_angle`/
`camera_movement`/`lens`/`composition`/`dialogue`/`sound`) — both feed into
the SAME shared `pipeline._render_scene_frames`, which always rebuilds the
actual Veo prompt via `reel.veo_prompt`'s five-part formula from these
structured fields, never from a free-text prompt this module might build.
So the two things that matter for parity with the main pipeline's render are
(1) this module populates every field the shared formula reads, and (2) the
`image_prompt` this module writes for a human preview is built via that SAME
shared formula, not a separate approximation — both fixed in this module's
2026-07-14 rework (see PROGRESS.md's session log for the full audit that
found the two entry points had silently drifted).

The draft the screenplay agent emits isn't strict Fountain (it mixes inline
speaker/parenthetical/dialogue), so the parser is deliberately lenient: it
pulls scene headings, narrative *action* beats, and (speaker, line) dialogue,
and is forgiving about the rest.

KNOWN LIMITATION (not fixed here — would need `parse()` itself reworked):
`parse()` keeps a scene's action beats and dialogue lines in two SEPARATE
lists, discarding their original interleaved order — so which beat a given
dialogue line "belongs to" can only ever be approximated (`_distribute_dialogue`
spreads dialogue lines proportionally across beats, the same technique
`_camera` already uses to spread cinematography's shot list across beats),
never read exactly off the source text the way `storyboard.py`'s per-panel
`dialogue` (sourced from the screenplay agent's own per-shot JSON) can.
"""
from __future__ import annotations

import re

from . import veo_prompt

_HEADING = re.compile(r"^(INT|EXT|EST|INT\.?/EXT|I/E)[\.\s/]", re.I)
_SKIP = re.compile(r"^(Title:|Credit:|Author:|Draft date:|CAPTIONS:|=|\[\[)", re.I)
# "Crew Leader *shouts over the wind.* We need to set up the generators!"
_INLINE = re.compile(r"^([A-Z][A-Za-z0-9'’ \-]{1,28}?)\s*\*([^*]*)\*\s*(.+)$")


def _clean(line: str) -> str:
    return line.strip().strip("*").strip()


def parse(text: str) -> list[dict]:
    """Parse fountain into scenes: [{slugline, action:[str], dialogue:[(who,line)]}]."""
    scenes: list[dict] = []
    cur: dict | None = None
    pending_speaker: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # NB: don't drop a pending speaker here — this draft puts a blank line
            # between the speaker cue and the spoken line.
            continue
        if _HEADING.match(line):
            cur = {"slugline": line, "action": [], "dialogue": []}
            scenes.append(cur)
            pending_speaker = None
            continue
        if cur is None or _SKIP.match(line):
            continue

        # inline "SPEAKER *stage dir* dialogue"
        m = _INLINE.match(line)
        if m:
            who, _stage, said = m.group(1).strip(), m.group(2), m.group(3).strip()
            cur["dialogue"].append((who, said))
            continue

        body = _clean(line)
        # a bare speaker cue (short, mostly a name) → its dialogue is the next line
        words = body.split()
        if len(words) <= 4 and body == body.strip() and not body.endswith((".", "!", "?", ",")) \
                and (body.isupper() or body.istitle()):
            pending_speaker = body
            continue
        if pending_speaker:
            cur["dialogue"].append((pending_speaker, body))
            pending_speaker = None
            continue
        cur["action"].append(body)
    return scenes


# ── screenplay → storyboard (shots with A/V) ─────────────────────────────────

# Cinematography agent angle → Veo guide vocabulary.
_VEO_ANGLE: dict[str, str] = {
    "bird's-eye": "bird's eye view",
    "bird's eye": "bird's eye view",
    "worm's-eye": "worms eye",
    "worm's eye": "worms eye",
    "dutch-tilt": "Dutch tilt",
    "dutch tilt": "Dutch tilt",
    "high": "high angle",
    "low": "low angle",
}

# Cinematography agent movement → Veo guide vocabulary.
_VEO_MOVEMENT: dict[str, str] = {
    "dolly-in": "dolly in",
    "dolly-out": "dolly out",
    "crane-up": "crane up",
    "crane-down": "crane down",
    "whip-pan": "whip pan",
    "zoom-in": "zoom in",
    "zoom-out": "zoom out",
    "steadicam": "Steadicam tracking",
    "static": "static",
}


def _sample(items: list, k: int) -> list:
    """Up to k items, evenly spread across the list (keep order)."""
    if not items or k <= 0:
        return []
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _cast_entries(casting: dict, *, kind: str | None = None, exclude_kind: str | None = None) -> list[dict]:
    entries = []
    for c in casting.get("casting", []):
        k = c.get("kind")
        if kind is not None and k != kind:
            continue
        if exclude_kind is not None and k == exclude_kind:
            continue
        entries.append(c)
    return entries


def _named_in_text(text: str, names: list[str]) -> list[str]:
    """Names (word-boundary, case-insensitive) that appear in `text`, in
    `names`' own order — the same matching approach
    `storyboard._panel_characters_in_frame`/`veo_prompt.panel_relevant_characters`
    use to ground a character list in actual prose rather than assuming."""
    if not text:
        return []
    return [n for n in names if n and re.search(rf"\b{re.escape(n)}\b", text, re.IGNORECASE)]


def _distribute_dialogue(dialogue: list[tuple[str, str]], n_beats: int) -> list[list[tuple[str, str]]]:
    """Approximate mapping of each dialogue line to ONE beat index, via
    proportional position — the same technique `_camera` uses to spread
    cinematography's shot list across action beats. See the module
    docstring's "KNOWN LIMITATION" note: `parse()` discards the original
    interleaving between action and dialogue, so this can only approximate
    which beat a line belongs to, not read it exactly."""
    buckets: list[list[tuple[str, str]]] = [[] for _ in range(max(n_beats, 0))]
    if not dialogue or n_beats <= 0:
        return buckets
    for i, pair in enumerate(dialogue):
        beat_idx = min(int(i * n_beats / len(dialogue)), n_beats - 1)
        buckets[beat_idx].append(pair)
    return buckets


def _canonical_name(spk: str, names: list[str]) -> str | None:
    """Case-insensitive match of a Fountain-parsed speaker cue against
    casting.json's own names, resolved to the CANONICAL (casting.json)
    casing. Fountain speaker cues are conventionally ALL CAPS ("ALICE")
    while casting.json names are usually Title Case ("Alice") — comparing
    case-sensitively would both miss real matches and, worse, leak the raw
    ALL-CAPS cue into `characters_in_frame`, where every other consumer
    (`casting_lookup.get(name)` in `veo_prompt.py`) does an exact-string
    dict lookup keyed by casting.json's own casing."""
    lspk = spk.lower()
    return next((n for n in names if n.lower() == lspk), None)


def _beat_characters(action: str, beat_dialogue: list[tuple[str, str]],
                     scene_dialogue: list[tuple[str, str]], casting: dict) -> list[str]:
    """Every non-location, non-prop cast name relevant to ONE beat: named in
    its own action text, or speaking in dialogue attributed to this beat.
    Falls back to whoever speaks ANYWHERE in the scene (a beat with no
    dialogue of its own but a scene that clearly has speaking characters is
    more likely to include them than not), then to the single first cast
    entry with a rendered image (preserving the old single-character
    fallback for a scene with no resolvable named characters at all) —
    mirrors `storyboard._panel_characters_in_frame`'s own fallback chain
    (narrow first, degrade gracefully, never end up with an empty list if
    ANY cast entry could plausibly apply)."""
    names = [c.get("name", "") for c in _cast_entries(casting, exclude_kind="location")
            if c.get("kind") != "prop" and c.get("name")]
    relevant = _named_in_text(action, names)
    for spk, _ in beat_dialogue:
        canon = _canonical_name(spk, names)
        if canon and canon not in relevant:
            relevant.append(canon)
    if relevant:
        return relevant
    scene_speakers: list[str] = []
    for spk, _ in scene_dialogue:
        canon = _canonical_name(spk, names)
        if canon and canon not in scene_speakers:
            scene_speakers.append(canon)
    if scene_speakers:
        return scene_speakers
    for c in _cast_entries(casting, exclude_kind="location"):
        ch = c.get("character", c)
        if (ch.get("image_path") or c.get("image_path")) and c.get("name"):
            return [c["name"]]
    return []


def _panel_sound(action: str, soundscape_scene: dict) -> str:
    """ambient|sfx string in the `" | "`-delimited format
    `veo_prompt.panel_video_prompt` expects — mirrors
    `storyboard._panel_sound`'s own construction: ambient bed (empty when
    `silence` is true) plus whichever `sound_events` entry's `moment` text
    overlaps this beat's own action text."""
    ambient = "" if soundscape_scene.get("silence") else (soundscape_scene.get("ambient_bed") or "")
    sfx = ""
    moment = (action or "").lower()
    for ev in soundscape_scene.get("sound_events", []):
        ev_moment = (ev.get("moment") or "").lower() if isinstance(ev, dict) else ""
        if ev_moment and moment and (ev_moment in moment or moment in ev_moment):
            sfx = ev.get("sound", "") if isinstance(ev, dict) else str(ev)
            break
    return " | ".join(p for p in (ambient, sfx) if p)


def _camera(scene_no: int, frame_idx: int, n_frames: int, cinematography: dict) -> dict:
    """This frame's camera fields (shot_type/camera_angle/camera_movement/
    lens/composition), Veo-vocabulary-normalized, from the DP's shot list.

    cinematography.json declares an ordered list of `shots` per scene (type/
    angle/movement/lens/framing). There are usually fewer planned shots than
    action beats, so map each frame to a shot by proportional index — every
    beat inherits the nearest planned shot's camera grammar. Empty dict when
    the scene has no cinematography entry (the caller's fields all degrade
    gracefully to blank/omitted)."""
    sc = next((s for s in cinematography.get("scenes", []) if s.get("scene_number") == scene_no), {})
    shots = sc.get("shots", []) if isinstance(sc, dict) else []
    if not shots:
        return {}
    sh = shots[min(int(frame_idx * len(shots) / max(n_frames, 1)), len(shots) - 1)]

    raw_type = (sh.get("type") or "").strip()
    raw_angle = (sh.get("angle") or "").strip().lower()
    raw_movement = (sh.get("movement") or "").strip().lower()
    raw_lens = (sh.get("lens") or "").strip()
    raw_framing = (sh.get("framing") or "").strip()

    return {
        "shot_type": raw_type,
        "camera_angle": _VEO_ANGLE.get(raw_angle, raw_angle) if raw_angle else "",
        "camera_movement": ("" if raw_movement == "static" else _VEO_MOVEMENT.get(raw_movement, raw_movement)),
        "lens": raw_lens,
        "composition": raw_framing,
    }


def _scene_context(scene_no: int, scenes: dict | None, soundscape: dict, visuals: dict,
                   casting: dict, slugline: str) -> dict:
    """This scene's `header`/`visual_overview`/`audio_overview` — the fields
    `pipeline._render_scene_frames` reads once per scene (not per panel).
    Field mapping matches `storyboard._build_scene_board`'s own construction
    exactly (color_palette/lighting->lighting_setup/emotional_function->mood;
    ambient_bed->ambient, score_direction->score_cue), so a scene rendered
    via this standalone path gets the same Style & Ambiance / audio grounding
    a full pipeline run would give it.

    Location prefers scenes.json's own `location` field (scenes.json's
    `number`, not `scene_number` — a different key name than
    soundscape/visuals/cinematography use, an existing quirk of this
    codebase's schemas) — falling back to a crude slugline extraction
    ("INT. BAR - NIGHT" -> "BAR") when scenes.json isn't available, so this
    command still degrades gracefully rather than requiring an artifact its
    own docstring doesn't otherwise depend on."""
    loc_name = ""
    if scenes:
        sc = next((s for s in scenes.get("scenes", []) if s.get("number") == scene_no), None)
        if sc:
            loc_name = (sc.get("location") or "").strip()
    if not loc_name:
        m = re.match(r"^(?:INT|EXT|EST|INT\.?/EXT|I/E)[\.\s/]+([^-]+)", slugline, re.I)
        loc_name = (m.group(1).strip().rstrip(".") if m else "")

    vis = next((s for s in visuals.get("scenes", []) if s.get("scene_number") == scene_no), {})
    snd = next((s for s in soundscape.get("soundscapes", []) if s.get("scene_number") == scene_no), {})

    return {
        "header": {"location": loc_name},
        "visual_overview": {
            "color_palette": vis.get("color_palette", ""),
            "lighting_setup": vis.get("lighting", ""),
            "mood": vis.get("emotional_function", ""),
        },
        "audio_overview": {
            "score_cue": snd.get("score_direction", ""),
            "ambient": "" if snd.get("silence") else snd.get("ambient_bed", ""),
        },
        "_soundscape_scene": snd,  # internal — consumed by _panel_sound below only
    }


def to_storyboard(scenes: list[dict], soundscape: dict, visuals: dict, casting: dict,
                  out, max_scenes: int | None = None, max_shots: int | None = None,
                  cinematography: dict | None = None, scenes_json: dict | None = None) -> dict:
    """Build a render-ready storyboard: scenes → panels, each carrying every
    field `pipeline._render_scene_frames`/`reel.veo_prompt` need to build the
    SAME five-part Veo prompt a full pipeline run's storyboard.json would —
    location, visual/audio overview, structured dialogue, camera fields, and
    a (possibly multi-character) `characters_in_frame`, not just a single
    best-guess name. `image_prompt` is still written per panel, purely as a
    human-readable preview (`output/storyboard.json` is written to disk for
    inspection) — built via the SAME shared `veo_prompt.panel_video_prompt`
    formula the real render uses, so the preview is truthful rather than a
    separate approximation.

    No artificial caps by default — the **story** sets the extent: every
    scene the screenplay drafted, and every action beat within it, becomes a
    panel. `max_scenes`/`max_shots` stay available as optional overrides
    (None = all). `cinematography` is cinematography.json; its per-scene
    shot list drives the camera language. `scenes_json` (scenes.json,
    optional) supplies each scene's authoritative `location` — see
    `_scene_context`'s docstring for the fallback when it's absent.
    """
    cine = cinematography or {}
    board = {"storyboard_style": "cinematic, photorealistic", "storyboard": []}
    chosen = scenes[:max_scenes] if max_scenes else scenes

    casting_lookup: dict[str, dict] = {c.get("name"): c for c in casting.get("casting", []) if c.get("name")}

    for idx, sc in enumerate(chosen, start=1):
        ctx = _scene_context(idx, scenes_json, soundscape, visuals, casting, sc["slugline"])
        soundscape_scene = ctx.pop("_soundscape_scene")
        location_desc = (casting_lookup.get(ctx["header"]["location"], {})
                        .get("character", {}).get("visual_prompt")) or ctx["header"]["location"]

        beats = (_sample(sc["action"], max_shots) if max_shots else sc["action"]) or [sc["slugline"]]
        dialogue_buckets = _distribute_dialogue(sc["dialogue"], len(beats))

        panels = []
        for fnum, action in enumerate(beats, start=1):
            beat_dialogue = dialogue_buckets[fnum - 1] if fnum - 1 < len(dialogue_buckets) else []
            characters_in_frame = _beat_characters(action, beat_dialogue, sc["dialogue"], casting)
            cam = _camera(idx, fnum - 1, len(beats), cine)

            # `speaker` is normalized to the CANONICAL casting.json name when
            # resolvable (case-insensitive match — Fountain's "ALICE" vs.
            # casting.json's "Alice") — both the in-frame/off-screen check
            # below AND veo_prompt.panel_video_prompt's own voice_index
            # lookup do exact-string matches keyed by casting.json's casing,
            # so an unnormalized ALL-CAPS cue would silently miss both. A
            # speaker with no cast match at all (a minor/unnamed character)
            # keeps whatever Fountain actually parsed.
            dialogue = []
            for spk, line in beat_dialogue:
                canon = _canonical_name(spk, characters_in_frame) or spk
                dialogue.append({
                    "speaker": canon, "line": line,
                    "modifier": "" if _canonical_name(spk, characters_in_frame) else "O.S.",
                })

            panel = {
                "panel": fnum,
                "shot_type": cam.get("shot_type", ""),
                "camera_angle": cam.get("camera_angle", ""),
                "camera_movement": cam.get("camera_movement", ""),
                "lens": cam.get("lens", ""),
                "composition": cam.get("composition", ""),
                "characters_in_frame": characters_in_frame,
                "action": action,
                "dialogue": dialogue,
                "sound": _panel_sound(action, soundscape_scene),
            }
            panel["image_prompt"] = veo_prompt.panel_video_prompt(
                panel, ctx["audio_overview"], casting_lookup=casting_lookup,
                location_desc=location_desc, visual_overview=ctx["visual_overview"],
            )
            panels.append(panel)

        board["storyboard"].append({
            "scene_number": idx,
            "slugline": sc["slugline"],
            "header": ctx["header"],
            "visual_overview": ctx["visual_overview"],
            "audio_overview": ctx["audio_overview"],
            "panels": panels,
        })
    return board
