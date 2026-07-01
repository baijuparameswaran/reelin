"""Minimal Fountain parser + screenplay→storyboard builder for video rendering.

Reads `output/screenplay.fountain` (the draft the screenplay agent writes) and
turns it into a render plan: per scene, an ordered set of **shots**, each fusing
the screenplay's action/dialogue with the scene's **visual design** (color, light,
filter — from visuals.json) and **audio** (ambient bed, sound events, spoken
lines — from soundscape.json). Veo 3.x renders synchronized audio from the prompt,
so the audio cues and quoted dialogue drive the clip's sound.

The draft the screenplay agent emits isn't strict Fountain (it mixes inline
speaker/parenthetical/dialogue), so the parser is deliberately lenient: it pulls
scene headings, narrative *action* beats, and (speaker, line) dialogue, and is
forgiving about the rest.
"""
from __future__ import annotations

import re

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

# Veo prompting guide — focus/lens terms by shot type.
# Covers both standard abbreviations (from storyboard schema) and the natural
# language values emitted by the cinematography agent, so the focus hint fires
# regardless of which path produced the shot_type label.
_VEO_FOCUS_FOUNTAIN: dict[str, str] = {
    # abbreviations
    "ECU": "portrait, extreme close-up, shallow focus",
    "CU": "portrait, shallow focus",
    "MCU": "shallow focus",
    "INSERT": "macro lens",
    "WS": "deep focus",
    "ELS": "deep focus",
    # natural language (cinematography agent output)
    "EXTREME-CLOSE-UP": "portrait, extreme close-up, shallow focus",
    "EXTREME CLOSE-UP": "portrait, extreme close-up, shallow focus",
    "EXTREME CLOSE UP": "portrait, extreme close-up, shallow focus",
    "CLOSE-UP": "portrait, shallow focus",
    "CLOSE UP": "portrait, shallow focus",
    "WIDE": "deep focus",
    "WIDE SHOT": "deep focus",
    "ESTABLISHING": "deep focus",
    "ESTABLISHING SHOT": "deep focus",
    "TWO-SHOT": "shallow focus",
    "TWO SHOT": "shallow focus",
    "OVER-THE-SHOULDER": "shallow focus",
    "OTS": "shallow focus",
    "POV": "shallow focus",
    "POV SHOT": "shallow focus",
}

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


def _resolve_character(text: str, scene_dialogue: list, casting: dict, out) -> tuple[str | None, str | None]:
    """Best character + image for a beat: explicit name match in the beat text,
    then any character whose name appears in the surrounding dialogue, else the
    first cast entry that has a rendered image."""
    from pathlib import Path
    entries = []
    for c in casting.get("casting", []):
        img = (c.get("character") or {}).get("image_path") or c.get("image_path")
        entries.append((c.get("name", ""), img))
    blob = (text + " " + " ".join(w for _s, w in scene_dialogue)).lower()

    for name, img in entries:                       # explicit name in the beat text
        if name and img and name.lower() in (text or "").lower() and (Path(out) / img).exists():
            return name, str(Path(out) / img)
    for name, img in entries:                       # name appears anywhere in the scene dialogue
        if img and (Path(out) / img).exists():
            if name.lower() in blob:
                return name, str(Path(out) / img)
    return (entries[0][0], str(Path(out) / entries[0][1])) if entries and entries[0][1] else (None, None)


def _camera(scene_no: int, frame_idx: int, n_frames: int, cinematography: dict) -> tuple[str, str]:
    """(camera clause, shot-type label) for a frame from the DP's shot list.

    cinematography.json declares an ordered list of `shots` per scene (type/angle/
    movement/lens/framing). There are usually fewer planned shots than action beats,
    so map each frame to a shot by proportional index — every beat inherits the
    nearest planned shot's camera grammar. ("", "") when the scene has no entry.

    Output uses Veo guide vocabulary throughout — no labeled sections ("Camera:/
    Framing:"), natural language terms, Veo-normalized angle and movement names.
    """
    sc = next((s for s in cinematography.get("scenes", []) if s.get("scene_number") == scene_no), {})
    shots = sc.get("shots", []) if isinstance(sc, dict) else []
    if not shots:
        return "", ""
    sh = shots[min(int(frame_idx * len(shots) / max(n_frames, 1)), len(shots) - 1)]

    raw_type = (sh.get("type") or "").strip()
    raw_angle = (sh.get("angle") or "").strip().lower()
    raw_movement = (sh.get("movement") or "").strip().lower()
    raw_lens = (sh.get("lens") or "").strip()
    raw_framing = (sh.get("framing") or "").strip()

    parts: list[str] = []
    # Composition — Veo guide: "wide shot", "close-up", "two-shot", etc.
    if raw_type:
        label = raw_type if raw_type.endswith("shot") or raw_type.endswith("view") else f"{raw_type} shot"
        parts.append(label)
    # Camera positioning — Veo guide: "eye-level", "low angle", "bird's eye view", "worms eye"
    if raw_angle:
        parts.append(_VEO_ANGLE.get(raw_angle, raw_angle))
    # Camera motion — Veo guide: "dolly in", "tracking", "panning", "aerial view"
    if raw_movement and raw_movement != "static":
        parts.append(_VEO_MOVEMENT.get(raw_movement, raw_movement))
    # Lens — Veo guide: "wide-angle lens", "macro lens", "telephoto lens"
    if raw_lens:
        parts.append(f"{raw_lens} lens" if "lens" not in raw_lens.lower() else raw_lens)
    # Framing note integrated naturally (no "Framing:" label)
    if raw_framing:
        parts.append(raw_framing)

    clause = ", ".join(parts) + "." if parts else ""
    return clause, raw_type


def _av(scene_no: int, soundscape: dict, visuals: dict) -> tuple[str, str, str]:
    """(visual, ambient, sfx) strings for a scene from the design docs.

    Veo guide separates three audio cue types:
      ambient — environment's soundscape ("A faint hum in the background")
      sfx     — explicitly described sounds ("tires screeching loudly")
    We map: ambient_bed → ambient; sound_events → sfx (explicit, action-driven).
    """
    snd = next((s for s in soundscape.get("soundscapes", []) if s.get("scene_number") == scene_no), {})
    vis = next((s for s in visuals.get("scenes", []) if s.get("scene_number") == scene_no), {})
    vparts = [vis.get("color_palette"), vis.get("lighting"), vis.get("visual_filter")]
    visual = ", ".join(p for p in vparts if p)
    # Ambient noise: the environment's soundscape (scene-level bed).
    ambient = (snd.get("ambient_bed") or "").strip()
    # SFX: explicitly described sound events (action-driven, panel-specific).
    sfx_events = [e.get("sound") if isinstance(e, dict) else str(e)
                  for e in snd.get("sound_events", [])]
    sfx = ", ".join(p for p in sfx_events if p)
    return visual, ambient, sfx


def to_storyboard(scenes: list[dict], soundscape: dict, visuals: dict, casting: dict,
                  out, max_scenes: int | None = None, max_shots: int | None = None,
                  cinematography: dict | None = None) -> dict:
    """Build a render-ready storyboard: scenes → shots, each a Veo prompt fusing the
    screenplay action + dialogue with the scene's **camera grammar** (cinematography:
    shot type/angle/movement/lens/framing), visual style and audio.

    No artificial caps by default — the **story** sets the extent: every scene the
    screenplay drafted, and every action beat within it, becomes a shot. `max_scenes`/
    `max_shots` stay available as optional overrides (None = all). `cinematography`
    is cinematography.json; its per-scene shot list drives the camera language.
    """
    cine = cinematography or {}
    style = "cinematic, photorealistic"
    board = {"storyboard_style": style, "storyboard": []}
    chosen = scenes[:max_scenes] if max_scenes else scenes
    for idx, sc in enumerate(chosen, start=1):
        visual, ambient, sfx = _av(idx, soundscape, visuals)
        beats = (_sample(sc["action"], max_shots) if max_shots else sc["action"]) or [sc["slugline"]]
        frames = []
        for fnum, action in enumerate(beats, start=1):
            who, _img = _resolve_character(action, sc["dialogue"], casting, out)
            # Dialogue — Veo guide: use quotation marks for specific speech.
            # In-frame: Speaker says, "line."  Off-screen: "line" (Speaker, off screen).
            speech_parts: list[str] = []
            for spk, line in sc["dialogue"]:
                if who and who.split()[0].lower() in spk.lower():
                    speech_parts.insert(0, f'{spk} says, "{line}"')
                else:
                    speech_parts.append(f'"{line}" ({spk}, off screen)')
            camera, shot_type = _camera(idx, fnum - 1, len(beats), cine)
            # Focus & Ambiance — shot-type-driven lens/focus hint (Veo guide).
            focus = _VEO_FOCUS_FOUNTAIN.get(shot_type.upper(), "") if shot_type else ""

            # Veo guide element order (strict):
            # 1 Subject+Action — who/what does what (action beat is the primary vehicle)
            # 2 Style          — cinematic style keywords
            # 3 Camera         — positioning (eye-level, aerial view) + motion (dolly, tracking)
            # 4 Composition    — framing (wide shot, close-up) already encoded in camera clause
            # 5 Focus & Ambiance — lens/focus term + color/lighting mood
            # Audio (appended after visuals per guide):
            #   Ambient noise  — environment's soundscape (ambient_bed)
            #   SFX            — explicitly described sounds (sound_events)
            #   Dialogue       — quoted speech (Veo voices these lines)
            p_parts = [action]                              # 1 Subject + Action
            p_parts.append("Cinematic, photorealistic.")   # 2 Style
            if camera:
                p_parts.append(camera)                     # 3+4 Camera & Composition
            if visual:                                     # 5 Ambiance (color/lighting)
                p_parts.append(visual + ".")
            if focus:                                      # 5 Focus/lens
                p_parts.append(focus + ".")
            if ambient:                                    # Audio: ambient noise
                p_parts.append(ambient + ("." if not ambient.rstrip().endswith(".") else ""))
            if sfx:                                        # Audio: SFX
                p_parts.append(sfx + ("." if not sfx.rstrip().endswith(".") else ""))
            if speech_parts:                               # Audio: dialogue (quoted)
                p_parts.extend(speech_parts[:3])
            prompt = " ".join(p.strip() for p in p_parts if p.strip())
            frames.append({
                "frame": fnum,
                "moment": action[:80],
                "shot_type": shot_type,
                "characters_in_frame": [who] if who else [],
                "image_prompt": prompt,
            })
        board["storyboard"].append({"scene_number": idx, "slugline": sc["slugline"], "frames": frames})
    return board
