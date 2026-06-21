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

def _sample(items: list, k: int) -> list:
    """Up to k items, evenly spread across the list (keep order)."""
    if not items or k <= 0:
        return []
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _resolve_character(text: str, scene_dialogue: list, casting: dict, out) -> tuple[str | None, str | None]:
    """Best character + image for a beat: explicit casting-name match, then alias
    keywords (crew/sailor/villager), else the scene's first speaking character."""
    from pathlib import Path
    entries = []
    for c in casting.get("casting", []):
        img = (c.get("character") or {}).get("image_path") or c.get("image_path")
        entries.append((c.get("name", ""), img))
    blob = (text + " " + " ".join(w for _s, w in scene_dialogue)).lower()

    for name, img in entries:                       # explicit name in the beat
        if name and img and name.lower() in (text or "").lower() and (Path(out) / img).exists():
            return name, str(Path(out) / img)
    aliases = {"crew": "Automation", "generator": "Automation", "automation": "Automation",
               "sailor": "Saoirse", "ship": "Saoirse", "boat": "Saoirse", "villager": "Villag"}
    for kw, frag in aliases.items():
        if kw in blob:
            for name, img in entries:
                if frag.lower() in name.lower() and img and (Path(out) / img).exists():
                    return name, str(Path(out) / img)
    for name, img in entries:                       # fallback: any present character
        if img and (Path(out) / img).exists():
            if name.lower() in blob:
                return name, str(Path(out) / img)
    return (entries[0][0], str(Path(out) / entries[0][1])) if entries and entries[0][1] else (None, None)


def _av(scene_no: int, soundscape: dict, visuals: dict) -> tuple[str, str]:
    """(visual style, audio) strings for a scene from the design docs."""
    snd = next((s for s in soundscape.get("soundscapes", []) if s.get("scene_number") == scene_no), {})
    vis = next((s for s in visuals.get("scenes", []) if s.get("scene_number") == scene_no), {})
    vparts = [vis.get("color_palette"), vis.get("lighting"), vis.get("visual_filter")]
    visual = ", ".join(p for p in vparts if p)
    aparts = [snd.get("ambient_bed")]
    aparts += [e.get("sound") if isinstance(e, dict) else str(e) for e in snd.get("sound_events", [])]
    audio = "; ".join(p for p in aparts if p)
    return visual, audio


def to_storyboard(scenes: list[dict], soundscape: dict, visuals: dict, casting: dict,
                  out, max_scenes: int = 2, max_shots: int | None = None) -> dict:
    """Build a render-ready storyboard: scenes → shots, each a Veo prompt fusing the
    screenplay action + dialogue with the scene's visual style and audio.

    `max_scenes` limits how many SCENES are built (the demo uses 2-3). `max_shots`
    is None by default → **every** action beat in a scene becomes a shot (all shots
    rendered); set it only if you deliberately want to cap shots per scene.
    """
    board = {"storyboard_style": "photoreal cinematic, screenplay-driven, native audio", "storyboard": []}
    for idx, sc in enumerate(scenes[:max_scenes], start=1):
        visual, audio = _av(idx, soundscape, visuals)
        beats = (_sample(sc["action"], max_shots) if max_shots else sc["action"]) or [sc["slugline"]]
        frames = []
        for fnum, action in enumerate(beats, start=1):
            who, _img = _resolve_character(action, sc["dialogue"], casting, out)
            # quote a line of dialogue (if any) so Veo voices it
            said = ""
            for spk, line in sc["dialogue"]:
                if who and who.split()[0].lower() in spk.lower():
                    said = f' {spk} says: "{line}".'
                    break
            prompt = (f"{sc['slugline']}. {action}"
                      f"{said}"
                      f" Cinematic, natural motion."
                      + (f" Visual style: {visual}." if visual else "")
                      + (f" Audio: {audio}." if audio else ""))
            frames.append({
                "frame": fnum,
                "moment": action[:80],
                "characters_in_frame": [who] if who else [],
                "image_prompt": prompt,
            })
        board["storyboard"].append({"scene_number": idx, "slugline": sc["slugline"], "frames": frames})
    return board
