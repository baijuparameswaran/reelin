"""Pipeline orchestration for the screenplay-material phase.

Phase graph:

    ingest ─┬─▶ structure ──┐
            └─▶ characters ──┴─▶ scenes ─▶ casting ─┬─▶ soundscape ─────┐
                                                      ├─▶ visuals ─────────┼─▶ storyboard ─┐
                                                      └─▶ cinematography ──┘               ├─▶ assemble
                                                                          screenplay ──────┘
   (structure & characters concurrent; scenes then casting run sequentially — casting
   also casts each scene's `location` (scenes.py's scene→location field), so it needs
   scenes' output and can no longer run concurrently with it)
   (soundscape, visuals, cinematography concurrent)

Creative crew roles:
  casting        — locks each character's visual form (and each distinct scene
                   location's, kind: "location", no actor layer); renders one
                   reference image per character/location via Gemini
                   (`output/casting/<name>.png`)
  soundscape     — background score / sound design
  visuals        — art production (color, props, production design)
  cinematography — Director of Photography (shot types, angles, movement, lens)
  storyboard     — fuses casting + art + camera + score into an image_prompt per
                   panel (a fallback format only — see _five_part_veo_prompt below)

After storyboard + screenplay, an optional scene-render phase
(`_render_scene_frames` → `output/video/`) renders each storyboard panel into a
video clip via Gemini Veo (`reel.i2v`), seeded by the character's casting image
for identity continuity. Best-effort: no API key → skips with a hint. The
prompt actually sent to Veo is NOT the storyboard's free-text image_prompt —
it's reconstructed from structured data (casting.json, the scene's
visual_overview, the panel's own camera fields) into a fixed five-part
formula, always in this order: [Cinematography] + [Subject] + [Action] +
[Context] + [Style & Ambiance] (Google's Veo 3.1 prompting guide; see
_five_part_veo_prompt / _panel_video_prompt).

Each LLM stage passes through a human-in-the-loop gate: the operator can
approve the result, supply revision feedback, or let it auto-approve on
timeout. Parallel branches are gated independently after all complete.
HITL is controlled via `config/models.yaml` under the `hitl` key.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agents.ingest import ingest
from .agents.structure import analyze_structure
from .agents.characters import extract_characters
from .agents.scenes import segment_scenes
from .agents.casting import cast_characters
from .agents.soundscape import design_soundscape
from .agents.visuals import design_visuals
from .agents.cinematography import plan_cinematography
from .agents.storyboard import plan_storyboard
from .agents.screenplay import draft_screenplay, to_fountain
from .agents import fidelity
from .agents import genre as genre_agent
from .agents.moodboard import design_moodboard, guidance as moodboard_guidance
from .gate import Gate
from . import llm
from . import imagegen
from . import i2v
from . import veo_guide
from . import gemini
from . import session


def _log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


def _scenes_label(max_scenes: int | None) -> str:
    """Display form of a max_scenes cap — 'all' when unbounded (None), matching
    the `--max-scenes all` CLI value, instead of printing the literal 'None'."""
    return "all" if max_scenes is None else str(max_scenes)


# ── per-stage gate summarizers ────────────────────────────────────────────────

def _summarize_structure(r: dict) -> str:
    lines = [
        f"Logline:  {r.get('logline', '?')[:100]}",
        f"Genre:    {r.get('genre', '?')}  |  Tone: {r.get('tone', '?')}",
        f"Themes:   {', '.join(r.get('themes', []))}",
        f"Conflict: {r.get('central_conflict', '?')[:100]}",
    ]
    for act, beats in r.get("three_act", {}).items():
        lines.append(f"  {act}:")
        for b in (beats or [])[:3]:
            lines.append(f"    · {b}")
    return "\n".join(lines)


def _summarize_characters(r: dict) -> str:
    rows = []
    for c in r.get("characters", []):
        rows.append(
            f"  · {c.get('name','?')} ({c.get('role','?')}): "
            f"{c.get('description','?')[:80]}"
        )
    return "\n".join(rows) or "  (none)"


def _summarize_scenes(r: dict) -> str:
    rows = []
    for s in r.get("scenes", []):
        rows.append(f"  {s.get('number','?'):>2}. {s.get('slugline','?')}")
        rows.append(f"      {s.get('summary','?')[:80]}")
    dropped = r.get("dropped_scenes") or []
    if dropped:
        # Gaps in the numbering above (e.g. 2,3,5,6,7) usually mean the model
        # generated a scene whose source_line couldn't be matched in the source
        # text — surfaced here instead of silently vanishing from the list.
        rows.append(f"  ⚠ {len(dropped)} scene(s) dropped — source_line not found in source text:")
        for s in dropped:
            rows.append(f"      {s.get('number','?')}. {s.get('slugline','?')}  "
                        f"(source_line: \"{(s.get('source_line') or '')[:60]}\")")
    return "\n".join(rows) or "  (none)"


def _summarize_casting(r: dict) -> str:
    rows = []
    for c in r.get("casting", []):
        character = c.get("character", c)
        if c.get("kind") == "location":
            rows.append(f"  · {c.get('name','?')}  [location]")
            vp = character.get("visual_prompt", "")
            if vp:
                rows.append(f"      {vp[:80]}")
            continue
        actor = c.get("actor", c)
        brief = actor.get("casting_brief", "?")
        rows.append(f"  · {c.get('name','?')}: {brief[:70]}")
        pf = character.get("physical_form", "")
        if pf:
            rows.append(f"      {pf[:80]}")
    return "\n".join(rows) or "  (none)"


def _summarize_soundscape(r: dict) -> str:
    rows = [f"Audio palette: {r.get('audio_palette', '?')}"]
    for s in r.get("soundscapes", []):
        bed = s.get("ambient_bed") or "(silence)"
        rows.append(f"  {s.get('scene_number','?'):>2}. {bed[:70]}")
        fn = s.get("emotional_function", "")
        if fn:
            rows.append(f"      → {fn[:80]}")
    return "\n".join(rows)


def _summarize_visuals(r: dict) -> str:
    rows = [
        f"Visual palette: {r.get('visual_palette', '?')}",
        f"Color language: {r.get('color_language', '?')[:80]}",
    ]
    for s in r.get("scenes", []):
        rows.append(f"  {s.get('scene_number','?'):>2}. {s.get('color_palette','?')[:70]}")
        vf = s.get("visual_filter", "")
        if vf:
            rows.append(f"      filter: {vf}")
    return "\n".join(rows)


def _summarize_moodboard(r: dict) -> str:
    rows = [f"Aesthetic: {r.get('overall_aesthetic', '?')[:90]}"]
    if r.get("palette"):
        rows.append(f"  Palette: {', '.join(str(c) for c in r['palette'][:6])}")
    if r.get("lighting_mood"):
        rows.append(f"  Light: {r['lighting_mood'][:80]}")
    if r.get("atmosphere_keywords"):
        rows.append(f"  Atmosphere: {', '.join(str(a) for a in r['atmosphere_keywords'][:6])}")
    if r.get("visual_influences"):
        rows.append(f"  Influences: {', '.join(str(i) for i in r['visual_influences'][:4])}")
    if r.get("tiles"):
        rows.append(f"  Tiles: {len(r['tiles'])} reference frame(s)")
    return "\n".join(rows)


def _summarize_cinematography(r: dict) -> str:
    rows = [
        f"Style: {r.get('cinematography_style', '?')}",
        f"Movement: {r.get('dominant_movement', '?')}",
    ]
    for s in r.get("scenes", []):
        shots = s.get("shots", [])
        first = shots[0] if shots else {}
        shot_preview = (
            f"{first.get('type','')} {first.get('movement','')}".strip()
            if first else "—"
        )
        rows.append(
            f"  {s.get('scene_number','?'):>2}. {s.get('coverage','?')[:60]}"
            f"  [{len(shots)} shots, opens: {shot_preview}]"
        )
    return "\n".join(rows)


def _summarize_storyboard(r: dict) -> str:
    rows = [f"Board style: {r.get('storyboard_style', '?')}"]
    for s in r.get("storyboard", []):
        panels = s.get("panels") or s.get("frames", [])
        hdr = s.get("header", {})
        slugline = hdr.get("slugline") or s.get("scene_number", "?")
        purpose = hdr.get("purpose", "")
        dur = hdr.get("duration_estimate", "")
        rows.append(f"  Scene {s.get('scene_number','?'):>2}  {slugline}"
                    + (f"  [{dur}]" if dur else "") + f"  — {len(panels)} panel(s)")
        if purpose:
            rows.append(f"      purpose: {purpose[:80]}")
        vo = s.get("visual_overview", {})
        if vo.get("color_palette"):
            rows.append(f"      palette: {vo['color_palette'][:70]}")
        for p in panels[:3]:
            cam = (f"{p.get('shot_type','')} / {p.get('camera_angle','')} / "
                   f"{p.get('camera_movement','')}").strip(" /")
            rows.append(f"      p{p.get('panel', p.get('frame','?'))}  [{cam}]"
                        f"  {p.get('action', p.get('moment',''))[:55]}"
                        + (f"  — {p.get('emotional_note','')[:24]}" if p.get('emotional_note') else ""))
    return "\n".join(rows)


def _summarize_screenplay(r: dict) -> str:
    rows = [f"Drafted: {r.get('drafted_count', 0)} of {r.get('total_scenes', 0)} scenes"]
    for s in r.get("scenes", []):
        preview = s.get("fountain", "")[:200].replace("\n", " ↵ ")
        rows.append(f"  Scene {s.get('number','?')}: {preview} …")
    return "\n".join(rows)


# ── stop / resume support ─────────────────────────────────────────────────────

class PipelineStopped(Exception):
    """Raised when the operator pauses the run at a review gate."""

    def __init__(self, stage: str):
        super().__init__(f"stopped at stage '{stage}'")
        self.stage = stage


def _slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", (name or "").lower()).strip("_") or "character"


def _content_hash(*parts) -> str:
    """Short hash over prompt text (and, for image parts, file bytes) — lets a
    render step tell a stale asset (rendered from a prompt since revised via
    HITL feedback — e.g. `stage casting --feedback` or a post-resume rerun)
    apart from one that's still current, instead of trusting file existence
    alone."""
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            continue
        if isinstance(p, Path):
            try:
                h.update(p.read_bytes())
            except OSError:
                pass
        else:
            h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _stale(asset: Path, hash_path: Path, current_hash: str) -> bool:
    """True if `asset` is missing, or was rendered from a prompt that no longer
    matches (source stage revised since). An asset with no hash sidecar predates
    this tracking — accepted as the baseline rather than re-rendered."""
    if not asset.exists():
        return True
    if not hash_path.exists():
        return False
    return hash_path.read_text().strip() != current_hash


def _render_casting_images(casting: dict, out: Path,
                           active_names: set[str] | None = None) -> int:
    """Render ONE image per casting entry — character OR location representation —
    via the image backend. Kind-agnostic: a `kind: location` entry has no `actor`
    block, so the prompt lookup falls through to `character.visual_prompt`
    directly, same as any other entry. Capped to `active_names` when provided
    (character/location names that appear in the scenes being rendered).
    Idempotent by prompt hash: skips an entry whose rendered image still matches
    its current `visual_prompt`, but re-renders when feedback has revised that
    prompt since (--resume or a standalone `stage casting --feedback` rerun).
    """
    if not imagegen.available():
        _log(f"      casting renders skipped — {imagegen.unavailable_hint()}")
        return 0
    cast_dir = out / "casting"
    cast_dir.mkdir(exist_ok=True)
    n = 0
    for c in casting.get("casting", []):
        name = c.get("name", "")
        if active_names is not None and name not in active_names:
            _log(f"      {name} — not in active scenes; skipping render")
            continue
        slug = _slug(name or "character")
        kind = c.get("kind", "person")
        character = c.get("character") or {}
        target = character if character else c
        prompt = (character.get("visual_prompt")
                  or character.get("physical_form")
                  or c.get("visual_prompt") or c.get("physical_form")
                  or name)
        if not prompt:
            _log(f"      ⚠ {name} ({kind}) — no visual_prompt; skipping render")
            continue
        img = cast_dir / f"{slug}.png"
        hash_path = cast_dir / f"{slug}.hash"
        current_hash = _content_hash(prompt)
        if not _stale(img, hash_path, current_hash):
            target["image_path"] = str(img.relative_to(out))
            hash_path.write_text(current_hash)
            n += 1
            continue
        _log(f"      rendering {name} [{kind}] …"
             + (" (prompt revised — re-rendering)" if img.exists() else ""))
        if imagegen.generate_image(prompt, img):
            target["image_path"] = str(img.relative_to(out))
            hash_path.write_text(current_hash)
            n += 1
    return n


def _render_moodboard_tiles(moodboard: dict, out: Path) -> int:
    """Render the moodboard's reference `tiles` into images via the image backend
    (Gemini when a key is configured, else the open image backend) → output/
    moodboard/tile_NN.png. The moodboard's palette + lighting are appended to each
    tile prompt so the board coheres as one look. Stores the path on each tile.
    Idempotent (skips files on disk). Best-effort — never blocks the run.

    NB policy: only the moodboard *tiles* (images) use the image provider; the
    moodboard spec itself is generated on the open text models like every stage.
    Idempotent by prompt hash (see `_stale`): re-renders a tile if the moodboard
    was revised via feedback since it was last rendered."""
    tiles = moodboard.get("tiles") or []
    if not tiles:
        return 0
    if not imagegen.available():
        _log(f"      moodboard tiles skipped — {imagegen.unavailable_hint()}")
        return 0
    mdir = out / "moodboard"
    mdir.mkdir(exist_ok=True)
    look = ", ".join(x for x in [
        ", ".join(str(c) for c in (moodboard.get("palette") or [])[:4]),
        moodboard.get("lighting_mood", ""),
    ] if x)
    n = 0
    for i, tile in enumerate(tiles, start=1):
        prompt = tile.get("image_prompt") or tile.get("label")
        if not prompt:
            continue
        if look:
            prompt = f"{prompt}. Moodboard look: {look}."
        img = mdir / f"tile_{i:02d}.png"
        hash_path = mdir / f"tile_{i:02d}.hash"
        current_hash = _content_hash(prompt)
        if not _stale(img, hash_path, current_hash):
            tile["image_path"] = str(img.relative_to(out))
            hash_path.write_text(current_hash)
            n += 1
            continue
        if imagegen.generate_image(prompt, img):
            tile["image_path"] = str(img.relative_to(out))
            hash_path.write_text(current_hash)
            n += 1
    return n


# Veo prompting guide — lens/framing terms by shot type. "portrait" enhances
# facial detail on close-ups; "macro lens" suits inserts. Keyed by both
# abbreviations (storyboard schema) and natural language variants
# (cinematography agent / storyboard agent prose output).
_VEO_FOCUS: dict[str, str] = {
    # standard abbreviations
    "ECU": "portrait",
    "CU": "portrait",
    "INSERT": "macro lens",
    # natural language (model may output these) — covers both the
    # cinematography agent's spelled-out shot types and the storyboard
    # agent's own abbreviation schema, matching fountain.py's
    # _VEO_FOCUS_FOUNTAIN (the pair of dicts veo_guide.py tracks together).
    # Values carry ONLY the framing hint, not shot-type wording (e.g. not
    # "extreme close-up") — the shot-type label itself (_VEO_SHOT_LABEL in
    # pipeline.py / raw_type in fountain.py's _camera()) already supplies
    # that, so repeating it here would duplicate it in the final prompt.
    "EXTREME-CLOSE-UP": "portrait",
    "EXTREME CLOSE-UP": "portrait",
    "EXTREME CLOSE UP": "portrait",
    "CLOSE-UP": "portrait",
    "CLOSE UP": "portrait",
}

# Shot-type abbreviation -> natural-language label, for [Cinematography] in
# _panel_cinematography. Values already spelled out in full (e.g. "wide shot")
# pass through unchanged.
_VEO_SHOT_LABEL: dict[str, str] = {
    "ECU": "extreme close-up",
    "CU": "close-up",
    "MCU": "medium close-up",
    "MS": "medium shot",
    "FS": "full shot",
    "WS": "wide shot",
    "ELS": "extreme long shot",
    "2S": "two-shot",
    "OTS": "over-the-shoulder shot",
    "POV": "POV shot",
    "INSERT": "insert shot",
}


def _panel_video_prompt(panel: dict, audio_overview: dict | None = None, *,
                        casting_lookup: dict[str, dict] | None = None,
                        location_desc: str = "",
                        visual_overview: dict | None = None,
                        voice_index: dict[str, str] | None = None,
                        no_bg_music: bool = True,
                        room_tone: bool = True,
                        no_subtitles: bool = True) -> str:
    """Assemble a Veo-aligned prompt from a storyboard panel.

    Visual half ALWAYS follows this fixed five-part formula, in this order,
    assembled from an explicit dict (see _five_part_veo_prompt):
      [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
    Requires `casting_lookup` (name -> casting.json entry) to build [Subject]
    and [Context] from structured data; callers without it (e.g. a bare
    prompt with no casting/scene context) fall back to the storyboard
    agent's own free-text image_prompt as the visual half instead — that
    text has no guaranteed section order, so the fixed formula above is only
    guaranteed when casting_lookup is supplied.

    Audio half is built entirely via `veo_guide`'s construction helpers
    (`ambient_cue`/`sfx_cue`/`music_directive`/`dialogue_cue`/
    `no_subtitles_directive`) — that module owns the guide-vocabulary
    knowledge (which labels the guide actually confirms vs. which cases have
    no official keyword and need an unambiguous sentence instead), so this
    function only gathers the panel/scene data and hands it off. A future
    non-Veo backend can supply an equivalent module with the same call shape
    without touching this assembly logic.

    Veo generates audio per clip independently, with no memory of prior
    clips, so leaving any of these implicit invites drift across a scene:
    voice-over dialogue that gets lip-synced to an on-screen face, a
    hallucinated score that wasn't there before, room tone that suddenly
    gains an echo, or burned-in subtitle text. Spelling each one out
    explicitly — even the negative "no background music" / "no subtitles"
    cases — is what keeps the audio track consistent shot to shot (native
    audio generation itself cannot be disabled or added after the fact, so
    this only steers *what* it generates, not *whether* it does).

    voice_index — character name → vocal-quality description (from characters.voice),
    so the same character sounds the same across separately-generated clips (Veo
    has no cross-generation voice cloning; this is the manual substitute).
    """
    voice_index = voice_index or {}
    ao = audio_overview or {}

    # Ambient (environment) and score/music are kept as two SEPARATE cues (not
    # merged) — folding a score cue into "ambient" makes both drift together
    # when Veo hallucinates; keeping them apart lets the no-music directive
    # below cleanly override just the music half.
    panel_sound_raw = (panel.get("sound") or "").strip()
    if " | " in panel_sound_raw:
        panel_ambient, panel_sfx = [s.strip() for s in panel_sound_raw.split(" | ", 1)]
    else:
        panel_ambient, panel_sfx = "", panel_sound_raw

    ambient = panel_ambient or ao.get("ambient", "")
    sfx = panel_sfx
    if sfx and ambient and sfx.strip(".") == ambient.strip("."):
        sfx = ""

    # Anchor the room tone to consistent acoustics so it doesn't drift between
    # clips of the same scene (e.g. a sudden echo that wasn't there a shot ago).
    if room_tone and ambient and not any(
        t in ambient.lower() for t in ("echo", "reverb", "acoustic", "muffled", "hollow")
    ):
        ambient = f"{ambient.rstrip('.')}, dry acoustics, no echo"

    score = (ao.get("score_cue") or "").strip()

    dialogue_cues: list[str] = []
    for d in (panel.get("dialogue") or []):
        line = (d.get("line") or "").strip()
        if not line:
            continue
        modifier = (d.get("modifier") or "").strip().upper()
        speaker = (d.get("speaker") or "").strip()
        cue = veo_guide.dialogue_cue(
            speaker, line,
            voice=voice_index.get(speaker, ""),
            tone=(d.get("parenthetical") or "").strip().strip("()"),
            vo=bool(d.get("vo")),
            off_screen="O.S." in modifier,
        )
        if cue:
            dialogue_cues.append(cue)

    # Visual half — always the fixed five-part formula when casting_lookup is
    # available (the real render path always supplies it); otherwise fall
    # back to the storyboard agent's own free-text image_prompt.
    if casting_lookup is not None:
        visual = _five_part_veo_prompt(panel, casting_lookup=casting_lookup,
                                       location_desc=location_desc,
                                       visual_overview=visual_overview or {})
    else:
        visual = (panel.get("image_prompt") or panel.get("action")
                 or panel.get("moment", "")).strip()

    # Audio half: ambient → SFX → music/no-music → dialogue (+ no-subtitles).
    audio_bits: list[str] = []
    for cue in (veo_guide.ambient_cue(ambient), veo_guide.sfx_cue(sfx),
               veo_guide.music_directive(score, no_background_music=no_bg_music)):
        if cue:
            audio_bits.append(cue)
    audio_bits.extend(dialogue_cues)
    if dialogue_cues and no_subtitles:
        audio_bits.append(veo_guide.no_subtitles_directive())

    return (visual + (" " + " ".join(audio_bits) if audio_bits else "")).strip()


def _panel_dialogue_lines(panel: dict) -> list[str]:
    """Format dialogue entries from a storyboard panel as subtitle strings."""
    lines = []
    for d in (panel.get("dialogue") or []):
        speaker = (d.get("speaker") or "").strip()
        line = (d.get("line") or "").strip()
        if speaker and line:
            lines.append(f"{speaker}: {line}")
        elif line:
            lines.append(line)
    return lines


def _write_scene_prompt_log(out: Path, snum, model: str, seed_note: str,
                            frame_logs: list[dict]) -> None:
    """Plain-text log of the exact Veo prompt behind each clip in a scene —
    always reflects the CURRENT set of frames that make up the scene's video
    (re-rendered or skipped-as-unchanged alike), so it stays a truthful record
    even when most frames were cached from an earlier run."""
    logs_dir = out / "logs"
    logs_dir.mkdir(exist_ok=True)
    tag = f"{int(snum):02d}" if isinstance(snum, int) else str(snum)
    lines = [
        f"Scene {snum} — Veo render log",
        f"Session: {session.current(out) or '-'}",
        f"Model: {model}",
        f"Seed method: {seed_note}",
        "",
    ]
    for fl in frame_logs:
        lines.append("=" * 80)
        lines.append(f"FRAME {fl['tag']}  ->  {fl['clip'] or '(not rendered)'}")
        lines.append("-" * 80)
        lines.append(fl["prompt"])
        lines.append("")
    lines.append("=" * 80)
    (logs_dir / f"scene_{tag}_veo_prompts.txt").write_text("\n".join(lines), encoding="utf-8")


def _panel_cinematography(panel: dict) -> str:
    """[Cinematography] — camera work and shot composition, built fresh from
    the panel's own structured fields (shot_type/camera_angle/camera_movement/
    lens), plus the shot-type-appropriate framing hint (_VEO_FOCUS: "portrait"
    for close-ups, "macro lens" for inserts — depth-of-field is intentionally
    not asserted here, see _VEO_FOCUS's own docstring)."""
    shot = (panel.get("shot_type") or "").strip()
    angle = (panel.get("camera_angle") or "").strip()
    movement = (panel.get("camera_movement") or "").strip()
    lens = (panel.get("lens") or "").strip()
    bits: list[str] = []
    if shot:
        label = _VEO_SHOT_LABEL.get(shot.upper())
        if not label:
            base_label = shot.capitalize() if shot.isupper() else shot
            label = base_label if base_label.lower().endswith(("shot", "up", "view")) else f"{base_label} shot"
        bits.append(label)
    if angle:
        bits.append(angle.lower())
    if movement:
        bits.append("static camera" if movement.lower() == "static" else movement.lower())
    if lens:
        bits.append(lens if "lens" in lens.lower() else f"{lens} lens")

    hint_terms = _VEO_FOCUS.get(shot.upper(), "")
    if hint_terms:
        joined_lower = ", ".join(bits).lower()
        key = hint_terms.split(",")[0].strip()
        if key not in joined_lower:
            bits.append(hint_terms)

    return ", ".join(b for b in bits if b)


def _panel_subject(panel: dict, casting_lookup: dict[str, dict]) -> str:
    """[Subject] — the locked on-screen look of every character in frame,
    from casting.json's character.physical_form (falls back to the bare name
    if casting has no matching entry, e.g. an uncast background figure)."""
    names = panel.get("characters_in_frame") or []
    subjects: list[str] = []
    for name in names:
        entry = casting_lookup.get(name)
        if not entry:
            subjects.append(name)
            continue
        ch = entry.get("character", entry)
        form = (ch.get("physical_form") or "").strip()
        subjects.append(f"{name} ({form})" if form else name)
    return " and ".join(subjects)


def _panel_context(panel: dict, location_desc: str) -> str:
    """[Context] — environment and background elements: the scene's locked
    location (from casting.json's location entry) plus this panel's own
    composition note (who/what is where in frame, depth layers)."""
    bits = []
    if location_desc:
        bits.append(location_desc.rstrip("."))
    composition = (panel.get("composition") or "").strip()
    if composition:
        bits.append(composition.rstrip("."))
    return "; ".join(bits)


def _panel_style_ambiance(visual_overview: dict) -> str:
    """[Style & Ambiance] — overall aesthetic, mood, and lighting: a fixed
    Veo style keyword plus the scene's visual_overview (color_palette/
    lighting_setup/mood)."""
    vo = visual_overview or {}
    bits = ["Cinematic, photorealistic"]
    bits += [b.strip() for b in
            (vo.get("color_palette", ""), vo.get("lighting_setup", ""), vo.get("mood", ""))
            if b.strip()]
    return ". ".join(b.rstrip(".") for b in bits if b) + "."


def _five_part_veo_prompt(panel: dict, *, casting_lookup: dict[str, dict],
                          location_desc: str, visual_overview: dict) -> str:
    """Assemble the VISUAL half of a Veo prompt as an explicit, fixed-order
    dict, per the five-part formula:
      [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
    Each section is built fresh from structured data (casting.json, the
    scene's visual_overview, the panel's own fields) rather than the
    storyboard agent's free-text image_prompt — the model's own prose has no
    guaranteed internal ordering, so reordering an opaque blob after the fact
    isn't reliable; reconstructing from structured fields is."""
    parts: dict[str, str] = {
        "cinematography": _panel_cinematography(panel),
        "subject": _panel_subject(panel, casting_lookup),
        "action": (panel.get("action") or panel.get("moment") or "").strip(),
        "context": _panel_context(panel, location_desc),
        "style_ambiance": _panel_style_ambiance(visual_overview),
    }
    return " ".join(f"{v.rstrip('.')}." for v in parts.values() if v.strip())


def _frame_char_anchor(frame: dict, cast_index: dict, out: Path) -> Path | None:
    """The casting image of the first in-frame character — Veo identity seed."""
    for name in frame.get("characters_in_frame", []):
        rel = cast_index.get(name)
        if rel and (out / rel).exists():
            return out / rel
    return None


def _render_scene_frames(storyboard: dict, casting: dict, out: Path,
                         max_scenes: int | None = None,
                         characters: dict | None = None) -> dict:
    """Render each storyboard frame as a video clip (Veo image-to-video), then
    stitch each scene's clips into a per-scene video (output/video/scene_NN.mp4)
    and assemble all scene videos into the final movie (output/video/movie.mp4).

    Identity seeding: the first frame of each scene seeds from the in-frame
    character's representation image; subsequent frames chain from the previous
    clip's last frame for continuity within the scene. Scene boundary = hard cut.

    `characters` (optional, from characters.json) supplies each speaking
    character's vocal-quality description so dialogue cues stay consistent
    across separately-generated clips (see `_panel_video_prompt`).

    max_scenes caps how many scenes are rendered; every shot within each rendered
    scene is always included. Best-effort + idempotent (skips existing files;
    re-renders when a feedback revision changed the prompt — see `_stale`).
    """
    if not i2v.enabled():
        _log(f"      scene render skipped — {i2v.unavailable_hint()}")
        return {}
    if not i2v.available():
        _log(f"      scene render skipped — {i2v.unavailable_hint()}")
        return {}
    vcfg = i2v._cfg()
    continuity = bool(vcfg.get("continuity", True))
    audio_cfg = vcfg.get("audio", {})
    no_bg_music = bool(audio_cfg.get("no_background_music", True))
    room_tone = bool(audio_cfg.get("room_tone", True))
    no_subtitles = bool(audio_cfg.get("no_subtitles", True))

    cast_index = {}
    casting_lookup: dict[str, dict] = {}
    for c in casting.get("casting", []):
        ch = c.get("character", c)
        rel = ch.get("image_path") or c.get("image_path")
        if rel:
            cast_index[c.get("name")] = rel
        casting_lookup[c.get("name")] = c

    voice_index = {
        c.get("name"): c.get("voice", "")
        for c in (characters or {}).get("characters", [])
        if c.get("name") and c.get("voice")
    }

    vdir = out / "video"
    vdir.mkdir(exist_ok=True)
    manifest = {"continuity": continuity, "clips": 0, "failed": 0, "scenes": []}

    board = storyboard.get("storyboard", [])
    if max_scenes:
        board = board[:max_scenes]              # limit scenes, never the shots within
    for scene in board:
        snum = scene.get("scene_number", "x")
        sdir = vdir / (f"scene_{snum:02d}" if isinstance(snum, int) else f"scene_{snum}")
        sdir.mkdir(exist_ok=True)
        prev_tail = None                        # reset each scene → hard cut between scenes
        prev_clip_path = None                   # previous clip mp4 — for continuity_mode: extend
        frames_out = []
        prompt_log: list[dict] = []
        # scene-level audio overview for panels that have no explicit sound field
        audio_overview = scene.get("audio_overview") or {}
        # `panels` is the new schema; fall back to `frames` for old checkpoints
        panels = scene.get("panels") or scene.get("frames", [])

        # [Context] source: the scene's locked location, from casting.json's
        # kind:"location" entry (its own visual_prompt) — falls back to the
        # bare location name if this scene's location wasn't cast.
        loc_name = (scene.get("header", {}).get("location") or "").strip()
        loc_entry = casting_lookup.get(loc_name, {})
        loc_ch = loc_entry.get("character", loc_entry)
        location_desc = loc_ch.get("visual_prompt") or loc_name
        # [Style & Ambiance] source: the scene's own color/lighting/mood.
        visual_overview = scene.get("visual_overview") or {}

        for fr in panels:
            fnum = fr.get("panel") or fr.get("frame", len(frames_out) + 1)
            prompt = _panel_video_prompt(fr, audio_overview,
                                         casting_lookup=casting_lookup,
                                         location_desc=location_desc,
                                         visual_overview=visual_overview,
                                         voice_index=voice_index,
                                         no_bg_music=no_bg_music, room_tone=room_tone,
                                         no_subtitles=no_subtitles)
            tag = f"{int(fnum):02d}" if isinstance(fnum, int) else str(fnum)

            # Seed: continue from the previous frame's tail (carries the look
            # forward); the first frame of a scene seeds from the in-frame
            # character's representation image (identity reference).
            seed = prev_tail if (prev_tail and continuity) else _frame_char_anchor(fr, cast_index, out)
            clip = sdir / f"frame_{tag}.mp4"
            tail_img = sdir / f"frame_{tag}_tail.png"
            hash_path = sdir / f"frame_{tag}.hash"
            # Hash covers both the prompt AND the seed image bytes: a feedback
            # revision to an earlier frame changes its tail frame, which changes
            # this frame's seed, which — even with an unchanged prompt — must
            # still invalidate this clip so continuity re-chains correctly.
            current_hash = _content_hash(prompt, seed, prev_clip_path)
            if _stale(clip, hash_path, current_hash):
                if clip.exists():
                    _log(f"      scene {snum} frame {tag} — prompt/seed revised, re-rendering …")
                if i2v.generate_clip([seed] if seed else [], prompt, clip, prev_clip=prev_clip_path):
                    manifest["clips"] += 1
                    # Burn subtitle + shot-label overlays onto the clip (in-place)
                    # when the operator enables them in config video.overlays.
                    if i2v.overlays_enabled():
                        dlg = _panel_dialogue_lines(fr)
                        stype = fr.get("shot_type") or ""
                        shot_lbl = f"S{snum}·F{tag}" + (f"·{stype}" if stype else "")
                        i2v.add_overlays(clip, clip,
                                         dialogue_lines=dlg or None,
                                         shot_label=shot_lbl)
                    # Always extract the tail frame so every clip has one on disk.
                    # Continuity chains it forward as the next-clip seed;
                    # ffmpeg stitching benefits from having clean cut-points regardless.
                    tail = i2v.last_frame(clip, tail_img)
                    hash_path.write_text(current_hash)
                    if continuity:
                        prev_tail = tail or seed
                        prev_clip_path = clip
                else:
                    manifest["failed"] += 1
                    _log(f"      ⚠ scene {snum} frame {tag} — clip not produced")
            elif continuity:
                # Already up to date — still chain forward from its tail frame
                # (previously this branch left prev_tail untouched, so a resumed
                # run with some frames already rendered would reset newer frames
                # to the character anchor instead of continuing the scene).
                prev_tail = tail_img if tail_img.exists() else seed
                prev_clip_path = clip

            frames_out.append({
                "panel": fnum,
                "shot_type": fr.get("shot_type", ""),
                "action": fr.get("action") or fr.get("moment", ""),
                "seed": str(Path(seed).relative_to(out)) if seed and Path(seed).exists() else None,
                "clip": str(clip.relative_to(out)) if clip.exists() else None,
            })
            prompt_log.append({
                "tag": tag,
                "prompt": prompt,
                "clip": str(clip.relative_to(out)) if clip.exists() else None,
            })

        _write_scene_prompt_log(
            out, snum, vcfg.get("model", "unknown"),
            "tail-frame continuity (previous clip's last frame)" if continuity
            else "character/location identity anchor per frame (no continuity)",
            prompt_log,
        )

        # Stitch this scene's frame clips into a scene-level video.
        scene_vid = vdir / (f"scene_{snum:02d}.mp4" if isinstance(snum, int) else f"scene_{snum}.mp4")
        scene_vid_rel: str | None = None
        if scene_vid.exists():
            scene_vid_rel = str(scene_vid.relative_to(out))   # already done (resume)
        else:
            scene_clips = [out / fr["clip"] for fr in frames_out
                           if fr.get("clip") and (out / fr["clip"]).exists()]
            if scene_clips:
                if i2v.stitch(scene_clips, scene_vid):
                    scene_vid_rel = str(scene_vid.relative_to(out))
                    _log(f"      scene {snum}: stitched {len(scene_clips)} clip(s) → {scene_vid.name}")
                else:
                    _log(f"      ⚠ scene {snum}: per-scene stitch failed")

        manifest["scenes"].append({
            "scene_number": snum,
            "frames": frames_out,
            "scene_video": scene_vid_rel,
        })

    # Final assembly: stitch scene videos (preferred) or raw clips into movie.mp4.
    movie = _assemble_movie(manifest, out)
    if movie:
        manifest["movie"] = str(movie.relative_to(out))

    (vdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def _scene_videos_in_order(manifest: dict, out: Path) -> list[Path]:
    """Per-scene stitched video paths in playback order."""
    videos: list[Path] = []
    for scene in sorted(manifest.get("scenes", []),
                        key=lambda s: s.get("scene_number") if isinstance(s.get("scene_number"), int) else 1e9):
        rel = scene.get("scene_video")
        if rel and (out / rel).exists():
            videos.append(out / rel)
    return videos


def _clips_in_order(manifest: dict, out: Path) -> list[Path]:
    """Individual frame clip paths in playback order (fallback for final stitch)."""
    clips: list[Path] = []
    for scene in sorted(manifest.get("scenes", []),
                        key=lambda s: s.get("scene_number") if isinstance(s.get("scene_number"), int) else 1e9):
        for fr in sorted(scene.get("frames", []),
                         key=lambda f: f.get("frame") if isinstance(f.get("frame"), int) else 1e9):
            rel = fr.get("clip")
            if rel and (out / rel).exists():
                clips.append(out / rel)
    return clips


def _assemble_movie(manifest: dict, out: Path) -> Path | None:
    """Concatenate into output/video/movie.mp4. Prefers per-scene videos (cleaner
    seams at scene boundaries); falls back to raw frame clips. Best-effort."""
    movie = out / "video" / "movie.mp4"
    scene_vids = _scene_videos_in_order(manifest, out)
    if scene_vids:
        _log(f"      assembling movie from {len(scene_vids)} scene video(s) …")
        return movie if i2v.stitch(scene_vids, movie) else None
    clips = _clips_in_order(manifest, out)
    if not clips:
        return None
    _log(f"      assembling movie from {len(clips)} frame clip(s) …")
    return movie if i2v.stitch(clips, movie) else None


def _checkpoint_load(out: Path, name: str) -> dict | None:
    """Load a previously-approved stage artifact, or None if absent/unreadable."""
    f = out / f"{name}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _spec(name: str, compute: Callable, summarize: Callable, rerun: Callable) -> dict:
    """Describe one stage: how to compute it, summarize it, and re-run it."""
    return {"name": name, "compute": compute, "summarize": summarize, "rerun": rerun}


# ── gate loop helper ──────────────────────────────────────────────────────────

def _format_fidelity(rep: dict | None, min_score: int = 70) -> str:
    """One-block fidelity readout for the review gate (score + a hint to re-run)."""
    if not rep:
        return ""
    score = rep.get("fidelity_score")
    verdict = (rep.get("verdict") or "?").upper()
    line = f"\n  story fidelity: {verdict}  {score}/100"
    if isinstance(score, (int, float)) and score < min_score:
        line += f"  ⚠ below {min_score} — consider re-running with feedback"
    issues = (rep.get("drift") or []) + (rep.get("contradictions") or [])
    if issues:
        line += "\n    drift: " + "; ".join(str(i) for i in issues[:2])
    return line


def _format_genre(rep: dict | None, min_score: int = 70) -> str:
    """One-block genre-alignment readout for the review gate."""
    if not rep:
        return ""
    score = rep.get("genre_score")
    verdict = (rep.get("verdict") or "?").upper()
    name = rep.get("genre") or "?"
    line = f"\n  genre [{name}]: {verdict}  {score}/100"
    if isinstance(score, (int, float)) and score < min_score:
        line += f"  ⚠ below {min_score} — consider re-running with feedback"
    issues = (rep.get("off_genre") or []) + (rep.get("missing_conventions") or [])
    if issues:
        line += "\n    off-genre: " + "; ".join(str(i) for i in issues[:2])
    return line


def _model_label(profile: str | None) -> str:
    """'profile / resolved-model' string for display; graceful on lookup failure."""
    if not profile:
        return ""
    try:
        model = llm.resolve_model(llm.get_profile(profile))
        return f"{profile} / {model}"
    except Exception:
        return profile


def _gated(
    gate: Gate,
    name: str,
    initial_result: dict,
    summarize_fn: Callable,
    rerun_fn: Callable,             # rerun_fn(feedback: str, profile: str | None) -> dict
    fidelity_fn: Callable | None = None,
    min_score: int = 70,
    genre_fn: Callable | None = None,
    genre_min: int = 70,
    profile: str | None = None,     # resolved profile name (display + escalation)
    escalate_after: int = 3,        # consecutive low-score reruns before gradual escalation
    escalate_score_gap: int = 20,   # escalate immediately when score is this far below threshold
) -> tuple[dict, dict | None, dict | None]:
    """Show gate for initial_result; re-run with feedback until approved.

    Two escalation paths (fast → quality → quality_high):

    1. Immediate: if fidelity OR genre is more than `escalate_score_gap` points below
       its threshold on the rerun just requested, switch to the next profile right away
       (don't wait for multiple attempts — a huge gap means the current model clearly
       can't handle this stage).

    2. Gradual: if scores are consistently below threshold for `escalate_after`
       consecutive reruns, escalate to the next profile tier.

    Both paths reset the counter on escalation. If already at quality_high, a clear
    message is logged instead. Rerun lambdas must accept (feedback, profile=None).

    Returns (approved_result, fidelity_report, genre_report). Raises PipelineStopped.
    """
    result = initial_result
    current_profile = profile
    low_score_run = 0
    iteration = 0

    while True:
        report = fidelity_fn(result) if fidelity_fn else None
        grep = genre_fn(result) if genre_fn else None

        iter_label = "initial" if iteration == 0 else f"iteration {iteration + 1}"
        _log(f"      {name}  [{_model_label(current_profile)}]  ({iter_label})")

        def _summary(r, _rep=report, _g=grep):
            return summarize_fn(r) + _format_fidelity(_rep, min_score) + _format_genre(_g, genre_min)

        decision = gate.review(name, result, _summary)
        if decision.approved:
            return result, report, grep
        if decision.stop:
            raise PipelineStopped(name)

        if decision.edited is not None:
            # A manual edit is a new candidate, not an approval: adopt it and loop
            # back to the top — fidelity_fn/genre_fn re-score it fresh and the same
            # gate (approve / feedback / view again / stop) reappears. Skips
            # rerun_fn/escalation below entirely; no model call, no profile change.
            _log(f"      [{name}] manual edit applied — re-checking fidelity/genre …")
            result = decision.edited
            iteration += 1
            continue

        # Extract numeric scores (None when a checker wasn't run / returned no score).
        fid_score = report.get("fidelity_score") if report else None
        gen_score = grep.get("genre_score") if grep else None
        fid_score = fid_score if isinstance(fid_score, (int, float)) else None
        gen_score = gen_score if isinstance(gen_score, (int, float)) else None

        fid_low  = fid_score is not None and fid_score < min_score
        gen_low  = gen_score is not None and gen_score < genre_min
        fid_huge = fid_score is not None and escalate_score_gap > 0 and fid_score < min_score - escalate_score_gap
        gen_huge = gen_score is not None and escalate_score_gap > 0 and gen_score < genre_min - escalate_score_gap

        def _try_escalate(reason: str) -> bool:
            """Promote current_profile one tier; log and return True if escalated."""
            nonlocal current_profile, low_score_run
            if not current_profile:
                return False
            next_p = llm.next_profile(current_profile)
            if next_p:
                _log(f"      ↑ [{name}] {reason} — escalating {current_profile} → {next_p}")
                current_profile = next_p
                low_score_run = 0
                return True
            _log(f"      [{name}] {reason} but already at top profile ({current_profile}); "
                 f"try stronger feedback or edit manually")
            return False

        if fid_huge or gen_huge:
            # Path 1 — immediate: score is massively off, don't wait.
            parts = []
            if fid_huge:
                parts.append(f"fidelity {fid_score}/100 (>{escalate_score_gap} pts below {min_score})")
            if gen_huge:
                parts.append(f"genre {gen_score}/100 (>{escalate_score_gap} pts below {genre_min})")
            _try_escalate(f"huge misalignment ({'; '.join(parts)})")
        elif fid_low or gen_low:
            # Path 2 — gradual: accumulate consecutive low-score reruns.
            low_score_run += 1
            if low_score_run >= escalate_after:
                _try_escalate(f"{low_score_run} consecutive low-score iterations")
        else:
            low_score_run = 0   # scores recovered; reset counter

        iteration += 1
        _log(f"      re-running {name}  [{_model_label(current_profile)}]  with feedback …")
        llm.unload_model(current_profile)
        result = rerun_fn(decision.feedback, current_profile)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    input_path: str,
    out_dir: str = "output",
    max_scenes: int | None = 1,
    profile_override: str | None = None,
    resume: bool = False,
    genre: str | None = None,
) -> dict:
    """Run the full screenplay-material phase and write artifacts to `out_dir`.

    Pause anytime by typing 'stop' at a review gate (or Ctrl-C); every stage
    already approved stays on disk. Re-run with `resume=True` to load those
    checkpoints and continue from the first stage that hasn't been completed.
    """
    # On a fresh run, clear any direction left over from a previous run that may
    # have crashed before reaching the llm.set_direction(None) at the end.
    # On a resumed run, leave it alone — genre and moodboard checkpoints are loaded
    # shortly below and apply_direction() re-establishes the correct direction
    # before any creative stage runs.
    if not resume:
        llm.set_direction(None)

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    session_id = session.start(out, source=input_path, fresh=not resume)
    _log(f"session {session_id} → {out}/session.json")
    gemini.set_log_dir(out)

    # Persist each stage as soon as it's approved, so a failure, timeout, or pause
    # in a later (slow) stage never discards completed work.
    def save(name: str, data: dict) -> None:
        _write_json(out / f"{name}.json", data)

    gate = Gate.from_config(llm.config())
    parallel = llm.config().get("runtime", {}).get("max_parallel_agents", 1) > 1

    # Per-stage fidelity: after each stage is approved, check its output stays
    # consistent with the original story (open model, per policy). Toggle via
    # config `fidelity.per_stage`.
    fid_cfg = llm.config().get("fidelity", {})
    fid_on = bool(fid_cfg.get("per_stage", True))
    fid_min = int(fid_cfg.get("min_score", 70))
    fid_reports: dict = {}
    _FID_STAGES = {"structure", "characters", "scenes", "casting", "soundscape",
                   "visuals", "cinematography", "screenplay", "storyboard"}

    def fidelity_report(name: str, result: dict) -> dict | None:
        """Score this stage's output against the original story (open model).
        Computed BEFORE the gate so the operator sees the score when deciding
        whether to re-iterate. Best-effort — never blocks the pipeline."""
        if not fid_on or name not in _FID_STAGES:
            return None
        try:
            return fidelity.check_stage(name, result, source.get("text", ""))
        except Exception as e:
            _log(f"      fidelity[{name}] skipped ({type(e).__name__})")
            return None

    def save_fidelity(name: str, rep: dict | None) -> None:
        if rep is None:
            return
        fid_reports[name] = rep
        fdir = out / "fidelity"
        fdir.mkdir(exist_ok=True)
        _write_json(fdir / f"{name}.json", rep)
        _log(f"      fidelity[{name}]: {rep.get('verdict', '?')} "
             f"{rep.get('fidelity_score', '?')}/100")

    # Genre: fix ONE genre for the run (CLI > config value > auto from storyline),
    # STEER every creative stage with it (llm.set_direction), and ENFORCE alignment
    # per stage (open model, per policy). Toggle via config `genre.{steer,enforce}`.
    rt_cfg = llm.config().get("runtime", {})
    escalate_after = int(rt_cfg.get("escalate_after", 3))
    escalate_score_gap = int(rt_cfg.get("escalate_score_gap", 20))
    gen_cfg = llm.config().get("genre", {})
    gen_enforce = bool(gen_cfg.get("enforce", True))
    gen_steer = bool(gen_cfg.get("steer", True))
    gen_min = int(gen_cfg.get("min_score", 70))
    genre_spec: dict = {}
    gen_reports: dict = {}

    # Moodboard: the film-wide visual-tone bible, fixed after structure and folded
    # into the steering direction so every creative stage composes toward one look.
    mood_cfg = llm.config().get("moodboard", {})
    mood_on = bool(mood_cfg.get("enabled", True))
    mood_steer = bool(mood_cfg.get("steer", True))
    moodboard: dict = {}
    _GENRE_STAGES = _FID_STAGES | {"moodboard"}

    def apply_direction() -> None:
        """Compose the shared creative direction from genre + moodboard and steer
        all subsequent creative generations with it (graders stay neutral)."""
        parts = []
        if gen_steer and genre_spec:
            parts.append(genre_agent.guidance(genre_spec))
        if mood_steer and moodboard:
            parts.append(moodboard_guidance(moodboard))
        llm.set_direction("\n\n".join(p for p in parts if p) or None)

    def genre_report(name: str, result: dict) -> dict | None:
        """Score this stage's output against the chosen genre (open model, neutral).
        Computed BEFORE the gate so the operator sees alignment when deciding."""
        if not gen_enforce or not genre_spec or name not in _GENRE_STAGES:
            return None
        try:
            return genre_agent.enforce_stage(name, result, genre_spec)
        except Exception as e:
            _log(f"      genre[{name}] skipped ({type(e).__name__})")
            return None

    def save_genre(name: str, rep: dict | None) -> None:
        if rep is None:
            return
        gen_reports[name] = rep
        gdir = out / "genre"
        gdir.mkdir(exist_ok=True)
        _write_json(gdir / f"{name}.json", rep)
        _log(f"      genre[{name}]: {rep.get('verdict', '?')} "
             f"{rep.get('genre_score', '?')}/100")

    def run_group(label_num: str, label: str, specs: list[dict]) -> dict:
        """Compute/gate/save a set of stages, loading any already-checkpointed.

        Cached members (present on disk when resuming) skip both compute and the
        gate. Remaining members compute concurrently when the host allows it,
        then gate sequentially. Returns {name: approved_result}.
        """
        loaded = {s["name"]: c for s in specs
                  if resume and (c := _checkpoint_load(out, s["name"])) is not None}
        pending = [s for s in specs if s["name"] not in loaded]

        if not pending:
            _log(f"{label_num} {label} — resumed from checkpoints")
            return {s["name"]: loaded[s["name"]] for s in specs}

        concurrent = parallel and len(pending) > 1
        note = f"  [resumed: {', '.join(loaded)}]" if loaded else ""
        _log(f"{label_num} {label}{' (concurrent)' if concurrent else ''}{note} …")
        for s in pending:
            pname = profile_override or llm.agent_profile(s["name"])
            _log(f"        {s['name']}: {_model_label(pname)}")

        raws: dict = {}
        if concurrent:
            with ThreadPoolExecutor(max_workers=len(pending)) as ex:
                futs = {ex.submit(s["compute"]): s["name"] for s in pending}
                for fut in futs:
                    raws[futs[fut]] = fut.result()
        else:
            for s in pending:
                raws[s["name"]] = s["compute"]()

        results = {}
        for s in specs:
            nm = s["name"]
            if nm in loaded:
                results[nm] = loaded[nm]
                continue
            stage_profile = profile_override or llm.agent_profile(nm)
            fid_fn = (lambda res, _nm=nm: fidelity_report(_nm, res)) \
                if (fid_on and nm in _FID_STAGES) else None
            gen_fn = (lambda res, _nm=nm: genre_report(_nm, res)) \
                if (gen_enforce and nm in _GENRE_STAGES) else None
            r, rep, grep = _gated(gate, nm, raws[nm], s["summarize"], s["rerun"],
                                  fidelity_fn=fid_fn, min_score=fid_min,
                                  genre_fn=gen_fn, genre_min=gen_min,
                                  profile=stage_profile, escalate_after=escalate_after,
                                  escalate_score_gap=escalate_score_gap)
            save(nm, r)
            save_fidelity(nm, rep)
            save_genre(nm, grep)
            results[nm] = r
        return results

    def _resolved(tier: str) -> str:
        try:
            return llm.resolve_model(llm.get_profile(profile_override or tier))
        except Exception:
            return "(unavailable)"
    fast_model = _resolved("fast")
    quality_model = _resolved("quality")
    synthesis_model = _resolved("synthesis")
    quality_high_model = _resolved("quality_high")
    _log(f"models — fast: {fast_model} | quality: {quality_model} | "
         f"synthesis: {synthesis_model} | quality_high: {quality_high_model}")
    if resume:
        _log(f"resume: loading any completed stages from {out}/")

    # ── 1/10  ingest (deterministic — no gate) ───────────────────────────────
    _log("1/10 ingest …")
    source = ingest(input_path)
    save("source", source)   # checkpoint so source-dependent stages can run standalone
    _log(f"      '{source['title']}' — {source['word_count']} words")

    # Fix the genre once (CLI > config value > auto from storyline) before any
    # creative stage, then steer every stage with it. Reuses a checkpoint on resume.
    genre_loaded = _checkpoint_load(out, "genre") if resume else None
    if genre_loaded:
        genre_spec = genre_loaded
        _log(f"      genre: {genre_spec.get('genre', '?')} (resumed)")
    elif gen_steer or gen_enforce:
        try:
            genre_spec = genre_agent.resolve_genre(
                source.get("text", ""), explicit=genre,
                config_value=gen_cfg.get("value"), profile=profile_override)
            save("genre", genre_spec)
            label = genre_spec.get("genre", "?")
            if genre_spec.get("subgenre"):
                label += f" / {genre_spec['subgenre']}"
            _log(f"      genre: {label} ({genre_spec.get('source', 'auto')})")
        except Exception as e:
            _log(f"      genre resolution skipped ({type(e).__name__}: {e})")
    apply_direction()   # steer with genre now (moodboard joins after its stage)

    # ── 2/10  structure + characters ─────────────────────────────────────────
    g = run_group("2/10", "structure ‖ characters", [
        _spec("structure",
              lambda: analyze_structure(source, profile_override),
              _summarize_structure,
              lambda fb, p=None: analyze_structure(source, p or profile_override, feedback=fb)),
        _spec("characters",
              lambda: extract_characters(source, profile_override),
              _summarize_characters,
              lambda fb, p=None: extract_characters(source, p or profile_override, feedback=fb)),
    ])
    structure, characters = g["structure"], g["characters"]
    _log(f"      logline: {structure.get('logline', '(parse failed)')[:80]}")
    _log(f"      characters: {len(characters.get('characters', []))}")

    # ── moodboard (film-wide visual-tone bible) — set once, steers all below ──
    if mood_on:
        g = run_group("moodboard", "moodboard", [
            _spec("moodboard",
                  lambda: design_moodboard(structure, source.get("text", ""), genre_spec,
                                           max_scenes=max_scenes, profile=profile_override),
                  _summarize_moodboard,
                  lambda fb, p=None: design_moodboard(structure, source.get("text", ""), genre_spec,
                                                       max_scenes=max_scenes, profile=p or profile_override,
                                                       feedback=fb)),
        ])
        moodboard = g["moodboard"]
        _log(f"      moodboard: {moodboard.get('overall_aesthetic', '?')[:80]}")
        n_tiles = len(moodboard.get("tiles") or [])
        if n_tiles:
            _log(f"      moodboard: {n_tiles} tile(s) kept as text cues for storyboard")
        apply_direction()   # fold the moodboard into the steering for every stage below

    # ── 3/10  scenes (scenes←structure) ────────────────────────────────────────
    g = run_group("3/10", "scenes", [
        _spec("scenes",
              lambda: segment_scenes(source, structure, profile=profile_override),
              _summarize_scenes,
              lambda fb, p=None: segment_scenes(source, structure, profile=p or profile_override, feedback=fb)),
    ])
    scenes = g["scenes"]
    n_dropped = len(scenes.get("dropped_scenes") or [])
    _log(f"      {len(scenes.get('scenes', []))} scenes"
        + (f"  ({n_dropped} dropped — source_line not found)" if n_dropped else ""))

    # ── 4/10  casting (casting←characters, scenes — locations need scenes' scene→
    # location mapping, so casting now runs after scenes rather than concurrently) ─
    g = run_group("4/10", "casting", [
        _spec("casting",
              lambda: cast_characters(structure, characters, profile_override, scenes=scenes),
              _summarize_casting,
              lambda fb, p=None: cast_characters(structure, characters, p or profile_override,
                                                 feedback=fb, scenes=scenes)),
    ])
    casting = g["casting"]
    _log(f"      cast {len(casting.get('casting', []))}")

    # Render character + location portraits only for names appearing in the scenes
    # that will actually be rendered (1..max_scenes). Capping here avoids burning
    # API quota on characters/locations the video stage will never reference.
    if imagegen.enabled():
        active_names: set[str] = set()
        for sc in scenes.get("scenes", [])[:max_scenes]:
            for nm in (sc.get("characters") or []):
                active_names.add(nm)
            if sc.get("location"):
                active_names.add(sc["location"])
        _log(f"      rendering character/location portraits for {len(active_names)} "
             f"name(s) in scene(s) 1..{_scenes_label(max_scenes)} …")
        if _render_casting_images(casting, out, active_names=active_names or None):
            save("casting", casting)
            _log(f"      portraits → {out}/casting/")

    # ── 5–7/10  soundscape + visuals + cinematography ────────────────────────
    g = run_group("5/10", "soundscape ‖ visuals ‖ cinematography", [
        _spec("soundscape",
              lambda: design_soundscape(structure, scenes, profile_override),
              _summarize_soundscape,
              lambda fb, p=None: design_soundscape(structure, scenes, p or profile_override, feedback=fb)),
        _spec("visuals",
              lambda: design_visuals(structure, scenes, profile_override),
              _summarize_visuals,
              lambda fb, p=None: design_visuals(structure, scenes, p or profile_override, feedback=fb)),
        _spec("cinematography",
              lambda: plan_cinematography(structure, scenes, profile_override),
              _summarize_cinematography,
              lambda fb, p=None: plan_cinematography(structure, scenes, p or profile_override, feedback=fb)),
    ])
    soundscape, visuals, cinematography = g["soundscape"], g["visuals"], g["cinematography"]

    # ── 8/10  screenplay draft — NOT capped by max_scenes: like soundscape/
    # visuals/cinematography/storyboard, screenplay drafting is a design stage,
    # not a rendering stage, so it stays aligned with the full scene list and
    # drafts every scene by default. Only casting-image rendering and
    # scene_render (actual media generation) restrict themselves to max_scenes.
    def _draft(fb=None, p=None):
        return draft_screenplay(
            source, structure, characters, scenes,
            soundscape=soundscape, visuals=visuals, cinematography=cinematography,
            casting=casting, profile=p or profile_override, feedback=fb,
        )
    g = run_group("8/10 screenplay (all scenes)", "draft", [
        _spec("screenplay", lambda: _draft(), _summarize_screenplay, _draft),
    ])
    draft = g["screenplay"]
    fountain = to_fountain(source, structure, draft)
    (out / "screenplay.fountain").write_text(fountain, encoding="utf-8")

    # ── 9/10  storyboard (fuses casting + art + camera + score per moment) ────
    def _board(fb=None, p=None):
        return plan_storyboard(
            structure, scenes, casting, soundscape, visuals, cinematography,
            characters=characters, draft=draft, genre=genre_spec,
            moodboard=moodboard,
            source=source,                      # full source dict with chunks
            source_text=source.get("text", ""), # backward-compat fallback
            profile=p or profile_override, feedback=fb, out=out,
        )
    g = run_group("9/10", "storyboard", [
        _spec("storyboard", lambda: _board(), _summarize_storyboard, _board),
    ])
    storyboard = g["storyboard"]

    # ── 10/11  video render — explicit stage: clips → per-scene video → movie ───
    # Depends on: storyboard (frames + prompts), casting (character images for seeding).
    # Produces: output/video/scene_NN/frame_MM.mp4  (frame clips)
    #           output/video/scene_NN.mp4            (per-scene stitch)
    #           output/video/movie.mp4               (final assembly)
    # Best-effort: skipped gracefully when no video backend is available.
    scene_render = {}
    if not i2v.enabled():
        _log(f"10/11 video render — skipped ({i2v.unavailable_hint()})")
    else:
        backend_label = i2v.backend()
        total_scenes = min(max_scenes, len(storyboard.get("storyboard", []))) if max_scenes else len(storyboard.get("storyboard", []))
        _log(f"10/11 video render  [{backend_label}]  "
             f"({total_scenes} scene(s), all shots, per-scene stitch + final assembly) …")
        scene_render = _render_scene_frames(storyboard, casting, out, max_scenes=max_scenes,
                                            characters=characters)
        if scene_render:
            ok = scene_render.get("clips", 0)
            failed = scene_render.get("failed", 0)
            total_clips = ok + failed
            scenes_stitched = sum(1 for s in scene_render.get("scenes", []) if s.get("scene_video"))
            total_scenes_rendered = len(scene_render.get("scenes", []))
            movie = scene_render.get("movie")
            if failed and ok:
                _log(f"      ⚠ {ok}/{total_clips} clips, {failed} failed — "
                     f"{scenes_stitched}/{total_scenes_rendered} scene video(s)"
                     + (f" → {movie}" if movie else ""))
            elif failed and not ok:
                _log(f"      ⚠ 0/{total_clips} clips — all frames failed; check logs above")
            else:
                _log(f"      {ok} clips → {scenes_stitched}/{total_scenes_rendered} scene video(s)"
                     + (f" → {movie}" if movie else " (stitch pending)"))

    # Aggregate the per-stage fidelity checks into one pipeline story-fidelity score.
    fidelity_summary = {}
    if fid_reports:
        overall = fidelity.score_pipeline(fid_reports)
        fidelity_summary = {"overall": overall, "per_stage": fid_reports}
        save("fidelity", fidelity_summary)
        _log(f"      story-fidelity: {overall.get('verdict')} "
             f"{overall.get('overall_score')}/100"
             + (f"; drift in {overall['drifting_stages']}" if overall.get("drifting_stages") else ""))

    # Aggregate the per-stage genre checks into one pipeline genre-alignment score.
    genre_summary = {}
    if genre_spec:
        overall_g = genre_agent.score_pipeline(gen_reports) if gen_reports else {}
        genre_summary = {"spec": genre_spec, "overall": overall_g, "per_stage": gen_reports}
        save("genre_alignment", genre_summary)
        if overall_g:
            _log(f"      genre [{genre_spec.get('genre', '?')}]: {overall_g.get('verdict')} "
                 f"{overall_g.get('overall_score')}/100"
                 + (f"; off-genre in {overall_g['off_genre_stages']}" if overall_g.get("off_genre_stages") else ""))
    llm.set_direction(None)   # clear steering once the creative stages are done

    # ── 11/11  assemble artifacts ─────────────────────────────────────────────
    # Per-stage JSON + screenplay.fountain are already written above (incremental,
    # crash-safe). Here we add the per-character files and the combined manifest.
    _log("11/11 assemble artifacts …")

    chars_dir = out / "characters"
    chars_dir.mkdir(exist_ok=True)
    for char in characters.get("characters", []):
        _write_json(chars_dir / f"{_slug(char.get('name', 'unknown'))}.json", char)

    project = {
        "title": source["title"],
        "session_id": session_id,
        "source": source["source_path"],
        "word_count": source["word_count"],
        "structure": structure,
        "moodboard": moodboard,
        "characters": characters,
        "casting": casting,
        "scenes": scenes,
        "soundscape": soundscape,
        "visuals": visuals,
        "cinematography": cinematography,
        "storyboard": storyboard,
        "screenplay_draft": draft,
        "scene_render": scene_render,
        "fidelity": fidelity_summary,
        "genre": genre_summary or genre_spec,
        "models": {"fast": fast_model, "quality": quality_model},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    save("project", project)
    session.finish(out, "complete")

    _log(f"done in {project['elapsed_seconds']}s → {out}/")
    return project


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
