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
                   panel (a fallback format only — see reel.veo_prompt below)

After storyboard + screenplay, an optional scene-render phase
(`_render_scene_frames` → `output/video/`) renders each storyboard panel into a
video clip via Gemini Veo (`reel.i2v`), seeded by the character's casting image
for identity continuity. Best-effort: no API key → skips with a hint. The
prompt actually sent to Veo is NOT the storyboard's free-text image_prompt —
it's reconstructed from structured data (casting.json, the scene's
visual_overview, the panel's own camera fields) into a fixed five-part
formula, always in this order: [Cinematography] + [Subject] + [Action] +
[Context] + [Style & Ambiance] (Google's Veo 3.1 prompting guide) by
`reel.veo_prompt` — shared by this module AND `reel.fountain`'s standalone
render path, so both build the same prompt shape from equivalent data (see
`veo_prompt`'s module docstring).

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
from .agents import structure as structure_agent
from .agents.characters import extract_characters
from .agents import characters as characters_agent
from .agents.scenes import segment_scenes
from .agents import scenes as scenes_agent
from .agents.casting import cast_characters
from .agents import casting as casting_agent
from .agents.soundscape import design_soundscape
from .agents import soundscape as soundscape_agent
from .agents.visuals import design_visuals
from .agents import visuals as visuals_agent
from .agents.cinematography import plan_cinematography
from .agents import cinematography as cinematography_agent
from .agents.storyboard import plan_storyboard
from .agents import storyboard as storyboard_agent
from .agents.screenplay import draft_screenplay, to_fountain
from .agents import screenplay as screenplay_agent
from .agents import fidelity
from .agents import genre as genre_agent
from .agents import critique as critique_agent
from .agents.moodboard import design_moodboard, guidance as moodboard_guidance
from .agents import moodboard as moodboard_agent
from .gate import Gate
from . import llm
from . import imagegen
from . import i2v
from . import veo_guide
from . import veo_prompt
from . import gemini
from . import session
from . import duration_budget
from .revision_merge import merge_by_key


def _log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


def _scenes_label(max_scenes: int | None) -> str:
    """Display form of a max_scenes cap — 'all' when unbounded (None), matching
    the `--max-scenes all` CLI value, instead of printing the literal 'None'."""
    return "all" if max_scenes is None else str(max_scenes)


# Which stages get story-fidelity / genre-alignment scoring at their gate —
# module-level (not local to `run()`) so `cli.py`'s `revise` flow can reuse
# the exact same stage sets for its own per-stage gates, rather than
# maintaining a second, potentially-drifting copy.
FIDELITY_GATED_STAGES = {"structure", "characters", "scenes", "casting", "soundscape",
                         "visuals", "cinematography", "screenplay", "storyboard"}
GENRE_GATED_STAGES = FIDELITY_GATED_STAGES | {"moodboard"}


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


def _summarize_storyboard(r: dict, target_seconds: int | None = None) -> str:
    rows = [f"Board style: {r.get('storyboard_style', '?')}"]
    if target_seconds:
        est = duration_budget.estimated_total_seconds(r)
        delta = est - target_seconds
        flag = "" if abs(delta) <= max(5, round(target_seconds * 0.15)) else "  ⚠ off target"
        rows.append(f"Estimated total runtime: {duration_budget.format_seconds(est)} "
                    f"(target: ~{target_seconds}s){flag}")
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
    dropped = r.get("dropped_scenes") or []
    if dropped:
        # A scene from scenes.json that the per-scene storyboard call returned
        # nothing usable for — surfaced here instead of silently vanishing
        # from the board (mirrors _summarize_scenes' own dropped_scenes).
        rows.append(f"  ⚠ {len(dropped)} scene(s) missing from the board — no usable "
                    "storyboard response for that scene:")
        for s in dropped:
            rows.append(f"      scene {s.get('scene_number', '?')}: {s.get('reason', '')}")
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
                           active_names: set[str] | None = None,
                           dry_run: bool = False) -> int:
    """Render ONE image per casting entry — character OR location representation —
    via the image backend. Kind-agnostic: a `kind: location` entry has no `actor`
    block, so the prompt lookup falls through to `character.visual_prompt`
    directly, same as any other entry. Capped to `active_names` when provided
    (character/location names that appear in the scenes being rendered).
    Idempotent by prompt hash: skips an entry whose rendered image still matches
    its current `visual_prompt`, but re-renders when feedback has revised that
    prompt since (--resume or a standalone `stage casting --feedback` rerun).

    `dry_run=True` (`--no-render`) still walks every entry and does the exact
    same staleness check — an entry whose image is already current is left
    alone either way — but for an entry that WOULD need a fresh render,
    `imagegen.generate_image(..., dry_run=True)` logs the real request
    params to `gemini_api.log` instead of actually calling the API. Since a
    dry-run call always returns False, the success bookkeeping below
    (image_path/hash/`n`) is naturally skipped too — an entry rendered this
    way stays exactly as "not yet rendered" as it was before this call, so a
    later real render still triggers for it.
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
        _log(f"      {'recording intended API call for' if dry_run else 'rendering'} "
             f"{name} [{kind}] …"
             + (" (prompt revised)" if dry_run and img.exists() else
                " (prompt revised — re-rendering)" if img.exists() else ""))
        if imagegen.generate_image(prompt, img, dry_run=dry_run):
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


# Veo prompt construction (the five-part formula, subject-anchoring,
# relevance filtering, audio-cue assembly) lives in reel/veo_prompt.py — a
# pure, no-render-loop-state module shared with reel/fountain.py's
# standalone render path, so both entry points build the exact same prompt
# from equivalent panel/scene data instead of maintaining two
# implementations that can silently drift apart. See veo_prompt.py's module
# docstring for the extraction rationale.


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


def _frame_char_anchor(frame: dict, cast_index: dict, out: Path) -> Path | None:
    """The casting image of the first RELEVANT in-frame character (see
    `veo_prompt.panel_relevant_characters`) — Veo identity seed."""
    for name in veo_prompt.panel_relevant_characters(frame):
        rel = cast_index.get(name)
        if rel and (out / rel).exists():
            return out / rel
    return None


def _char_set_changed(fr: dict, prev_char_key) -> bool:
    """True if THIS panel is a character "shot boundary" relative to
    `prev_char_key` — the immediately preceding panel's in-frame character
    SET (a `frozenset`, or `None` for a scene's first panel, meaning
    "nothing to compare against yet", which itself counts as a boundary).
    A panel with no listed characters at all is never a boundary (nothing
    to compare) — it's treated as a continuation of whatever came before.

    Shared by `_resolve_panel_references` (decides whether to use
    multi-character `reference_images` instead of a single seed) AND by
    every caller that decides whether Veo's `continuity_mode: extend` is
    even eligible for this panel (`_render_scene_frames`/`rerender_panels`
    null out `prev_clip_path` when this is True) — extending the PREVIOUS
    clip only makes sense when this panel is showing the same subject(s)
    that clip was; carrying it forward across a cast change would extend
    the wrong characters' continuity (and audio) into a shot that doesn't
    feature them. Scene-level PROPS (`visual_overview.key_props`) don't
    need an equivalent check here — they're attributed scene-wide, not
    per-panel, in this codebase's data model (no artifact currently says
    which panel a given prop appears in — see `veo_prompt.panel_context`),
    so they can only ever change at a SCENE boundary, which already resets
    `prev_clip_path`/`prev_tail` to `None` unconditionally between scenes."""
    names = fr.get("characters_in_frame") or []
    if not names:
        return False
    return prev_char_key is None or frozenset(names) != prev_char_key


def _resolve_panel_references(fr: dict, prev_char_key, cast_index: dict, out: Path):
    """Multi-character Veo `reference_images` for THIS panel, at a "shot
    boundary" (see `_char_set_changed`) — and only when 2+ of the in-frame
    characters have a resolvable casting portrait. A boundary panel with
    0-1 resolvable portraits returns no references at all: the caller keeps
    using its existing single-`seed` path (`_frame_char_anchor`/
    `prev_tail`), since Veo's `image=` conditioning is a stronger, more
    literal grounding for a lone subject than a loose identity reference,
    and there's nothing "multi" about a single character anyway. This is
    the fix for a real pipeline gap: `_frame_char_anchor` above only ever
    anchors the FIRST name in `characters_in_frame`, so a boundary panel
    with two or more characters previously gave every character AFTER the
    first one zero identity grounding.

    Returns `(reference_image_paths, this_panel's_char_key)` — the caller
    threads the second value back in as `prev_char_key` for its NEXT call
    (walking a scene panel-by-panel). A panel with no listed characters at
    all returns the caller's OWN `prev_char_key` unchanged (nothing here to
    update boundary tracking with, and nothing to reference either).

    The actual reference images are picked from
    `veo_prompt.panel_relevant_characters` (this panel's OWN action/
    dialogue-grounded subset), not the raw `characters_in_frame` list — so a
    wide/establishing panel whose `characters_in_frame` defaults to the
    entire scene cast doesn't burn the 3-reference budget on someone
    technically in the scene but absent from this specific panel. `char_key`
    (boundary/continuity tracking, handed back to the caller) is still
    computed from the FULL raw list — deliberately unaffected by this
    narrowing, since continuity/extend-mode eligibility is about whether the
    scene's cast context changed, not about which of them this one panel
    happens to foreground."""
    names = list(dict.fromkeys(fr.get("characters_in_frame") or []))
    char_key = frozenset(names) if names else prev_char_key
    if not _char_set_changed(fr, prev_char_key) or not names:
        return [], char_key
    refs = []
    for name in veo_prompt.panel_relevant_characters(fr)[:3]:  # Veo 3.1 accepts at most 3
        rel = cast_index.get(name)
        if rel and (out / rel).exists():
            refs.append(out / rel)
    return (refs if len(refs) >= 2 else []), char_key


def _render_one_panel(fr: dict, snum, sdir: Path, seed: Path | None,
                      prev_clip_path: Path | None, out: Path, *,
                      casting_lookup: dict, location_desc: str, visual_overview: dict,
                      voice_index: dict, audio_overview: dict,
                      no_bg_music: bool, room_tone: bool, no_subtitles: bool,
                      reference_images: list[Path] | None = None,
                      force: bool = False, dry_run: bool = False) -> dict:
    """Render (or skip, if unchanged) exactly ONE storyboard panel's video
    clip. Shared by `_render_scene_frames`'s per-scene loop and
    `rerender_panels`'s targeted re-render, so the hash/render/overlay/tail-
    extraction logic lives in exactly one place rather than drifting between
    two call sites.

    `force=True` bypasses the `_stale` staleness check entirely and always
    (re-)renders — used by `rerender_panels` for its explicitly targeted
    panel(s), since a caller re-rendering a specific panel on purpose wants
    that panel rebuilt even in the rare case its prompt+seed happen to hash
    identically to before.

    `dry_run=True` (`--no-render`) still computes the prompt and the exact
    same staleness check as a real call — a panel whose clip is already
    current is left alone either way — but for a panel that WOULD need a
    fresh render, `i2v.generate_clip(..., dry_run=True)` logs the real Veo
    request params without spending API quota. A dry-run call always
    returns False, so it's treated the same as `_stale`'s "not rendered"
    case in the caller's bookkeeping (no clip/hash/tail written, `rendered`
    stays False) — see the `elif dry_run:` branch below for why this is
    NOT logged as `failed` too (it's an intentional skip, not an error).

    `reference_images` — see `_resolve_panel_references` (both call sites
    compute it the same way, given to this function pre-computed so the
    boundary-detection logic lives in exactly one place too). Passed
    through to `i2v.generate_clip` ALONGSIDE `seed` (not instead of it) —
    `i2v._gen_gemini` is the one that decides which actually gets used,
    falling back to `seed` on its own if the reference-image call is
    disabled/unavailable/fails, so this function doesn't need to know
    which path was actually taken.

    Returns a dict: `frame_record` (the frames_out-shaped manifest entry —
    including the `start_frame`/`end_frame` fields), `prompt` (the exact Veo
    prompt used, for the per-scene prompt log), `tag`, `tail_path`/
    `clip_path` (Path objects, for the caller to chain continuity forward),
    `rendered` (True if a fresh clip was actually generated this call — as
    opposed to skipped because it was already current), `failed` (True if
    generation was attempted but did not produce a clip)."""
    fnum = fr.get("panel") or fr.get("frame", 0)
    prompt = veo_prompt.panel_video_prompt(fr, audio_overview,
                                 casting_lookup=casting_lookup,
                                 location_desc=location_desc,
                                 visual_overview=visual_overview,
                                 voice_index=voice_index,
                                 no_bg_music=no_bg_music, room_tone=room_tone,
                                 no_subtitles=no_subtitles,
                                 out=out, seed=seed, reference_images=reference_images)
    # This panel's own requested clip length, from the storyboard's already-
    # estimated `duration` field (see storyboard._estimate_duration) — each
    # video backend is its own adaptor for what to actually do with the
    # request (Veo rounds to its 3 valid values; others honor it as a frame
    # count) — see reel.i2v.generate_clip's docstring. None (unparseable/
    # absent) lets the backend fall back to its own default.
    requested_seconds = duration_budget.parse_duration_seconds(fr.get("duration")) or None
    tag = f"{int(fnum):02d}" if isinstance(fnum, int) else str(fnum)
    clip = sdir / f"frame_{tag}.mp4"
    tail_img = sdir / f"frame_{tag}_tail.png"
    hash_path = sdir / f"frame_{tag}.hash"
    # Hash covers the prompt, the seed image bytes, the requested duration,
    # AND every reference image's bytes (a casting portrait re-rendered via
    # a revision must invalidate a boundary panel that references it, same
    # as it already invalidates a panel that SEEDS from it): a feedback/
    # revision to an earlier frame changes its tail frame, which changes
    # this frame's seed, which — even with an unchanged prompt — must still
    # invalidate this clip so continuity re-chains correctly; a duration-
    # only change (e.g. after a revision shifts the panel's estimated
    # screen time) must invalidate it too, since neither the prompt text
    # nor the seed reflects that on their own.
    current_hash = _content_hash(prompt, seed, prev_clip_path, requested_seconds,
                                 *(reference_images or []))
    rendered = False
    failed = False
    if force or _stale(clip, hash_path, current_hash):
        if clip.exists():
            _log(f"      scene {snum} frame {tag} — prompt/seed revised"
                 + (", API params logged (--no-render) …" if dry_run else ", re-rendering …"))
        elif dry_run:
            _log(f"      scene {snum} frame {tag} — recording intended API call (--no-render) …")
        if i2v.generate_clip([seed] if seed else [], prompt, clip, prev_clip=prev_clip_path,
                             duration_seconds=requested_seconds,
                             reference_images=reference_images, dry_run=dry_run):
            rendered = True
            # Burn subtitle + shot-label overlays onto the clip (in-place)
            # when the operator enables them in config video.overlays.
            if i2v.overlays_enabled():
                dlg = veo_prompt.panel_dialogue_lines(fr)
                stype = fr.get("shot_type") or ""
                shot_lbl = f"S{snum}·F{tag}" + (f"·{stype}" if stype else "")
                i2v.add_overlays(clip, clip, dialogue_lines=dlg or None, shot_label=shot_lbl)
            # Always extract the tail frame so every clip has one on disk.
            # Continuity chains it forward as the next-clip seed; ffmpeg
            # stitching benefits from having clean cut-points regardless.
            i2v.last_frame(clip, tail_img)
            hash_path.write_text(current_hash)
        elif dry_run:
            pass  # intentional skip, not a failure — already logged above
        else:
            failed = True
            _log(f"      ⚠ scene {snum} frame {tag} — clip not produced")

    frame_record = {
        "panel": fnum,
        "shot_type": fr.get("shot_type", ""),
        "action": fr.get("action") or fr.get("moment", ""),
        "seed": str(Path(seed).relative_to(out)) if seed and Path(seed).exists() else None,
        # start_frame/end_frame: explicit record of this clip's first and last
        # frame (start_frame duplicates "seed" — kept for back-compat with any
        # existing reader of that field), so a specific panel can be
        # re-rendered alone later and correctly re-seeded/re-stitched against
        # its neighbors (see rerender_panels) without re-deriving these paths.
        "start_frame": str(Path(seed).relative_to(out)) if seed and Path(seed).exists() else None,
        "end_frame": str(tail_img.relative_to(out)) if tail_img.exists() else None,
        "clip": str(clip.relative_to(out)) if clip.exists() else None,
        # Recorded for traceability even though _gen_gemini decides at
        # render time whether the reference-image call was actually used
        # (vs. falling back to `seed`) — lets an operator see WHICH
        # characters this "shot boundary" panel was asked to identity-lock.
        "reference_images": [str(p.relative_to(out)) for p in reference_images] if reference_images else None,
    }
    return {
        "frame_record": frame_record,
        "prompt": prompt,
        "tag": tag,
        "tail_path": tail_img,
        "clip_path": clip,
        "rendered": rendered,
        "failed": failed,
    }


def _stitch_scene(scene_vid: Path, frames_out: list, out: Path, snum) -> str | None:
    """Unconditionally (re)build `scene_vid` from `frames_out`'s clips, in
    order (ffmpeg's `-y` overwrites any existing file). Returns the relative
    path string on success, None if there were no clips to stitch or the
    stitch failed. Callers decide whether a re-stitch is actually needed —
    `_render_scene_frames` skips calling this when `scene_vid` already exists
    (the normal resume case, nothing changed); `rerender_panels` always calls
    it, since it just changed at least one clip in this scene."""
    scene_clips = [out / fr["clip"] for fr in frames_out
                   if fr.get("clip") and (out / fr["clip"]).exists()]
    if not scene_clips:
        return None
    if i2v.stitch(scene_clips, scene_vid):
        _log(f"      scene {snum}: stitched {len(scene_clips)} clip(s) → {scene_vid.name}")
        return str(scene_vid.relative_to(out))
    _log(f"      ⚠ scene {snum}: per-scene stitch failed")
    return None


def _render_scene_frames(storyboard: dict, casting: dict, out: Path,
                         max_scenes: int | None = None,
                         only_scenes: set | None = None,
                         existing_manifest: dict | None = None,
                         characters: dict | None = None,
                         dry_run: bool = False) -> dict:
    """Render each storyboard frame as a video clip (Veo image-to-video), then
    stitch each scene's clips into a per-scene video (output/video/scene_NN.mp4)
    and assemble all scene videos into the final movie (output/video/movie.mp4).

    `dry_run=True` (`--no-render`) still walks every scene/panel and computes
    every prompt/seed/reference exactly as a real render would — see
    `_render_one_panel`'s dry_run docstring for what that logs. Since a
    dry-run panel never produces a clip, `_stitch_scene`/`_assemble_movie`
    below naturally find nothing new to stitch for scenes that don't
    already have a real clip on disk from an earlier render (and correctly
    leave an already-rendered scene's existing video alone) — no separate
    dry-run branch needed at the stitch/assembly level.

    Identity seeding: the first frame of each scene seeds from the in-frame
    character's representation image; subsequent frames chain from the previous
    clip's last frame for continuity within the scene. Scene boundary = hard cut.

    `characters` (optional, from characters.json) supplies each speaking
    character's vocal-quality description so dialogue cues stay consistent
    across separately-generated clips (see `_panel_video_prompt`).

    max_scenes caps how many scenes are rendered (a LEADING SLICE — the first
    N); every shot within each rendered scene is always included. `only_scenes`
    (a set of `scene_number`s) instead targets a SPECIFIC subset — the two are
    mutually exclusive; `only_scenes` wins if both are given. Used by the
    `revise` CLI flow to re-render exactly the scene(s) a revision touched
    without re-processing the whole story. Best-effort + idempotent (skips
    existing files; re-renders when a feedback revision changed the prompt —
    see `_stale`).

    `existing_manifest`: when scoping to `only_scenes`, this function would
    otherwise overwrite output/video/manifest.json with a manifest containing
    ONLY the scoped scene(s) — silently discarding every other scene's
    record. Pass the current manifest (e.g. loaded from
    output/video/manifest.json) so untouched scenes' entries and running
    clip/failed counts are preserved; only the processed scene(s)' entries are
    replaced/appended.
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
    if only_scenes is not None:
        board = [s for s in board if s.get("scene_number") in only_scenes]
    elif max_scenes:
        board = board[:max_scenes]              # limit scenes, never the shots within
    for scene in board:
        snum = scene.get("scene_number", "x")
        sdir = vdir / (f"scene_{snum:02d}" if isinstance(snum, int) else f"scene_{snum}")
        sdir.mkdir(exist_ok=True)
        prev_tail = None                        # reset each scene → hard cut between scenes
        prev_clip_path = None                   # previous clip mp4 — for continuity_mode: extend
        prev_char_key = None                    # reset each scene — see _resolve_panel_references
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
            # Seed: continue from the previous frame's tail (carries the look
            # forward); the first frame of a scene seeds from the in-frame
            # character's representation image (identity reference).
            seed = prev_tail if (prev_tail and continuity) else _frame_char_anchor(fr, cast_index, out)
            # Extend-mode continuity (config `continuity_mode: extend`) may
            # only continue from the previous CLIP when this panel's
            # in-frame characters are the SAME as that clip's — extending a
            # clip across a cast change would carry the wrong subject's
            # continuity/audio into a shot that doesn't feature them. Props
            # (visual_overview.key_props) need no separate check: they're
            # scene-wide, not per-panel, in this codebase's data model, so
            # they can only change at a SCENE boundary — already covered by
            # `prev_clip_path`'s per-scene reset above. Nulled here (per
            # call, not the loop variable itself) rather than in i2v, so
            # `_gen_gemini`'s extend branch is never even attempted for a
            # boundary panel.
            effective_prev_clip = None if _char_set_changed(fr, prev_char_key) else prev_clip_path
            # Multi-character identity lock at the SAME "shot boundary" —
            # see _resolve_panel_references. Computed alongside `seed`, not
            # instead of it: `i2v._gen_gemini` decides which actually gets
            # used, falling back to `seed` on its own if this is
            # disabled/unavailable/fails.
            reference_images, prev_char_key = _resolve_panel_references(
                fr, prev_char_key, cast_index, out)
            res = _render_one_panel(fr, snum, sdir, seed, effective_prev_clip, out,
                                    casting_lookup=casting_lookup, location_desc=location_desc,
                                    visual_overview=visual_overview, voice_index=voice_index,
                                    audio_overview=audio_overview,
                                    no_bg_music=no_bg_music, room_tone=room_tone,
                                    no_subtitles=no_subtitles,
                                    reference_images=reference_images or None,
                                    dry_run=dry_run)
            if res["rendered"]:
                manifest["clips"] += 1
                if continuity:
                    prev_tail = res["tail_path"] if res["tail_path"].exists() else seed
                    prev_clip_path = res["clip_path"]
            elif res["failed"]:
                manifest["failed"] += 1
            elif continuity:
                # Already up to date — still chain forward from its tail frame
                # (previously this branch left prev_tail untouched, so a resumed
                # run with some frames already rendered would reset newer frames
                # to the character anchor instead of continuing the scene).
                prev_tail = res["tail_path"] if res["tail_path"].exists() else seed
                prev_clip_path = res["clip_path"]

            frames_out.append(res["frame_record"])
            prompt_log.append({
                "tag": res["tag"],
                "prompt": res["prompt"],
                "clip": res["frame_record"]["clip"],
            })

        _write_scene_prompt_log(
            out, snum, vcfg.get("model", "unknown"),
            "tail-frame continuity (previous clip's last frame)" if continuity
            else "character/location identity anchor per frame (no continuity)",
            prompt_log,
        )

        # Stitch this scene's frame clips into a scene-level video.
        scene_vid = vdir / (f"scene_{snum:02d}.mp4" if isinstance(snum, int) else f"scene_{snum}.mp4")
        if scene_vid.exists():
            scene_vid_rel = str(scene_vid.relative_to(out))   # already done (resume)
        else:
            scene_vid_rel = _stitch_scene(scene_vid, frames_out, out, snum)

        manifest["scenes"].append({
            "scene_number": snum,
            "frames": frames_out,
            "scene_video": scene_vid_rel,
        })

    if existing_manifest is not None:
        # Splice this call's processed scene(s) into the full prior manifest —
        # otherwise a scoped (only_scenes) call would overwrite manifest.json
        # with just the scene(s) it touched, discarding every other scene's
        # record. Running counts add on top of the prior totals.
        processed_numbers = {s.get("scene_number") for s in manifest["scenes"]}
        manifest["scenes"] = merge_by_key(
            existing_manifest.get("scenes", []), manifest["scenes"],
            lambda s: s.get("scene_number"), processed_numbers,
        )
        manifest["clips"] = existing_manifest.get("clips", 0) + manifest["clips"]
        manifest["failed"] = existing_manifest.get("failed", 0) + manifest["failed"]

    # Final assembly: stitch scene videos (preferred) or raw clips into movie.mp4.
    movie = _assemble_movie(manifest, out)
    if movie:
        manifest["movie"] = str(movie.relative_to(out))

    (vdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def rerender_panels(storyboard: dict, casting: dict, out: Path, *,
                    scene_number, panel_numbers: list,
                    characters: dict | None = None) -> dict:
    """Re-render specific panel(s) of ONE scene, cascade exactly ONE panel
    forward (the panel immediately after the last targeted one, reseeded from
    its new end_frame, to keep that one seam visually smooth), then STOP —
    deliberately bounded to avoid re-rendering (and re-paying Veo API cost
    for) the rest of the scene. Re-stitches the scene and reassembles
    movie.mp4 afterward.

    Requires `output/video/manifest.json` to already have an entry for this
    scene (i.e. it was rendered at least once by `_render_scene_frames`) — a
    never-rendered scene has no prior end_frame to chain the first targeted
    panel's start from; render it via
    `_render_scene_frames(..., only_scenes={scene_number})` instead.

    Returns the updated manifest (also written to output/video/manifest.json).
    """
    vdir = out / "video"
    manifest_path = vdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no {manifest_path} — this scene has never been rendered; use "
            "_render_scene_frames(..., only_scenes={scene_number}) first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    scene_entry = next((s for s in manifest.get("scenes", [])
                        if s.get("scene_number") == scene_number), None)
    if scene_entry is None:
        raise FileNotFoundError(
            f"no manifest entry for scene {scene_number} — it has never been "
            "rendered; use _render_scene_frames(..., only_scenes={scene_number}) first")

    board_scene = next((s for s in storyboard.get("storyboard", [])
                        if s.get("scene_number") == scene_number), None)
    if board_scene is None:
        raise KeyError(f"scene {scene_number} not found in storyboard")

    vcfg = i2v._cfg()
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
    audio_overview = board_scene.get("audio_overview") or {}
    loc_name = (board_scene.get("header", {}).get("location") or "").strip()
    loc_entry = casting_lookup.get(loc_name, {})
    loc_ch = loc_entry.get("character", loc_entry)
    location_desc = loc_ch.get("visual_prompt") or loc_name
    visual_overview = board_scene.get("visual_overview") or {}
    panels = board_scene.get("panels") or board_scene.get("frames", [])
    panels_by_num = {(p.get("panel") or p.get("frame")): p for p in panels}

    snum = scene_number
    sdir = vdir / (f"scene_{snum:02d}" if isinstance(snum, int) else f"scene_{snum}")
    sdir.mkdir(exist_ok=True)

    frames_by_num = {f.get("panel"): f for f in scene_entry.get("frames", [])}
    first_panel_num = min(panels_by_num) if panels_by_num else None

    def _resolve_start_frame(pnum):
        """Start frame for panel `pnum`: the casting reference image if it's
        the scene's first panel, else the immediately-preceding panel's
        recorded end_frame — falling back to re-deriving the tail PNG from
        disk (it's always written on a fresh render regardless of whether an
        older manifest predates the end_frame field) if that record is
        missing."""
        if pnum == first_panel_num:
            return _frame_char_anchor(panels_by_num[pnum], cast_index, out)
        prev_num = pnum - 1 if isinstance(pnum, int) else None
        prev_record = frames_by_num.get(prev_num) if prev_num is not None else None
        if prev_record and prev_record.get("end_frame"):
            p = out / prev_record["end_frame"]
            if p.exists():
                return p
        if prev_num is not None:
            prev_tag = f"{int(prev_num):02d}" if isinstance(prev_num, int) else str(prev_num)
            p = sdir / f"frame_{prev_tag}_tail.png"
            if p.exists():
                return p
        return _frame_char_anchor(panels_by_num[pnum], cast_index, out)

    def _resolve_prev_clip_path(pnum):
        """The immediately-preceding panel's CURRENT clip path — mirrors what
        `_render_scene_frames`'s normal top-to-bottom walk always threads as
        `prev_clip_path`, regardless of whether that preceding panel was
        touched this call or is untouched from before. Getting this wrong
        would make this panel's `_content_hash` differ from what a later full
        `_render_scene_frames` pass computes for the same panel, spuriously
        marking it stale again (the exact bug this mirroring avoids)."""
        if pnum == first_panel_num:
            return None
        prev_num = pnum - 1 if isinstance(pnum, int) else None
        prev_record = frames_by_num.get(prev_num) if prev_num is not None else None
        if prev_record and prev_record.get("clip"):
            p = out / prev_record["clip"]
            if p.exists():
                return p
        return None

    def _prev_char_key(pnum):
        """The immediately-preceding panel's in-frame character set, read
        from the STORYBOARD (not the manifest — a targeted re-render may
        not be walking sequentially) — mirrors what `_render_scene_frames`'s
        own top-to-bottom walk would have as `prev_char_key` for this same
        panel, so `_resolve_panel_references`'s boundary detection (and
        thus this panel's `_content_hash`) stays consistent between the two
        entry points, same reasoning as `_resolve_prev_clip_path` above."""
        if pnum == first_panel_num:
            return None
        prev_num = pnum - 1 if isinstance(pnum, int) else None
        prev_fr = panels_by_num.get(prev_num) if prev_num is not None else None
        names = list(dict.fromkeys((prev_fr or {}).get("characters_in_frame") or []))
        return frozenset(names) if names else None

    def _render(pnum):
        fr = panels_by_num[pnum]
        seed = _resolve_start_frame(pnum)
        prev_key = _prev_char_key(pnum)
        # Same extend-eligibility rule as _render_scene_frames' own walk —
        # see _char_set_changed's docstring — kept consistent between the
        # two entry points for the same hash-mismatch reason
        # _resolve_prev_clip_path itself already documents.
        prev_clip_path = None if _char_set_changed(fr, prev_key) else _resolve_prev_clip_path(pnum)
        reference_images, _ = _resolve_panel_references(fr, prev_key, cast_index, out)
        res = _render_one_panel(fr, snum, sdir, seed, prev_clip_path, out,
                                casting_lookup=casting_lookup, location_desc=location_desc,
                                visual_overview=visual_overview, voice_index=voice_index,
                                audio_overview=audio_overview,
                                reference_images=reference_images or None,
                                no_bg_music=no_bg_music, room_tone=room_tone,
                                no_subtitles=no_subtitles, force=True)
        frames_by_num[pnum] = res["frame_record"]
        if res["rendered"]:
            manifest["clips"] = manifest.get("clips", 0) + 1
        elif res["failed"]:
            manifest["failed"] = manifest.get("failed", 0) + 1
        return res

    target_sorted = sorted(p for p in panel_numbers if p in panels_by_num)
    missing = [p for p in panel_numbers if p not in panels_by_num]
    for p in missing:
        _log(f"      ⚠ rerender_panels: panel {p} not found in scene {snum} storyboard — skipped")

    for pnum in target_sorted:
        _render(pnum)

    # One-hop cascade: the panel immediately after the LAST targeted panel.
    next_panel_num = None
    if target_sorted:
        after = sorted(n for n in panels_by_num if isinstance(n, int) and n > target_sorted[-1])
        next_panel_num = after[0] if after else None

    if next_panel_num is not None:
        next_result = _render(next_panel_num)

        # Cascade-stop bookkeeping: the panel AFTER next_panel_num must not be
        # spuriously judged stale on some future run just because
        # next_panel_num's tail bytes genuinely changed underneath it. We do
        # NOT touch that later panel's actual clip/end_frame — the accepted
        # one-hop-only visual discontinuity beyond this point is real and
        # should stay real, not be hidden. Instead we recompute and rewrite
        # ONLY its .hash sidecar to the value it WOULD have if fully
        # re-chained, so a future _render_scene_frames/rerender_panels call
        # doesn't redundantly re-trigger a render for it.
        after2 = sorted(n for n in panels_by_num if isinstance(n, int) and n > next_panel_num)
        after_num = after2[0] if after2 else None
        if after_num is not None:
            after_fr = panels_by_num[after_num]
            after_tag = f"{int(after_num):02d}" if isinstance(after_num, int) else str(after_num)
            after_clip = sdir / f"frame_{after_tag}.mp4"
            after_hash_path = sdir / f"frame_{after_tag}.hash"
            if after_clip.exists():
                new_end = next_result["frame_record"].get("end_frame")
                new_end_path = (out / new_end) if new_end else None
                new_clip = next_result["frame_record"].get("clip")
                new_clip_path = (out / new_clip) if new_clip else None
                # after_num's reference_images (if it's itself a boundary
                # panel) depend on next_panel_num's NEW character set as
                # prev_char_key — must be included here too, same reasoning
                # as new_end_path/new_clip_path above: this bookkeeping has
                # to match what _render_one_panel's own hash formula would
                # produce for after_num, or a boundary after_num would be
                # spuriously flagged stale (or not) on a later pass. Computed
                # BEFORE the prompt below, since the prompt's own Subject
                # text depends on which images (new_end_path/after_refs) are
                # actually attached — see veo_prompt.anchored_character_names.
                next_fr = panels_by_num[next_panel_num]
                next_names = list(dict.fromkeys(next_fr.get("characters_in_frame") or []))
                next_char_key = frozenset(next_names) if next_names else _prev_char_key(next_panel_num)
                after_refs, _ = _resolve_panel_references(after_fr, next_char_key, cast_index, out)
                after_prompt = veo_prompt.panel_video_prompt(after_fr, audio_overview,
                                                   casting_lookup=casting_lookup,
                                                   location_desc=location_desc,
                                                   visual_overview=visual_overview,
                                                   voice_index=voice_index,
                                                   no_bg_music=no_bg_music, room_tone=room_tone,
                                                   no_subtitles=no_subtitles,
                                                   out=out, seed=new_end_path,
                                                   reference_images=after_refs or None)
                # Same extend-eligibility rule as elsewhere: if after_num's
                # own cast differs from next_panel_num's, a full re-chain
                # would NOT have extended from new_clip_path either — must
                # match here too, or this bookkeeping hash would disagree
                # with _render_one_panel's real formula for after_num.
                accepted_prev_clip = None if _char_set_changed(after_fr, next_char_key) else new_clip_path
                accepted_hash = _content_hash(after_prompt, new_end_path, accepted_prev_clip,
                                              *(after_refs or []))
                after_hash_path.write_text(accepted_hash)

    # Re-stitch this scene's clips (some changed) and reassemble the movie.
    updated_frames_out = [frames_by_num[n] for n in sorted(
        frames_by_num, key=lambda x: x if isinstance(x, int) else 1e9)]
    scene_vid = vdir / (f"scene_{snum:02d}.mp4" if isinstance(snum, int) else f"scene_{snum}.mp4")
    scene_vid_rel = _stitch_scene(scene_vid, updated_frames_out, out, snum)

    scene_entry["frames"] = updated_frames_out
    scene_entry["scene_video"] = scene_vid_rel

    movie = _assemble_movie(manifest, out)
    if movie:
        manifest["movie"] = str(movie.relative_to(out))

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _spec(name: str, compute: Callable, summarize: Callable, rerun: Callable,
          realign: Callable | None = None, agent_module=None) -> dict:
    """Describe one stage: how to compute it, summarize it, re-run it on
    feedback, and (for scene-keyed stages) reiterate it to fix scene-structure
    alignment. `realign(result, revise_keys) -> dict` calls the SAME agent
    scoped to just the given scene numbers (existing=result, revise_keys=...)
    — distinct from `rerun`, which always regenerates from scratch on operator
    feedback. `None` for stages with no scene-keyed structure to align
    (structure, characters, casting, moodboard) — see run_group's use of it,
    and reel.agents.fidelity.check_scene_alignment for what "aligned" means.

    `agent_module` (optional) — the agent module whose `SYSTEM`/`PROMPT`
    constants describe what this stage was actually asked to do. When given,
    `_gated` runs one automatic self-critique-and-refine pass
    (`reel.agents.critique`, using `rerun` as the refine mechanism) BEFORE
    the operator ever sees the review gate — see `_gated`'s docstring.
    `None` for stages critique doesn't apply to (there are none among the
    creative stages currently, but a future non-text/render-only stage
    added to this registry should pass `None` here, same as it already
    would for `realign`)."""
    return {"name": name, "compute": compute, "summarize": summarize, "rerun": rerun,
            "realign": realign, "agent_module": agent_module}


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


def _format_alignment(rep: dict | None) -> str:
    """One-block scene-alignment readout for the review gate. Unlike fidelity/
    genre (a judgment call, advisory-only), a scene-alignment gap is a
    structural bug — run_group already tries to self-heal it (strip orphans,
    reiterate missing scenes) BEFORE the gate is shown, so this only ever
    prints when that reiteration still left a gap (orphans are always fully
    resolved by the deterministic strip; only `missing_scenes` can survive,
    e.g. no `realign` callable for this stage, or the LLM's scoped rerun
    still didn't cover it)."""
    if not rep or rep.get("aligned"):
        return ""
    missing = rep.get("missing_scenes") or []
    if not missing:
        return ""
    return (f"\n  ⚠ scene alignment: missing scene(s) {missing} vs scenes.json "
           "— reiteration did not fully resolve this; consider re-running with feedback")


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
    alignment_rep: dict | None = None,  # precomputed by run_group, AFTER its self-heal attempt
    profile: str | None = None,     # resolved profile name (display + escalation)
    escalate_after: int = 3,        # consecutive low-score reruns before gradual escalation
    escalate_score_gap: int = 20,   # escalate immediately when score is this far below threshold
    agent_module=None,              # module with SYSTEM/PROMPT — enables the self-critique pass
    critique_enabled: bool = True,  # config critique.enabled — no-ops the pass below when False
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

    `alignment_rep` (scene-structure alignment vs. scenes.json — see
    reel.agents.fidelity.check_scene_alignment) is a SNAPSHOT computed once by
    run_group, after it already tried to self-heal the stage (strip orphans,
    reiterate missing scenes) — unlike fidelity_fn/genre_fn it is not
    recomputed each loop iteration, since by the time this gate is showing,
    the automatic fix has already been attempted; it's surfaced here purely
    so the operator can see when that attempt still left a gap.

    Self-critique (`agent_module`, `critique_enabled`): ONE automatic pass,
    BEFORE the while loop below and thus before the operator ever sees the
    gate at all — `reel.agents.critique.critique_stage` reviews
    `initial_result` against `agent_module.SYSTEM`/`PROMPT` (what this stage
    was actually asked to do), and if it finds real issues, `rerun_fn` is
    called ONCE with the critique's `improvement_note` as feedback — the
    exact same mechanism a human's typed gate feedback already uses, just
    fired automatically first. Runs only for a genuinely fresh compute (the
    caller — run_group — never re-invokes this for a stage loaded from a
    `--resume` checkpoint), and only ONCE — a critique-driven refine is not
    itself re-critiqued, so this can't loop. Best-effort: any exception here
    is logged and treated as "solid" (no refine), never blocks the pipeline.
    The PRE-critique `initial_result` is preserved by the caller as
    `output/<name>.0.json` regardless of what happens here.

    Returns (approved_result, fidelity_report, genre_report). Raises PipelineStopped.
    """
    result = initial_result
    current_profile = profile
    low_score_run = 0
    iteration = 0

    if agent_module is not None and critique_enabled:
        try:
            crit = critique_agent.critique_stage(
                name, agent_module.SYSTEM, agent_module.PROMPT, result, profile=current_profile)
        except Exception as e:
            crit = None
            _log(f"      critique[{name}] skipped ({type(e).__name__})")
        if crit and crit.get("verdict") == "needs_improvement" and crit.get("improvement_note"):
            _log(f"      [{name}] self-critique found room to improve — refining once …")
            for issue in (crit.get("issues") or [])[:5]:
                _log(f"        - {issue}")
            try:
                result = rerun_fn(crit["improvement_note"], current_profile)
            except Exception as e:
                _log(f"      [{name}] self-critique refine failed ({type(e).__name__}) — "
                     "keeping the pre-critique result")
                result = initial_result

    while True:
        report = fidelity_fn(result) if fidelity_fn else None
        grep = genre_fn(result) if genre_fn else None

        iter_label = "initial" if iteration == 0 else f"iteration {iteration + 1}"
        _log(f"      {name}  [{_model_label(current_profile)}]  ({iter_label})")

        def _summary(r, _rep=report, _g=grep):
            return (summarize_fn(r) + _format_fidelity(_rep, min_score)
                   + _format_genre(_g, genre_min) + _format_alignment(alignment_rep))

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


def compose_direction(genre_spec: dict | None, moodboard_spec: dict | None) -> str | None:
    """Compose the shared creative-direction string from genre + moodboard
    guidance, honoring each's own config `steer` flag — the same composition
    `run()`'s local `apply_direction()` uses at pipeline startup, extracted
    to module level so `cli.py`'s `revise` flow can restore the SAME
    steering a completed run used before regenerating any stage. `revise` is
    a separate process invocation from the original `pipeline.run()` call,
    so `llm.set_direction`'s process-wide directive starts unset there —
    without this, every stage regenerated via `revise` would silently lose
    genre/moodboard steering even though genre.json/moodboard.json are
    sitting right there on disk from the original run."""
    cfg = llm.config()
    gen_steer = bool(cfg.get("genre", {}).get("steer", True))
    mood_steer = bool(cfg.get("moodboard", {}).get("steer", True))
    parts = []
    if gen_steer and genre_spec:
        parts.append(genre_agent.guidance(genre_spec))
    if mood_steer and moodboard_spec:
        parts.append(moodboard_guidance(moodboard_spec))
    return "\n\n".join(p for p in parts if p) or None


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    input_path: str,
    out_dir: str = "output",
    max_scenes: int | None = 1,
    profile_override: str | None = None,
    resume: bool = False,
    genre: str | None = None,
    target_duration_seconds: int | None = None,
    render: bool = True,
) -> dict:
    """Run the full screenplay-material phase and write artifacts to `out_dir`.

    Pause anytime by typing 'stop' at a review gate (or Ctrl-C); every stage
    already approved stays on disk. Re-run with `resume=True` to load those
    checkpoints and continue from the first stage that hasn't been completed.

    `target_duration_seconds` (default: config `duration.target_seconds`,
    itself defaulting to `reel.duration_budget.DEFAULT_TARGET_SECONDS`, 45) —
    the target total runtime of the rendered movie. Engine-independent
    planning guidance for scenes/cinematography (a budget hint, not forced
    arithmetic — see `reel.duration_budget`) AND the basis for each rendered
    clip's requested duration, translated by whichever video backend is
    configured (see `reel.i2v.generate_clip`).

    `render` (default `True`) — when `False`, skips the actual Gemini/Veo API
    invocation for casting-image generation AND video rendering for this
    run — the two ONLY stages that spend real API quota (see `cli.py`'s
    `--no-render` flag, and `revise`'s own `render` flag/`RENDER_SKIP_STAGES`
    for the analogous toggle in the revision flow). Every prompt/seed/
    reference is still computed exactly as a real render would, and its
    exact request params are still logged to `output/logs/gemini_api.log`
    (`outcome=skipped(no-render)`) — only the network/SDK call itself is
    disabled (`_render_casting_images`/`_render_scene_frames`'s `dry_run`
    param, threaded down through `imagegen`/`i2v` to `gemini.py`'s actual
    API functions). Every other stage (structure, characters, scenes,
    soundscape, visuals, cinematography, screenplay, storyboard) runs
    normally either way — this is purely a media-generation cost switch,
    not a scoping mechanism (unlike `max_scenes`, which still applies to
    whichever of the two stages DOES run when `render=True`).
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

    target_seconds = target_duration_seconds or \
        llm.config().get("duration", {}).get("target_seconds", duration_budget.DEFAULT_TARGET_SECONDS)
    _log(f"      target runtime: ~{target_seconds}s")

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
    _FID_STAGES = FIDELITY_GATED_STAGES

    # Self-critique: one automatic critique-and-refine pass per freshly-computed
    # stage, before the operator ever sees the gate — see _gated's docstring.
    # Toggle via config `critique.enabled`.
    crit_on = bool(llm.config().get("critique", {}).get("enabled", True))

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
    moodboard: dict = {}
    _GENRE_STAGES = GENRE_GATED_STAGES

    def apply_direction() -> None:
        """Compose the shared creative direction from genre + moodboard and steer
        all subsequent creative generations with it (graders stay neutral).
        Delegates to the module-level `compose_direction` (see its docstring)
        so `cli.py`'s `revise` flow can reuse the exact same composition."""
        llm.set_direction(compose_direction(genre_spec, moodboard))

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

            _save_initial_response(out, nm, raws[nm],
                                   crit_on and s.get("agent_module") is not None)

            # Scene-structure alignment: self-heal BEFORE the gate is shown,
            # since a misaligned scene isn't a judgment call for the operator
            # to weigh in on — it's an objective structural bug (see
            # reel.agents.fidelity.check_scene_alignment's docstring). Orphan
            # scenes (stale data scenes.json no longer has) are stripped
            # unconditionally; missing scenes (scenes.json has them, this
            # stage doesn't) are reiterated once via the SAME agent scoped to
            # just those scene numbers — the same existing=/revise_keys=
            # mechanism the `revise` CLI command uses.
            align_rep = None
            realign_fn = s.get("realign")
            if realign_fn is not None:
                raws[nm] = fidelity.strip_orphan_scenes(nm, raws[nm], scenes)
                align_rep = fidelity.check_scene_alignment(nm, raws[nm], scenes)
                if not align_rep["aligned"]:
                    missing = {k for k in align_rep["missing_scenes"] if isinstance(k, int)}
                    if missing:
                        _log(f"      ⚠ [{nm}] scene alignment drift vs scenes.json "
                             f"(missing {sorted(missing)}) — reiterating …")
                        raws[nm] = realign_fn(raws[nm], missing)
                        align_rep = fidelity.check_scene_alignment(nm, raws[nm], scenes)
                        _log(f"      {'✓' if align_rep['aligned'] else '⚠'} [{nm}] scene alignment "
                             + ("restored" if align_rep["aligned"]
                                else f"still missing {align_rep['missing_scenes']} after reiteration"))

            r, rep, grep = _gated(gate, nm, raws[nm], s["summarize"], s["rerun"],
                                  fidelity_fn=fid_fn, min_score=fid_min,
                                  genre_fn=gen_fn, genre_min=gen_min,
                                  alignment_rep=align_rep,
                                  profile=stage_profile, escalate_after=escalate_after,
                                  escalate_score_gap=escalate_score_gap,
                                  agent_module=s.get("agent_module"),
                                  critique_enabled=crit_on)
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
              lambda fb, p=None: analyze_structure(source, p or profile_override, feedback=fb),
              agent_module=structure_agent),
        _spec("characters",
              lambda: extract_characters(source, profile_override),
              _summarize_characters,
              lambda fb, p=None: extract_characters(source, p or profile_override, feedback=fb),
              agent_module=characters_agent),
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
                                                       feedback=fb),
                  agent_module=moodboard_agent),
        ])
        moodboard = g["moodboard"]
        _log(f"      moodboard: {moodboard.get('overall_aesthetic', '?')[:80]}")
        n_tiles = len(moodboard.get("tiles") or [])
        if n_tiles:
            _log(f"      moodboard: {n_tiles} tile(s) kept as text cues for storyboard")
        apply_direction()   # fold the moodboard into the steering for every stage below

    # ── 3/10  scenes (scenes←structure) ────────────────────────────────────────
    scene_target = duration_budget.suggest_scene_target(target_seconds)
    g = run_group("3/10", "scenes", [
        _spec("scenes",
              lambda: segment_scenes(source, structure, target=scene_target,
                                     profile=profile_override, characters=characters),
              _summarize_scenes,
              lambda fb, p=None: segment_scenes(source, structure, target=scene_target,
                                                profile=p or profile_override,
                                                feedback=fb, characters=characters),
              agent_module=scenes_agent),
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
                                                 feedback=fb, scenes=scenes),
              agent_module=casting_agent),
    ])
    casting = g["casting"]
    _log(f"      cast {len(casting.get('casting', []))}")

    # Render character + location portraits only for names appearing in the scenes
    # that will actually be rendered (1..max_scenes). Capping here avoids burning
    # API quota on characters/locations the video stage will never reference.
    # `--no-render` disables the actual API invocation only (dry_run=True) —
    # every prompt is still computed and its request params still logged to
    # gemini_api.log; only the network/SDK call itself is skipped.
    if imagegen.enabled():
        active_names: set[str] = set()
        for sc in scenes.get("scenes", [])[:max_scenes]:
            for nm in (sc.get("characters") or []):
                active_names.add(nm)
            if sc.get("location"):
                active_names.add(sc["location"])
            for p in (sc.get("props") or []):
                if p:
                    active_names.add(p.strip())
        _log(f"      {'recording intended API calls (--no-render) for' if not render else 'rendering'} "
             f"character/location/prop portraits — {len(active_names)} "
             f"name(s) in scene(s) 1..{_scenes_label(max_scenes)} …")
        if _render_casting_images(casting, out, active_names=active_names or None,
                                  dry_run=not render):
            save("casting", casting)
            _log(f"      portraits → {out}/casting/")
    elif not render:
        _log("      portraits skipped (--no-render; image backend also not configured)")

    # ── 5–7/10  soundscape + visuals + cinematography ────────────────────────
    shots_guidance = duration_budget.suggest_shots_per_scene(target_seconds, len(scenes.get("scenes", [])))
    g = run_group("5/10", "soundscape ‖ visuals ‖ cinematography", [
        _spec("soundscape",
              lambda: design_soundscape(structure, scenes, profile_override),
              _summarize_soundscape,
              lambda fb, p=None: design_soundscape(structure, scenes, p or profile_override, feedback=fb),
              realign=lambda result, keys: design_soundscape(structure, scenes, profile_override,
                                                              existing=result, revise_keys=keys),
              agent_module=soundscape_agent),
        _spec("visuals",
              lambda: design_visuals(structure, scenes, profile_override),
              _summarize_visuals,
              lambda fb, p=None: design_visuals(structure, scenes, p or profile_override, feedback=fb),
              realign=lambda result, keys: design_visuals(structure, scenes, profile_override,
                                                           existing=result, revise_keys=keys),
              agent_module=visuals_agent),
        _spec("cinematography",
              lambda: plan_cinematography(structure, scenes, profile_override, shots_guidance=shots_guidance),
              _summarize_cinematography,
              lambda fb, p=None: plan_cinematography(structure, scenes, p or profile_override, feedback=fb,
                                                      shots_guidance=shots_guidance),
              realign=lambda result, keys: plan_cinematography(structure, scenes, profile_override,
                                                                existing=result, revise_keys=keys,
                                                                shots_guidance=shots_guidance),
              agent_module=cinematography_agent),
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
    def _draft_realign(result, keys):
        return draft_screenplay(
            source, structure, characters, scenes,
            soundscape=soundscape, visuals=visuals, cinematography=cinematography,
            casting=casting, profile=profile_override,
            existing=result, revise_keys=keys,
        )
    g = run_group("8/10 screenplay (all scenes)", "draft", [
        _spec("screenplay", lambda: _draft(), _summarize_screenplay, _draft, realign=_draft_realign,
              agent_module=screenplay_agent),
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
    def _board_realign(result, keys):
        return plan_storyboard(
            structure, scenes, casting, soundscape, visuals, cinematography,
            characters=characters, draft=draft, genre=genre_spec,
            moodboard=moodboard, source=source, source_text=source.get("text", ""),
            profile=profile_override, out=out,
            existing=result, revise_keys=keys,
        )
    g = run_group("9/10", "storyboard", [
        _spec("storyboard", lambda: _board(),
              lambda r: _summarize_storyboard(r, target_seconds), _board, realign=_board_realign,
              agent_module=storyboard_agent),
    ])
    storyboard = g["storyboard"]

    # ── 10/11  video render — explicit stage: clips → per-scene video → movie ───
    # Depends on: storyboard (frames + prompts), casting (character images for seeding).
    # Produces: output/video/scene_NN/frame_MM.mp4  (frame clips)
    #           output/video/scene_NN.mp4            (per-scene stitch)
    #           output/video/movie.mp4               (final assembly)
    # Best-effort: skipped gracefully when no video backend is available.
    # `--no-render` disables the actual API invocation only (dry_run=True) —
    # every panel's Veo prompt/params is still computed and logged to
    # gemini_api.log; only the network/SDK call itself is skipped, so an
    # already-rendered scene's real clips are left untouched (see
    # `_render_one_panel`'s dry_run docstring).
    scene_render = {}
    if not i2v.enabled():
        _log(f"10/11 video render — skipped ({i2v.unavailable_hint()})")
    else:
        backend_label = i2v.backend()
        total_scenes = min(max_scenes, len(storyboard.get("storyboard", []))) if max_scenes else len(storyboard.get("storyboard", []))
        action = "recording intended Veo API calls (--no-render)" if not render else "video render"
        _log(f"10/11 {action}  [{backend_label}]  "
             f"({total_scenes} scene(s), all shots, per-scene stitch + final assembly) …")
        scene_render = _render_scene_frames(storyboard, casting, out, max_scenes=max_scenes,
                                            characters=characters, dry_run=not render)
        if scene_render:
            ok = scene_render.get("clips", 0)
            failed = scene_render.get("failed", 0)
            total_clips = ok + failed
            scenes_stitched = sum(1 for s in scene_render.get("scenes", []) if s.get("scene_video"))
            total_scenes_rendered = len(scene_render.get("scenes", []))
            movie = scene_render.get("movie")
            if not render:
                _log(f"      API params logged for {total_scenes_rendered} scene(s) — "
                     f"see {out}/logs/gemini_api.log"
                     + (f"; {scenes_stitched} scene video(s) already on disk from an earlier "
                        "render left untouched" if scenes_stitched else ""))
            elif failed and ok:
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


def _save_initial_response(out: Path, name: str, raw: dict, enabled: bool) -> None:
    """Write `raw` — the VERY FIRST result a stage produced this run, before
    scene-alignment self-heal, self-critique refinement, or any operator
    feedback touches it — to `output/<name>.0.json`, for reference (see
    `_gated`'s "Self-critique" docstring section and PROGRESS.md's session
    log). No-op when `enabled` is False: `run_group` passes
    `crit_on and s.get("agent_module") is not None` — i.e. skip writing this
    file when critique is disabled globally (config `critique.enabled`), OR
    when this particular stage has no `agent_module` (nothing would ever
    critique it, so the file would be a pointless, unused artifact)."""
    if not enabled:
        return
    _write_json(out / f"{name}.0.json", raw)
