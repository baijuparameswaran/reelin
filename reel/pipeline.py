"""Pipeline orchestration for the screenplay-material phase.

Phase graph:

    ingest ─┬─▶ structure ──┐
            └─▶ characters ──┴─▶ scenes ─┬─▶ soundscape ─────┐
                            └─▶ casting   ├─▶ visuals ─────────┼─▶ storyboard ─┐
                                          └─▶ cinematography ──┘               ├─▶ assemble
                                                              screenplay ──────┘
   (structure & characters concurrent; scenes & casting concurrent)
   (soundscape, visuals, cinematography concurrent)

Creative crew roles:
  casting       — locks each character's visual form as actor + character layers
                  (image-ready); can render them (stock photo → actor → character)
  soundscape    — background score / sound design
  visuals       — art production (color, props, production design)
  cinematography — Director of Photography (shot types, angles, movement, lens)
  storyboard    — fuses casting + art + camera + score into a visual image per moment

After storyboard + screenplay, an optional scene-render phase
(`_render_scene_frames` → `output/video/`) renders each storyboard frame to a
still then animates it into a clip (image-to-video via `reel.i2v`), chaining clips
for continuity within a scene. GPU-gated and best-effort: a no-op on CPU-only
hosts (stills still render where the image backend is available).

Each LLM stage passes through a human-in-the-loop gate: the operator can
approve the result, supply revision feedback, or let it auto-approve on
timeout. Parallel branches are gated independently after all complete.
HITL is controlled via `config/models.yaml` under the `hitl` key.
"""
from __future__ import annotations

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


def _log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


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
    return "\n".join(rows) or "  (none)"


def _summarize_casting(r: dict) -> str:
    rows = []
    for c in r.get("casting", []):
        actor = c.get("actor", c)
        character = c.get("character", c)
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


def _render_casting_images(casting: dict, out: Path) -> int:
    """Render ONE image per character — the **character representation** — via the
    image backend (Gemini). This is the only image generation in the pipeline; the
    character image is the reference handed to the video stage for identity. Stores
    the path on each entry. Idempotent (skips files on disk) for cheap --resume.

    The character look lives in the `character` block; older flat-schema casting
    (top-level visual_prompt) is tolerated.
    """
    if not imagegen.available():
        _log(f"      character renders skipped — {imagegen.unavailable_hint()}")
        return 0
    cast_dir = out / "casting"
    cast_dir.mkdir(exist_ok=True)
    n = 0
    for c in casting.get("casting", []):
        slug = _slug(c.get("name", "character"))
        kind = c.get("kind", "person")
        character = c.get("character") or {}
        target = character if character else c
        prompt = (character.get("visual_prompt")
                  or character.get("physical_form")
                  or c.get("visual_prompt") or c.get("physical_form")
                  or c.get("name", ""))
        if not prompt:
            _log(f"      ⚠ {c.get('name','?')} ({kind}) — no visual_prompt; skipping render")
            continue
        img = cast_dir / f"{slug}.png"
        _log(f"      rendering {c.get('name','?')} [{kind}] …")
        if img.exists() or imagegen.generate_image(prompt, img):
            target["image_path"] = str(img.relative_to(out))
            n += 1
    return n


def _render_moodboard_tiles(moodboard: dict, out: Path) -> int:
    """Render the moodboard's reference `tiles` into images via the image backend
    (Gemini when a key is configured, else the open image backend) → output/
    moodboard/tile_NN.png. The moodboard's palette + lighting are appended to each
    tile prompt so the board coheres as one look. Stores the path on each tile.
    Idempotent (skips files on disk). Best-effort — never blocks the run.

    NB policy: only the moodboard *tiles* (images) use the image provider; the
    moodboard spec itself is generated on the open text models like every stage."""
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
        if img.exists() or imagegen.generate_image(prompt, img):
            tile["image_path"] = str(img.relative_to(out))
            n += 1
    return n


def _frame_char_anchor(frame: dict, cast_index: dict, out: Path) -> Path | None:
    """The casting image of the first in-frame character — the identity anchor for
    a frame's still (so the right actor shows up, consistently)."""
    for name in frame.get("characters_in_frame", []):
        rel = cast_index.get(name)
        if rel and (out / rel).exists():
            return out / rel
    return None


def _render_scene_frames(storyboard: dict, casting: dict, out: Path,
                         max_scenes: int | None = None) -> dict:
    """Render each storyboard frame as a video clip (Veo image-to-video), then
    stitch each scene's clips into a per-scene video (output/video/scene_NN.mp4)
    and assemble all scene videos into the final movie (output/video/movie.mp4).

    Identity seeding: the first frame of each scene seeds from the in-frame
    character's representation image; subsequent frames chain from the previous
    clip's last frame for continuity within the scene. Scene boundary = hard cut.

    max_scenes caps how many scenes are rendered; every shot within each rendered
    scene is always included. Best-effort + idempotent (skips existing files).
    """
    if not i2v.enabled():
        _log(f"      scene render skipped — {i2v.unavailable_hint()}")
        return {}
    if not i2v.available():
        _log(f"      scene render skipped — {i2v.unavailable_hint()}")
        return {}
    continuity = bool(i2v._cfg().get("continuity", True))

    cast_index = {}
    for c in casting.get("casting", []):
        ch = c.get("character", c)
        rel = ch.get("image_path") or c.get("image_path")
        if rel:
            cast_index[c.get("name")] = rel

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
        frames_out = []
        # `panels` is the new schema; fall back to `frames` for old checkpoints
        panels = scene.get("panels") or scene.get("frames", [])

        for fr in panels:
            fnum = fr.get("panel") or fr.get("frame", len(frames_out) + 1)
            prompt = fr.get("image_prompt") or fr.get("action") or fr.get("moment", "")
            tag = f"{int(fnum):02d}" if isinstance(fnum, int) else str(fnum)

            # Seed: continue from the previous frame's tail (carries the look
            # forward); the first frame of a scene seeds from the in-frame
            # character's representation image (identity reference).
            seed = prev_tail if (prev_tail and continuity) else _frame_char_anchor(fr, cast_index, out)
            clip = sdir / f"frame_{tag}.mp4"
            tail_img = sdir / f"frame_{tag}_tail.png"
            if not clip.exists():
                if i2v.generate_clip([seed] if seed else [], prompt, clip):
                    manifest["clips"] += 1
                    # Always extract the tail frame so every clip has one on disk.
                    # Continuity chains it forward as the next-clip seed;
                    # ffmpeg stitching benefits from having clean cut-points regardless.
                    tail = i2v.last_frame(clip, tail_img)
                    if continuity:
                        prev_tail = tail or seed
                else:
                    manifest["failed"] += 1
                    _log(f"      ⚠ scene {snum} frame {tag} — clip not produced")

            frames_out.append({
                "panel": fnum,
                "shot_type": fr.get("shot_type", ""),
                "action": fr.get("action") or fr.get("moment", ""),
                "seed": str(Path(seed).relative_to(out)) if seed and Path(seed).exists() else None,
                "clip": str(clip.relative_to(out)) if clip.exists() else None,
            })

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
        result = rerun_fn(decision.feedback, current_profile)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    input_path: str,
    out_dir: str = "output",
    max_scenes: int = 1,
    profile_override: str | None = None,
    resume: bool = False,
    genre: str | None = None,
) -> dict:
    """Run the full screenplay-material phase and write artifacts to `out_dir`.

    Pause anytime by typing 'stop' at a review gate (or Ctrl-C); every stage
    already approved stays on disk. Re-run with `resume=True` to load those
    checkpoints and continue from the first stage that hasn't been completed.
    """
    # Clear any direction left over from a previous run that may have crashed or
    # been interrupted before reaching the llm.set_direction(None) at the end.
    llm.set_direction(None)

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

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
    thinking_model = _resolved("thinking")
    quality_high_model = _resolved("quality_high")
    _log(f"models — fast: {fast_model} | quality: {quality_model} | "
         f"thinking: {thinking_model} | quality_high: {quality_high_model}")
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

    # ── 3–4/10  scenes + casting (scenes←structure, casting←characters) ───────
    g = run_group("3/10", "scenes ‖ casting", [
        _spec("scenes",
              lambda: segment_scenes(source, structure, profile=profile_override),
              _summarize_scenes,
              lambda fb, p=None: segment_scenes(source, structure, profile=p or profile_override, feedback=fb)),
        _spec("casting",
              lambda: cast_characters(structure, characters, profile_override),
              _summarize_casting,
              lambda fb, p=None: cast_characters(structure, characters, p or profile_override, feedback=fb)),
    ])
    scenes, casting = g["scenes"], g["casting"]
    _log(f"      {len(scenes.get('scenes', []))} scenes; cast {len(casting.get('casting', []))}")

    # Render a basic portrait per character from its casting visual_prompt and
    # store the path in the casting details (part of the casting stage).
    if imagegen.enabled():
        _log("      rendering character portraits …")
        if _render_casting_images(casting, out):
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

    # ── 8/10  screenplay draft ────────────────────────────────────────────────
    def _draft(fb=None, p=None):
        return draft_screenplay(
            source, structure, characters, scenes,
            soundscape=soundscape, visuals=visuals, cinematography=cinematography,
            casting=casting, max_scenes=max_scenes, profile=p or profile_override, feedback=fb,
        )
    g = run_group(f"8/10 screenplay (first {max_scenes} scenes)", "draft", [
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
        scene_render = _render_scene_frames(storyboard, casting, out, max_scenes=max_scenes)
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

    _log(f"done in {project['elapsed_seconds']}s → {out}/")
    return project


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
