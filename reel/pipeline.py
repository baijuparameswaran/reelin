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
        frames = s.get("frames", [])
        rows.append(f"  Scene {s.get('scene_number','?'):>2}: {len(frames)} frames")
        for f in frames[:2]:
            rows.append(
                f"      f{f.get('frame','?')} {f.get('moment','?')[:50]}"
                f"  [{f.get('emotional_attribute','')[:24]} / "
                f"{f.get('audio_attribute','')[:24]}]"
            )
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
        character = c.get("character") or {}
        target = character if character else c
        prompt = (character.get("visual_prompt")
                  or character.get("physical_form")
                  or c.get("visual_prompt") or c.get("physical_form")
                  or c.get("name", ""))
        if not prompt:
            continue
        img = cast_dir / f"{slug}.png"
        if img.exists() or imagegen.generate_image(prompt, img):
            target["image_path"] = str(img.relative_to(out))
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


def _render_scene_frames(storyboard: dict, casting: dict, out: Path) -> dict:
    """Next phase (after storyboard + screenplay): render each storyboard frame as
    a **video clip** (Veo image-to-video), seeded for the first frame of a scene by
    the in-frame character's representation image (the reference from the image
    stage), and **chained from the previous frame's last image** so motion is
    continuous within the scene. Scene boundaries reset the chain (a cut). No
    intermediate stills are generated — image generation is reserved for the
    character representation. Best-effort, idempotent. Returns a render manifest.
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
    manifest = {"continuity": continuity, "clips": 0, "scenes": []}

    for scene in storyboard.get("storyboard", []):
        snum = scene.get("scene_number", "x")
        sdir = vdir / f"scene_{snum:02d}" if isinstance(snum, int) else vdir / f"scene_{snum}"
        sdir.mkdir(exist_ok=True)
        prev_tail = None                      # reset each scene → hard cut between scenes
        frames_out = []
        for fr in scene.get("frames", []):
            fnum = fr.get("frame", len(frames_out) + 1)
            prompt = fr.get("image_prompt") or fr.get("image") or fr.get("moment", "")
            tag = f"{int(fnum):02d}" if isinstance(fnum, int) else str(fnum)

            # Seed: continue from the previous frame's tail (carries the look
            # forward); the first frame of a scene seeds from the in-frame
            # character's representation image (identity reference).
            seed = prev_tail if (prev_tail and continuity) else _frame_char_anchor(fr, cast_index, out)
            clip = sdir / f"frame_{tag}.mp4"
            if not clip.exists():
                if i2v.generate_clip([seed] if seed else [], prompt, clip):
                    manifest["clips"] += 1
                    if continuity:
                        prev_tail = i2v.last_frame(clip, sdir / f"frame_{tag}_tail.png") or seed

            frames_out.append({
                "frame": fnum,
                "moment": fr.get("moment"),
                "seed": str(Path(seed).relative_to(out)) if seed and Path(seed).exists() else None,
                "clip": str(clip.relative_to(out)) if clip.exists() else None,
            })
        manifest["scenes"].append({"scene_number": snum, "frames": frames_out})

    (vdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


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

def _gated(
    gate: Gate,
    name: str,
    initial_result: dict,
    summarize_fn: Callable,
    rerun_fn: Callable,  # rerun_fn(feedback: str) -> dict
) -> dict:
    """Show gate for initial_result; re-run with feedback until approved.

    Raises PipelineStopped if the operator chooses to pause at the gate.
    """
    result = initial_result
    while True:
        decision = gate.review(name, result, summarize_fn)
        if decision.approved:
            return result
        if decision.stop:
            raise PipelineStopped(name)
        _log(f"      re-running {name} with feedback …")
        result = rerun_fn(decision.feedback)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    input_path: str,
    out_dir: str = "output",
    max_scenes: int = 3,
    profile_override: str | None = None,
    resume: bool = False,
) -> dict:
    """Run the full screenplay-material phase and write artifacts to `out_dir`.

    Pause anytime by typing 'stop' at a review gate (or Ctrl-C); every stage
    already approved stays on disk. Re-run with `resume=True` to load those
    checkpoints and continue from the first stage that hasn't been completed.
    """
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Persist each stage as soon as it's approved, so a failure, timeout, or pause
    # in a later (slow) stage never discards completed work.
    def save(name: str, data: dict) -> None:
        _write_json(out / f"{name}.json", data)

    gate = Gate.from_config(llm.config())
    parallel = llm.config().get("runtime", {}).get("max_parallel_agents", 1) > 1

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
            r = _gated(gate, nm, raws[nm], s["summarize"], s["rerun"])
            save(nm, r)
            results[nm] = r
        return results

    fast_model = llm.resolve_model(llm.get_profile(profile_override or "fast"))
    quality_model = llm.resolve_model(llm.get_profile(profile_override or "quality"))
    _log(f"models — fast: {fast_model} | quality: {quality_model}")
    if resume:
        _log(f"resume: loading any completed stages from {out}/")

    # ── 1/10  ingest (deterministic — no gate) ───────────────────────────────
    _log("1/10 ingest …")
    source = ingest(input_path)
    _log(f"      '{source['title']}' — {source['word_count']} words")

    # ── 2/10  structure + characters ─────────────────────────────────────────
    g = run_group("2/10", "structure ‖ characters", [
        _spec("structure",
              lambda: analyze_structure(source, profile_override),
              _summarize_structure,
              lambda fb: analyze_structure(source, profile_override, feedback=fb)),
        _spec("characters",
              lambda: extract_characters(source, profile_override),
              _summarize_characters,
              lambda fb: extract_characters(source, profile_override, feedback=fb)),
    ])
    structure, characters = g["structure"], g["characters"]
    _log(f"      logline: {structure.get('logline', '(parse failed)')[:80]}")
    _log(f"      characters: {len(characters.get('characters', []))}")

    # ── 3–4/10  scenes + casting (scenes←structure, casting←characters) ───────
    g = run_group("3/10", "scenes ‖ casting", [
        _spec("scenes",
              lambda: segment_scenes(source, structure, profile=profile_override),
              _summarize_scenes,
              lambda fb: segment_scenes(source, structure, profile=profile_override, feedback=fb)),
        _spec("casting",
              lambda: cast_characters(structure, characters, profile_override),
              _summarize_casting,
              lambda fb: cast_characters(structure, characters, profile_override, feedback=fb)),
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
              lambda fb: design_soundscape(structure, scenes, profile_override, feedback=fb)),
        _spec("visuals",
              lambda: design_visuals(structure, scenes, profile_override),
              _summarize_visuals,
              lambda fb: design_visuals(structure, scenes, profile_override, feedback=fb)),
        _spec("cinematography",
              lambda: plan_cinematography(structure, scenes, profile_override),
              _summarize_cinematography,
              lambda fb: plan_cinematography(structure, scenes, profile_override, feedback=fb)),
    ])
    soundscape, visuals, cinematography = g["soundscape"], g["visuals"], g["cinematography"]

    # ── 8/10  screenplay draft ────────────────────────────────────────────────
    def _draft(fb=None):
        return draft_screenplay(
            source, structure, characters, scenes,
            soundscape=soundscape, visuals=visuals, cinematography=cinematography,
            max_scenes=max_scenes, profile=profile_override, feedback=fb,
        )
    g = run_group(f"8/10 screenplay (first {max_scenes} scenes)", "draft", [
        _spec("screenplay", lambda: _draft(), _summarize_screenplay, _draft),
    ])
    draft = g["screenplay"]
    fountain = to_fountain(source, structure, draft)
    (out / "screenplay.fountain").write_text(fountain, encoding="utf-8")

    # ── 9/10  storyboard (fuses casting + art + camera + score per moment) ────
    def _board(fb=None):
        return plan_storyboard(
            structure, scenes, casting, soundscape, visuals, cinematography,
            profile=profile_override, feedback=fb,
        )
    g = run_group("9/10", "storyboard", [
        _spec("storyboard", lambda: _board(), _summarize_storyboard, _board),
    ])
    storyboard = g["storyboard"]

    # Next phase: with storyboard + screenplay done, render scenes frame by frame
    # (still per frame → image-to-video clip), chaining clips for continuity. Best-
    # effort: a no-op on hosts without a video backend (see config `video`).
    scene_render = {}
    if i2v.enabled():
        _log("      rendering scenes frame by frame (image-to-video) …")
        scene_render = _render_scene_frames(storyboard, casting, out)
        if scene_render:
            _log(f"      clips → {out}/video/ ({scene_render.get('clips', 0)} clips)")

    # ── 10/10  assemble artifacts ─────────────────────────────────────────────
    # Per-stage JSON + screenplay.fountain are already written above (incremental,
    # crash-safe). Here we add the per-character files and the combined manifest.
    _log("10/10 assemble artifacts …")

    chars_dir = out / "characters"
    chars_dir.mkdir(exist_ok=True)
    for char in characters.get("characters", []):
        _write_json(chars_dir / f"{_slug(char.get('name', 'unknown'))}.json", char)

    project = {
        "title": source["title"],
        "source": source["source_path"],
        "word_count": source["word_count"],
        "structure": structure,
        "characters": characters,
        "casting": casting,
        "scenes": scenes,
        "soundscape": soundscape,
        "visuals": visuals,
        "cinematography": cinematography,
        "storyboard": storyboard,
        "screenplay_draft": draft,
        "scene_render": scene_render,
        "models": {"fast": fast_model, "quality": quality_model},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    save("project", project)

    _log(f"done in {project['elapsed_seconds']}s → {out}/")
    return project


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
