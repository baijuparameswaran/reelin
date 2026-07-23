"""Command-line entry point for the reel screenplay-material pipeline.

Usage:
    python -m reel.cli SOURCE.txt [--out DIR] [--max-scenes N] [--profile NAME]
        [--target-duration N] [--gate-timeout N] [--resume] [--no-render]
    python -m reel.cli --list-models             # show local model status
    python -m reel.cli stages                    # list pipeline stages + their inputs
    python -m reel.cli stage NAME [SOURCE.txt]   # run ONE stage independently
    python -m reel.cli render [--fresh]          # render the whole drafted story to video
                                                 # (screenplay.fountain + cinematography.json)
    python -m reel.cli stitch                    # concatenate rendered clips → one movie.mp4
    python -m reel.cli gen-video "PROMPT"        # generate one clip directly from a prompt
        [--image PATH]                           #   optional seed image (image-to-video)
        [--out PATH]                             #   output .mp4 (default: output/gen_video_<ts>.mp4)
        [--model NAME] [--aspect-ratio 16:9]     #   override model / aspect ratio
        [--duration N]                           #   clip duration in seconds
    python -m reel.cli revise [--out DIR] [--render] [--render-images] [--render-video]
                                                 # revise a completed/paused run: pick any
                                                 # stage (or the raw source text) to hand-edit —
                                                 # modify existing content, add a new scene/
                                                 # character/location, or delete one — then
                                                 # selectively re-run what's actually affected;
                                                 # repeat rounds until 'quit'/'pause'. Deleting a
                                                 # scene from scenes.json removes it everywhere
                                                 # downstream (design artifacts + video manifest),
                                                 # not just from scenes.json itself.
                                                 # Casting + all image/video rendering are SKIPPED
                                                 # by default (cheap text-only iteration). Image
                                                 # rendering (casting + casting images + moodboard
                                                 # tiles) and video rendering (scene_render) can be
                                                 # enabled INDEPENDENTLY: --render-images / --render-video,
                                                 # or --render for both; inside the loop, 'render
                                                 # images on/off' / 'render video on/off' toggle
                                                 # each separately, or 'render on'/'render off' for
                                                 # both at once.
                                                 # Always operates as if --max-scenes all had been
                                                 # used, regardless of the original run's cap —
                                                 # only the diff-identified scenes actually get
                                                 # regenerated either way. Prints the identified
                                                 # scope as soon as it's known, and what each
                                                 # downstream stage is doing as it runs. Every
                                                 # stage the cascade regenerates gets the SAME
                                                 # review gate a fresh pipeline run's stages get
                                                 # (fidelity/genre score, feedback-driven re-run,
                                                 # 'view' to edit, auto-escalation, auto-approve
                                                 # timeout) — not a silent auto-apply.
    python -m reel.cli veo-sync                  # refresh Veo prompt guide snapshot
        [--status]                               #   print cache status only (no fetch)

Run via the package so relative imports resolve: `python -m reel.cli ...`.

Each stage can be invoked on its own (`stage NAME`): it loads the inputs it needs
from prior `output/<input>.json` checkpoints (ingesting SOURCE on demand) and
writes its own artifact — re-run a single stage without the whole pipeline. (Stage
runs are direct, with no HITL gate; use `--feedback` to pass a revision note.)

The pipeline pauses for human review after each stage (approve with Enter,
type feedback to re-run that stage, 'view' to open the full output in
$EDITOR/$VISUAL — default vim — to either just read it or edit and save it,
or 'stop' to pause). Saving a real change there is re-checked against
fidelity/genre and brings the same gate back up rather than being
auto-approved outright; closing without saving just reprompts. The
auto-approve timeout isn't running while you're inside the editor. Toggle
this in `config/models.yaml` under `hitl` (set `enabled: false` for unattended
runs; tune `timeout_seconds` for the auto-approve fallback).

Each approved stage is checkpointed to `output/<stage>.json`. After a pause
(typing 'stop', Ctrl-C) or a failure, re-run with `--resume` to reload the
finished stages and continue from the first one that isn't done.
"""
from __future__ import annotations

import argparse
import sys

from . import llm
from . import session
from . import artifact_diff
from .pipeline import run, PipelineStopped


def _max_scenes_arg(v: str) -> int | None:
    """--max-scenes value: an integer count, or the literal 'all' (case-insensitive)
    for every drafted scene, unbounded (None — downstream slicing `[:None]`
    naturally means "no cap" throughout the pipeline)."""
    if v.strip().lower() == "all":
        return None
    return int(v)


def _scenes_label(max_scenes: int | None) -> str:
    return "all" if max_scenes is None else str(max_scenes)


# Distinct from `_max_scenes_arg`'s own `None` (which means the explicit,
# meaningful value "all") — this sentinel means "the --max-scenes flag was
# not given at all", so `main()` can tell "explicitly asked for all" apart
# from "didn't say, inherit whatever the previous run used if resuming".
_MAX_SCENES_UNSET = object()


def _run_params_path(out):
    from pathlib import Path
    return Path(out) / "run_params.json"


def _load_run_params(out) -> dict:
    """Best-effort read of the CLI-level knobs (max_scenes/profile/genre/
    target_duration/render) the previous full-pipeline run at `out` was
    actually invoked with — written by `_save_run_params` below. Missing/
    corrupt file (e.g. a pre-existing
    `--out` from before this existed, or one only ever touched by standalone
    `stage`/`revise` commands) degrades to an empty dict, never raises —
    callers treat that the same as "no prior record, use the normal
    argparse default", never as an error."""
    import json
    p = _run_params_path(out)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_run_params(out, *, max_scenes, profile, genre, target_duration=None,
                     render: bool = True) -> None:
    """Persist the EFFECTIVE (already-resolved, post-inheritance) knobs a
    full-pipeline run used, so a later `--resume` that omits `--max-scenes`/
    `--profile`/`--target-duration`/`--no-render` inherits the same values
    instead of silently falling back to argparse's own defaults (1 scene, no
    profile override, config's default target runtime, rendering on) —
    which would otherwise be a correctness gap, not just a UX one: an
    unfinished `--max-scenes all` run resumed bare would only render scene
    1's worth of casting images/video from then on, and a `--no-render`
    run resumed bare would silently start spending real API quota on a
    resume that was explicitly meant to stay text-only. Re-written on every
    run (fresh or resumed) with whatever was actually used THIS time, so
    the stored value stays current across any number of resumes and an
    explicit override on one resume becomes the new inherited default for
    the next. `render` defaults to `True` (rendering on) — matches this
    project's pre-`--no-render` behavior for every existing caller that
    doesn't pass it explicitly (standalone `stage`/`revise` invocations,
    and any run_params.json predating this field)."""
    import json
    p = _run_params_path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"max_scenes": max_scenes, "profile": profile, "genre": genre,
                            "target_duration": target_duration, "render": render},
                            ensure_ascii=False, indent=2), encoding="utf-8")


def _restore_direction(out) -> None:
    """Reload genre.json/moodboard.json (if present) and re-apply the exact
    same creative-direction steering `pipeline.run()` set up for the
    original run, via the shared `pipeline.compose_direction`. `revise` is a
    separate process invocation — `llm.set_direction`'s process-wide
    directive is unset by default in a fresh process, so without this every
    stage regenerated through `revise` would silently lose genre/moodboard
    steering even though the original run's choices are sitting right there
    on disk. Best-effort: missing genre.json/moodboard.json (a run that
    disabled one, or never got that far) just means that part of the
    direction is empty — never raises."""
    from .stages import _load
    from .pipeline import compose_direction
    genre_spec = _load(out, "genre") or {}
    moodboard_spec = _load(out, "moodboard") or {}
    llm.set_direction(compose_direction(genre_spec, moodboard_spec))


def _effective_max_scenes(stage_name: str, max_scenes: int | None) -> int | None:
    """`--max-scenes` ONLY restricts actual media-rendering stages in a full
    pipeline run (casting-image generation, scene_render's video generation,
    and moodboard's render-ready tiles) — every design/planning stage,
    screenplay included, always drafts every scene regardless (see
    PROGRESS.md's "Current state" — `pipeline.run()` deliberately never
    passes its render-scoped `max_scenes` into the screenplay call). `revise`
    must match that policy rather than silently reintroducing a cap via
    `run_stage`'s own non-None default: `screenplay` always gets `None`
    (uncapped) here regardless of the inherited value; every other stage
    gets the inherited value as-is (harmless for stages that ignore it)."""
    return None if stage_name == "screenplay" else max_scenes


def _duration_kwargs(stage_name: str, out, target_duration: int | None) -> dict:
    """`target`/`shots_guidance` kwargs for `stages.run_stage`, restoring the
    ORIGINAL run's `--target-duration` planning guidance for the two stages
    that actually consume it — `scenes` (scene-count budget) and
    `cinematography` (shots-per-scene budget, which additionally needs the
    CURRENT scene count, reloaded fresh since `scenes` may have just been
    regenerated earlier in the same revision). Empty dict for every other
    stage — nothing meaningful to compute, and every other stage's `**_`
    catch-all would ignore these anyway. `target_duration=None` (no prior
    record, or the original run used the config default) falls back to
    `duration_budget.DEFAULT_TARGET_SECONDS`, matching what a fresh
    `pipeline.run()` does when `--target-duration` is omitted."""
    if stage_name not in ("scenes", "cinematography"):
        return {}
    from . import duration_budget
    from .stages import _load
    target_seconds = target_duration or duration_budget.DEFAULT_TARGET_SECONDS
    if stage_name == "scenes":
        return {"target": duration_budget.suggest_scene_target(target_seconds)}
    scenes_doc = _load(out, "scenes") or {}
    scene_count = len(scenes_doc.get("scenes", []))
    return {"shots_guidance": duration_budget.suggest_shots_per_scene(target_seconds, scene_count)}


def _list_models() -> int:
    cfg = llm.config()
    have = llm.installed_models()
    print(f"Ollama host: {llm.host()}")
    print(f"Installed models ({len(have)}): {', '.join(have) or '(none)'}\n")
    for name, prof in cfg["profiles"].items():
        p = llm.get_profile(name)
        try:
            resolved = llm.resolve_model(p)
            mark = "✓ preferred" if resolved == p.model else f"↳ fallback ({resolved})"
        except RuntimeError as e:
            resolved, mark = "—", f"✗ {e}"
        print(f"  profile '{name}': wants {p.model:<18} → {mark}")
    return 0


def _list_stages() -> int:
    from .stages import STAGES
    print("Pipeline stages — invoke one with:  python -m reel.cli stage NAME [SOURCE]\n")
    for s in STAGES:
        ins = ", ".join(s.inputs) + (f" (+{', '.join(s.optional)})" if s.optional else "")
        print(f"  {s.name:<16} inputs: {ins or '—':<46} → {s.artifact()}.json")
        if s.desc:
            print(f"  {'':16} {s.desc}")
    return 0


def _run_stage(argv: list[str]) -> int:
    from .stages import REGISTRY, names, run_stage
    ap = argparse.ArgumentParser(prog="reel stage",
                                 description="run one pipeline stage independently")
    ap.add_argument("name", help="stage name (see: reel stages)")
    ap.add_argument("source", nargs="?", help="source file (for ingest / first run)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--profile", choices=["fast", "quality"], default=None)
    ap.add_argument("--max-scenes", type=_max_scenes_arg, default=1,
                    help="scene count, or 'all' for every drafted scene")
    ap.add_argument("--feedback", default=None, help="revision note passed to the agent")
    a = ap.parse_args(argv)
    if a.name not in REGISTRY:
        ap.error(f"unknown stage '{a.name}'. Known: {', '.join(names())}")
    try:
        run_stage(a.name, out=a.out, input_path=a.source, profile=a.profile,
                  feedback=a.feedback, max_scenes=a.max_scenes)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[reel] {e}")
        return 2
    print(f"[reel] stage '{a.name}' done → {a.out}/{REGISTRY[a.name].artifact()}.json")
    return 0


def _load_json(path) -> dict:
    import json
    from pathlib import Path
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _render_video(argv: list[str]) -> int:
    """Render scene clips from screenplay.fountain + cinematography.json via Veo.

    By default renders the WHOLE drafted story — every scene the screenplay drafted
    and every action beat within it (camera grammar from cinematography.json) — no
    artificial caps. Works purely off existing artifacts; no LLM stage runs.
    Also loads scenes.json (if present) for each scene's authoritative
    `location` — see `fountain.to_storyboard`'s `scenes_json` param; this
    command still runs without it (degraded location detection), since its
    own contract is "works purely off existing artifacts" and scenes.json
    isn't a hard requirement, just an accuracy improvement when available."""
    import json
    import shutil
    from pathlib import Path

    from . import fountain, i2v, gemini, session
    from .pipeline import _render_scene_frames

    ap = argparse.ArgumentParser(prog="reel render",
                                 description="render scene videos from screenplay.fountain + cinematography.json")
    ap.add_argument("--out", default="output")
    ap.add_argument("--max-scenes", type=_max_scenes_arg, default=None,
                    help="optional cap on scenes, or 'all' (default: all drafted scenes)")
    ap.add_argument("--max-shots", type=int, default=None,
                    help="optional cap on shots per scene (default: every action beat)")
    ap.add_argument("--fresh", action="store_true",
                    help="re-render existing clips (clears output/video first; "
                         "old clips are backed up to output/video_prev)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    session.start(out, fresh=False)
    gemini.set_log_dir(out)

    fpath = out / "screenplay.fountain"
    if not fpath.exists():
        print(f"[reel] no {fpath} — run the pipeline (or `stage screenplay`) first")
        return 2
    if not i2v.available():
        print(f"[reel] video backend unavailable — {i2v.unavailable_hint()}")
        return 2

    scenes = fountain.parse(fpath.read_text(encoding="utf-8"))
    board = fountain.to_storyboard(
        scenes,
        _load_json(out / "soundscape.json"),
        _load_json(out / "visuals.json"),
        _load_json(out / "casting.json"),
        out, max_scenes=a.max_scenes, max_shots=a.max_shots,
        cinematography=_load_json(out / "cinematography.json"),
        scenes_json=_load_json(out / "scenes.json"),
    )
    (out / "storyboard.json").write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    nshots = sum(len(s["panels"]) for s in board["storyboard"])
    print(f"[reel] render plan: {len(board['storyboard'])} scene(s), {nshots} shot(s) "
          f"(story-defined, camera-directed from cinematography.json) → {out}/storyboard.json")

    if a.fresh and (out / "video").exists():
        backup = out / "video_prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(out / "video"), str(backup))
        print(f"[reel] cleared existing clips → backed up to {backup}/")

    manifest = _render_scene_frames(board, _load_json(out / "casting.json"), out,
                                    max_scenes=a.max_scenes,
                                    characters=_load_json(out / "characters.json"))
    print(f"[reel] rendered {manifest.get('clips', 0)} new clip(s) → {out}/video/")
    if manifest.get("movie"):
        print(f"[reel] movie → {out}/{manifest['movie']}")
    _print_spend_summary(out)
    return 0


def _gen_video_prompt(argv: list[str]) -> int:
    """Generate a single video clip directly from a prompt (no pipeline needed).

    Uses the same backend as the pipeline (Gemini Veo when a key is set, else
    the configured open_backend).  Handy for quick tests, stand-alone shots, or
    iterating on a prompt before wiring it into a storyboard.

    Examples
    --------
    # text-to-video
    python -m reel.cli gen-video "Wide shot of a city street at dusk, rain falling"

    # image-to-video (seed image drives the first frame)
    python -m reel.cli gen-video "The protagonist steps outside into the wind" \\
        --image output/casting/character.png

    # override model / aspect ratio
    python -m reel.cli gen-video "Crashing waves at sunset" \\
        --model veo-3.1-generate-preview --aspect-ratio 9:16 --out clips/waves.mp4
    """
    import datetime
    from pathlib import Path
    from . import gemini, i2v

    ap = argparse.ArgumentParser(
        prog="reel gen-video",
        description="generate a video clip directly from a prompt",
    )
    ap.add_argument("prompt", help="text prompt for the video")
    ap.add_argument("--image", default=None, metavar="PATH",
                    help="seed image for image-to-video (optional; Veo also works text-only)")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="output .mp4 path (default: output/gen_video_<timestamp>.mp4)")
    ap.add_argument("--model", default=None,
                    help="override video model (e.g. veo-3.1-generate-preview)")
    ap.add_argument("--aspect-ratio", default=None, dest="aspect_ratio",
                    help="aspect ratio: 16:9 (default) | 9:16 | 1:1")
    ap.add_argument("--duration", type=int, default=None,
                    help="clip duration in seconds (Veo default: 8)")
    a = ap.parse_args(argv)

    if not i2v.available():
        print(f"[reel] video backend unavailable — {i2v.unavailable_hint()}")
        return 2

    # Resolve output path.
    if a.out:
        out_path = Path(a.out)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("output") / f"gen_video_{ts}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gemini.set_log_dir(out_path.parent if a.out else Path("output"))

    # Pull config defaults, allow per-call overrides.
    cfg = i2v._cfg()
    model       = a.model        or cfg.get("model", "veo-3.1-generate-preview")
    aspect      = a.aspect_ratio or cfg.get("aspect_ratio", "16:9")
    resolution  = cfg.get("resolution", "720p")
    poll        = cfg.get("poll_seconds", 10)
    timeout     = cfg.get("timeout_seconds", 1200) or 1200
    duration    = a.duration     or 8

    image_path = Path(a.image) if a.image else None
    if image_path and not image_path.exists():
        print(f"[reel] seed image not found: {image_path}")
        return 2

    from . import veo_guide
    prompt = i2v._full_prompt(a.prompt)
    # Pre-flight Veo guide check — logged as warnings, never blocks generation.
    report = veo_guide.verify_prompt(prompt)
    if report["issues"] or report["warnings"]:
        if report["issues"]:
            print("[reel] ⚠  Veo prompt — issues:", flush=True)
            for iss in report["issues"]:
                print(f"[reel]    • {iss}", flush=True)
        if report["warnings"]:
            print("[reel]    Veo prompt — advisory:", flush=True)
            for w in report["warnings"]:
                print(f"[reel]    ℹ {w}", flush=True)
    mode = "image-to-video" if image_path else "text-to-video"
    print(f"[reel] gen-video  {mode}  model={model}  aspect={aspect}", flush=True)
    print(f"[reel]   prompt: {a.prompt[:120]}", flush=True)
    if image_path:
        print(f"[reel]   image:  {image_path}", flush=True)
    print(f"[reel]   out:    {out_path}", flush=True)

    ok = gemini.generate_video(
        prompt, out_path,
        image_path=image_path,
        model=model,
        aspect_ratio=aspect,
        resolution=resolution,
        duration_seconds=duration,
        poll_seconds=poll,
        timeout_seconds=timeout,
    )
    if ok:
        print(f"[reel] done → {out_path}")
        return 0
    print("[reel] video generation failed — check logs above")
    return 1


def _veo_sync(argv: list[str]) -> int:
    """Refresh the Veo prompt guide snapshot from ai.google.dev.

    Fetches the guide, checks for content changes, and logs exactly which
    source files need review when the guide has been updated.  Automatically
    tries alternate URLs if the primary path has moved.

    The snapshot is stored in config/veo_guide_snapshot.json and is auto-
    checked at pipeline startup when it is older than 14 days.
    """
    from . import veo_guide

    ap = argparse.ArgumentParser(
        prog="reel veo-sync",
        description="refresh Veo prompt guide snapshot from ai.google.dev",
    )
    ap.add_argument("--status", action="store_true",
                    help="print cache status only — do not fetch")
    a = ap.parse_args(argv)

    if a.status:
        print(f"[reel/veo-guide] {veo_guide.status()}")
        return 0

    veo_guide.sync(force=True, quiet=False)
    return 0


def _stitch(argv: list[str]) -> int:
    """Stitch already-rendered scene clips into one movie (no rendering)."""
    import json
    from pathlib import Path

    from .pipeline import _assemble_movie

    ap = argparse.ArgumentParser(prog="reel stitch",
                                 description="concatenate rendered scene clips into a single movie")
    ap.add_argument("--out", default="output")
    a = ap.parse_args(argv)
    out = Path(a.out)
    mpath = out / "video" / "manifest.json"
    if not mpath.exists():
        print(f"[reel] no {mpath} — render scenes first (pipeline run, or `reel.cli render`)")
        return 2
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    movie = _assemble_movie(manifest, out)
    if not movie:
        print("[reel] nothing stitched (no clips, or ffmpeg unavailable)")
        return 1
    manifest["movie"] = str(movie.relative_to(out))
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reel] movie → {movie}")
    return 0


_SCENE_KEYED_STAGES = artifact_diff.SCENE_KEYED_ARTIFACTS
_NAME_KEYED_STAGES = artifact_diff.NAME_KEYED_ARTIFACTS


def _translate_revise_keys(from_stage: str, to_stage: str, revise_keys, edited_from_artifact: dict):
    """Map a revision's scope from `from_stage`'s key type to `to_stage`'s.
    `None` means "drastic/whole-file — full regen everywhere", propagated
    unchanged. Same key type (scene_number-keyed -> scene_number-keyed, or
    name-keyed -> name-keyed) propagates `revise_keys` as-is. The one
    cross-type case worth translating: a `scenes` edit reaching `casting`
    (number-keyed -> name-keyed) — only the LOCATION names referenced by the
    revised scene numbers could plausibly need re-casting (a scenes edit
    can't affect a CHARACTER's casting at all). Any other cross-type
    combination has no known translation, so that one downstream stage falls
    back to a full, non-scoped regen — still correct, just not maximally
    scoped, rather than being silently skipped.

    An EMPTY (but non-None) `revise_keys` — e.g. a scenes.json revision that
    only DELETED scene(s), with nothing added/changed — must translate to
    an empty scope too, not `None`/drastic: there's nothing for `to_stage`
    to regenerate either. Checked before the "no locations found" case
    below, which legitimately still means "fall back to full regen,
    ambiguous" for a genuinely non-empty `revise_keys`."""
    if revise_keys is None:
        return None
    if not revise_keys:
        return set()
    if from_stage in _SCENE_KEYED_STAGES and to_stage in _SCENE_KEYED_STAGES:
        return revise_keys
    if from_stage in _NAME_KEYED_STAGES and to_stage in _NAME_KEYED_STAGES:
        return revise_keys
    if from_stage == "scenes" and to_stage == "casting":
        revised = [s for s in edited_from_artifact.get("scenes", []) if s.get("number") in revise_keys]
        locs = {s.get("location") for s in revised if s.get("location")}
        return locs or None
    return None


def _names_to_scene_numbers(out, names) -> set:
    """Map a set of character/location names to the scene numbers they
    actually appear in, via storyboard.json's panels (`characters_in_frame`)
    if available, else scenes.json's `characters` field. Used to translate a
    name-keyed casting/characters revision into the scene-number-keyed scope
    `_apply_scene_render_revision` needs — a character's casting entry
    (physical_form etc, which feeds every panel's Veo Subject) changing
    should re-render every scene that character appears in, not none at all."""
    from .stages import _load
    names = set(names)
    scene_numbers: set = set()
    storyboard = _load(out, "storyboard")
    if storyboard:
        for scene in storyboard.get("storyboard", []):
            for panel in (scene.get("panels") or scene.get("frames", [])):
                if names & set(panel.get("characters_in_frame", [])):
                    scene_numbers.add(scene.get("scene_number"))
                    break
        if scene_numbers:
            return scene_numbers
    scenes = _load(out, "scenes")
    if scenes:
        for scene in scenes.get("scenes", []):
            if names & set(scene.get("characters", [])):
                scene_numbers.add(scene.get("number"))
    return scene_numbers


def _apply_scene_render_revision(out, source_stage: str, revise_keys, panel_targets: dict) -> None:
    """Selectively re-render the video for a revision, preferring the
    cheaper, one-hop-cascade-limited `pipeline.rerender_panels` when the
    revision can be pinned to specific panels (a `storyboard` edit, via
    `panel_targets` — {scene_number: [panel_numbers]}, computed by the caller
    from `artifact_diff.diff_nested` BEFORE the edit was saved), and falling
    back to a full scene-level `_render_scene_frames(..., only_scenes=...)`
    otherwise — correct for an upstream stage (soundscape/visuals/
    cinematography/screenplay/scenes/casting) whose change legitimately
    affects every panel's prompt in the scene, not just one."""
    import json as _json
    from .stages import _load
    from .pipeline import rerender_panels, _render_scene_frames

    storyboard = _load(out, "storyboard")
    casting = _load(out, "casting")
    characters = _load(out, "characters")
    if storyboard is None or casting is None:
        print("[reel]   scene_render skipped — storyboard/casting artifact missing")
        return

    if revise_keys is None:
        print("[reel]   scene_render: full re-render (drastic upstream change)")
        _render_scene_frames(storyboard, casting, out, characters=characters)
        return

    scene_numbers = {k for k in revise_keys if isinstance(k, int)}
    if not scene_numbers:
        return   # e.g. a casting-only revision with no scene-number keys at all

    manifest_path = out / "video" / "manifest.json"

    def _load_manifest():
        return _json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    if source_stage == "storyboard" and panel_targets:
        for snum, panels in list(panel_targets.items()):
            if snum not in scene_numbers or not panels:
                continue
            existing_manifest = _load_manifest()
            has_manifest = existing_manifest and any(
                s.get("scene_number") == snum for s in existing_manifest.get("scenes", []))
            if has_manifest:
                print(f"[reel]   scene_render: panel-level re-render — scene {snum}, panels {panels}")
                rerender_panels(storyboard, casting, out, scene_number=snum,
                                panel_numbers=panels, characters=characters)
            else:
                print(f"[reel]   scene_render: scene {snum} never rendered — full scene render")
                _render_scene_frames(storyboard, casting, out, only_scenes={snum},
                                     existing_manifest=existing_manifest, characters=characters)
            scene_numbers.discard(snum)

    for snum in scene_numbers:
        print(f"[reel]   scene_render: full scene re-render — scene {snum}")
        _render_scene_frames(storyboard, casting, out, only_scenes={snum},
                             existing_manifest=_load_manifest(), characters=characters)


# Skipped by default during `revise` (both _revise_one's scoped downstream
# regen and _revise_source's full regen) — casting.json regeneration and
# every image/video rendering step are the only PAID-API-cost stages
# `revise` can trigger, so a text-only iteration session shouldn't pay that
# cost on every single round. Split into two independently toggleable
# categories — IMAGE_RENDER_STAGES (Gemini image spend: casting itself,
# since it's what casting_images renders from, plus casting_images/
# moodboard_tiles) and VIDEO_RENDER_STAGES (Veo spend: scene_render) — so an
# operator can include one without paying for the other (e.g. re-cast a
# character's look and see the portrait update without also re-rendering
# every video clip that references it, or vice versa). `--render-images`/
# `--render-video` (standalone `revise` command), or 'render images on/off'/
# 'render video on/off' inside the loop, toggle each independently;
# `--render`/'render on'/'render off' remain a shorthand for both at once.
# A skipped stage's existing checkpoint is left untouched (not deleted), so
# anything downstream that requires it (e.g. storyboard requires casting)
# still resolves — just against the last rendered/cast state, not a fresh one.
IMAGE_RENDER_STAGES = {"casting", "casting_images", "moodboard_tiles"}
VIDEO_RENDER_STAGES = {"scene_render"}
RENDER_SKIP_STAGES = IMAGE_RENDER_STAGES | VIDEO_RENDER_STAGES


def _render_stage_skipped(dname: str, render_images: bool, render_video: bool) -> bool:
    """Whether `dname` should be skipped this round, per its own category —
    the fine-grained replacement for a single `dname in RENDER_SKIP_STAGES
    and not render` check, now that image and video rendering can be
    enabled independently. A stage in neither category (every non-render
    stage) is never skipped by this check at all."""
    if dname in IMAGE_RENDER_STAGES:
        return not render_images
    if dname in VIDEO_RENDER_STAGES:
        return not render_video
    return False


def _render_status_message(render_images: bool, render_video: bool) -> str:
    """One-line summary of what this round's `render_images`/`render_video`
    flags will actually do, printed right before the confirm prompt in
    `_revise_source` — shared so the scoped and drastic-fallback paths
    can't describe the same flags differently."""
    if render_images and render_video:
        return "casting + all image/video rendering will be re-invoked"
    if render_images:
        return ("image rendering (casting/casting_images/moodboard_tiles) will be "
                "re-invoked; video rendering (scene_render) stays skipped — pass "
                "--render-video, or 'render video on' in the loop, to include it")
    if render_video:
        return ("video rendering (scene_render) will be re-invoked; image rendering "
                "(casting/casting_images/moodboard_tiles) stays skipped — pass "
                "--render-images, or 'render images on' in the loop, to include it")
    return ("casting/rendering stages will be SKIPPED by default — pass "
            "--render-images/--render-video (or --render for both), or "
            "'render images on'/'render video on' in the loop, to include them")


def _render_skip_suffix(render_images: bool, render_video: bool) -> str:
    """Trailing note for the "revision applied" summary line — blank when
    both categories are enabled (nothing was skipped), otherwise points back
    at the fuller `_render_status_message` printed earlier."""
    if render_images and render_video:
        return ""
    return " (image/video rendering as configured above — see the note before the confirm)"


def _gate_summary(name: str, result: dict) -> str | None:
    """Reuse the exact same per-stage summary renderers a fresh `pipeline.
    run()`'s live gate already uses, so a revise-round gate reads identically
    to a fresh-run gate for the same stage — one summarizer per stage, not a
    second copy. Returns `None` for a stage with no summarizer at all
    (casting_images/moodboard_tiles/scene_render/fidelity — render steps or
    the grader itself), which the caller treats as "don't gate this one"."""
    from . import pipeline as P
    table = {
        "structure": P._summarize_structure,
        "characters": P._summarize_characters,
        "scenes": P._summarize_scenes,
        "casting": P._summarize_casting,
        "soundscape": P._summarize_soundscape,
        "visuals": P._summarize_visuals,
        "moodboard": P._summarize_moodboard,
        "cinematography": P._summarize_cinematography,
        "screenplay": P._summarize_screenplay,
        "storyboard": P._summarize_storyboard,
    }
    fn = table.get(name)
    return fn(result) if fn else None


def _gate_stage_result(gate, name: str, result: dict, *, out, profile: str | None,
                       existing: dict | None, revise_keys, max_scenes: int | None,
                       duration_kwargs: dict, extra_kwargs: dict | None = None) -> dict:
    """Show the SAME per-stage review gate a fresh `pipeline.run()` uses —
    fidelity/genre score, feedback-driven re-run, view/edit-in-editor,
    auto-escalation, auto-approve timeout — for a stage `revise` just
    regenerated. Per direct request: `revise`'s downstream cascade should
    get the same gating experience a fresh pipeline run's stages already do,
    not just a plain auto-apply.

    `gate=None` (every pre-existing call site's implicit default, and what
    every non-gating test still passes) means "no gating at all" — the
    result is returned completely unchanged, identical to before this
    feature existed. Only `_revise_loop`'s real `Gate.from_config(...)`
    instance turns this on for an interactive session.

    A stage `_gate_summary` has no summarizer for (casting_images/
    moodboard_tiles/scene_render/fidelity — render steps or the grader
    itself) isn't gated either — nothing to meaningfully review.

    `rerun_fn` re-invokes `stages.run_stage` with the SAME `existing`/
    `revise_keys` this round already committed to (plus any operator
    feedback appended) — feedback at this gate refines the CURRENT scoped
    regeneration, it never widens or narrows what this round targets.
    `extra_kwargs` (e.g. `prior_scene_count` for a drastic "scenes" regen —
    see `scenes._revision_reminder_note`) is threaded into every rerun too,
    so a feedback-driven re-run doesn't silently lose a stage-specific
    reminder the initial call had.

    Fidelity/genre scoring reuses `pipeline.FIDELITY_GATED_STAGES`/
    `GENRE_GATED_STAGES` (the same stage sets a fresh run scores) and reloads
    `source.json`/`genre.json` fresh from `out` — cheap, and correct even if
    `scenes`/`genre` were just regenerated earlier in this same round.

    Raises `PipelineStopped` on 'stop', exactly like a fresh run's gate —
    callers catch it the same way `cli.main` already catches it for
    `pipeline.run()`."""
    if gate is None:
        return result
    summary = _gate_summary(name, result)
    if summary is None:
        return result
    from . import pipeline as P
    from .stages import _load, run_stage
    from .agents import fidelity as fidelity_agent
    from .agents import genre as genre_agent

    def summarize_fn(r):
        return _gate_summary(name, r) or ""

    def rerun_fn(fb, p=None):
        return run_stage(name, out=out, profile=p or profile, feedback=fb,
                         existing=existing, revise_keys=revise_keys,
                         max_scenes=max_scenes, **duration_kwargs,
                         **(extra_kwargs or {}))

    cfg = llm.config()
    fid_cfg = cfg.get("fidelity", {})
    gen_cfg = cfg.get("genre", {})
    rt_cfg = cfg.get("runtime", {})

    fidelity_fn = None
    if bool(fid_cfg.get("per_stage", True)) and name in P.FIDELITY_GATED_STAGES:
        source_text = (_load(out, "source") or {}).get("text", "")

        def fidelity_fn(r, _n=name, _txt=source_text):
            try:
                return fidelity_agent.check_stage(_n, r, _txt)
            except Exception:
                return None

    genre_fn = None
    genre_spec = _load(out, "genre") or {}
    if bool(gen_cfg.get("enforce", True)) and genre_spec and name in P.GENRE_GATED_STAGES:
        def genre_fn(r, _n=name, _spec=genre_spec):
            try:
                return genre_agent.enforce_stage(_n, r, _spec)
            except Exception:
                return None

    approved, _fid_rep, _gen_rep = P._gated(
        gate, name, result, summarize_fn, rerun_fn,
        fidelity_fn=fidelity_fn, min_score=int(fid_cfg.get("min_score", 70)),
        genre_fn=genre_fn, genre_min=int(gen_cfg.get("min_score", 70)),
        profile=profile, escalate_after=int(rt_cfg.get("escalate_after", 3)),
        escalate_score_gap=int(rt_cfg.get("escalate_score_gap", 20)))

    # `rerun_fn` (a feedback re-run) already persisted its own result via
    # `run_stage`'s own save=True default — but a 'view'-then-edit approval
    # never goes through `run_stage` at all (see `Gate.review`'s `edited`
    # path), so it's never actually written to disk unless we do it here
    # too — the same reason `pipeline.run_group` explicitly calls `save(nm,
    # r)` right after its own `_gated` call rather than trusting the loop
    # body to have already done it.
    from .stages import REGISTRY, _save_artifact
    _save_artifact(out, REGISTRY[name].artifact(), approved)
    if name == "screenplay":
        from .agents.screenplay import to_fountain
        source = _load(out, "source") or {}
        structure = _load(out, "structure") or {}
        (out / "screenplay.fountain").write_text(
            to_fountain(source, structure, approved), encoding="utf-8")
    return approved


def _run_downstream_revision(out, stage_name: str, downstream: list[str], revise_keys,
                             edited_artifact: dict, panel_targets: dict, *,
                             profile: str | None, max_scenes: int | None,
                             target_duration: int | None, render_images: bool,
                             render_video: bool, gate=None) -> None:
    """Selectively re-run every stage in `downstream` (already computed by
    the caller via `stages.downstream_of(stage_name)`), scoped to
    `revise_keys`. Shared by `_revise_one` (any directly hand-edited stage)
    and `_revise_source`'s scoped path (always `stage_name="scenes"`, since
    a source-text edit that survives the drastic check gets funneled
    through a scoped `scenes` re-run first, then cascades from there
    exactly like a direct `scenes.json` edit would) — one shared cascade
    implementation instead of two that could silently drift apart.

    Respects `RENDER_SKIP_STAGES` (see that constant), routes `scene_render`
    through `_apply_scene_render_revision` (translating a name-keyed source
    to the scene numbers those names actually appear in first, via
    `_names_to_scene_numbers`), and translates `revise_keys`'s key type per
    downstream stage via `_translate_revise_keys` — the same three pieces
    of logic `_revise_one`'s downstream loop always had, just no longer
    only reachable from there.

    Prints a one-line indication of what's actually changing for EACH
    downstream stage right before it runs (`regenerating [...]` for a
    scoped subset, `full regen` when no scoped translation applies,
    `nothing to regenerate` for an empty-but-known scope, or a skip notice
    for a render stage whose category — image or video — is disabled) —
    a companion to the
    "evaluation" printout `_revise_one`/`_revise_source` already print as
    soon as the affected scenes/names are identified, so the operator sees
    both WHAT was identified and WHAT each downstream stage is actually
    going to do about it, not just silence while `run_stage` executes.

    `max_scenes` is always `None` ("all") when called via the normal
    `revise` flow — `_revise_loop` never inherits the original run's
    `--max-scenes`, since a revision should never re-cap rendering to a
    prototype-scale value just because that's what the original full-
    pipeline run happened to use (see `_revise_loop`'s own docstring).

    Deliberately does NOT itself guarantee every scene-keyed stage stays
    aligned with scenes.json — this only touches `downstream` (whatever
    `stages.downstream_of(stage_name)` says is reachable from THIS edit).
    That guarantee is `_align_scene_keyed_stages`'s job, called once,
    unconditionally, by both `_revise_one` and `_revise_source` after this
    function returns — see its docstring for why an edit-scoped cascade
    alone isn't a strong enough guarantee.

    `gate` (default `None` = no gating, unchanged from before this param
    existed) is passed straight through to `_gate_stage_result` for each
    regenerated stage — a real `Gate` here shows the same fidelity/genre/
    feedback-loop review a fresh pipeline run's gate does, per direct
    request. Render stages (`casting_images`/`moodboard_tiles`/
    `scene_render`) aren't gated — `_gate_stage_result` has no summarizer
    for them anyway, but they're not even routed through it here since
    they're media-generation steps, not text output to review."""
    from .stages import REGISTRY, _load, run_stage

    for dname in downstream:
        if _render_stage_skipped(dname, render_images, render_video):
            adjective, flag = ("image", "images") if dname in IMAGE_RENDER_STAGES else ("video", "video")
            print(f"[reel]   {dname}: skipped ({adjective} rendering disabled by default — "
                 f"pass --render-{flag}, or type 'render {flag} on' in the revise "
                 "loop, to include it)")
            continue
        if dname == "scene_render":
            render_keys = revise_keys
            if revise_keys is not None and stage_name in _NAME_KEYED_STAGES:
                # name-keyed source (casting/characters) -> translate to the
                # scene numbers those names actually appear in, so a locked-
                # identity change re-renders the scenes it actually affects.
                render_keys = _names_to_scene_numbers(out, revise_keys) or None
            _apply_scene_render_revision(out, stage_name, render_keys, panel_targets)
            continue
        if dname in ("casting_images", "moodboard_tiles"):
            print(f"[reel]   {dname}: re-rendering (image provider)")
            run_stage(dname, out=out, profile=profile, max_scenes=max_scenes)
            continue
        d_existing = _load(out, REGISTRY[dname].artifact())
        d_keys = _translate_revise_keys(stage_name, dname, revise_keys, edited_artifact)
        if d_keys is not None and not d_keys:
            # Empty (not None) scope — e.g. a scenes.json revision that only
            # DELETED scene(s), or a cross-type translation that found no
            # locations to re-cast. Nothing to regenerate: `d_existing`
            # already reflects the current state (any deletions were already
            # stripped by `_strip_removed_scenes` before this loop runs), so
            # calling the LLM here would just burn tokens on output that
            # gets entirely discarded by `merge_by_key`.
            print(f"[reel]   {dname}: nothing to regenerate (already up to date)")
            continue
        # Indicate what's actually changing for this stage BEFORE running
        # it — `d_keys is None` covers both an upstream drastic edit and a
        # cross-type translation with no known mapping, both of which mean
        # "full regen" for this one downstream stage regardless of cause.
        if d_keys is None:
            print(f"[reel]   {dname}: full regen (no scoped subset applies)")
        else:
            print(f"[reel]   {dname}: regenerating {sorted(d_keys, key=str)}")
        d_max_scenes = _effective_max_scenes(dname, max_scenes)
        d_duration_kwargs = _duration_kwargs(dname, out, target_duration)
        result = run_stage(dname, out=out, profile=profile, max_scenes=d_max_scenes,
                          existing=d_existing, revise_keys=d_keys, **d_duration_kwargs)
        _gate_stage_result(gate, dname, result, out=out, profile=profile,
                          existing=d_existing, revise_keys=d_keys,
                          max_scenes=d_max_scenes, duration_kwargs=d_duration_kwargs)


def _align_scene_keyed_stages(out, *, profile: str | None, max_scenes: int | None,
                              target_duration: int | None, gate=None) -> None:
    """Unconditional guarantee, called once after EVERY revision round
    (`_revise_one` and `_revise_source` both call this after their own
    scoped/full regen completes): every scene-keyed artifact already on disk
    (soundscape/visuals/cinematography/screenplay/storyboard) is re-checked
    against the CURRENT scenes.json and healed if it's drifted — regardless
    of which stage the operator actually edited this round, and regardless
    of whether `stages.downstream_of` for that one edit happened to reach
    it. scenes.json is the single source of truth every one of these stages
    is declared to depend on, so there is no legitimate scenario where one
    of them should ever disagree with it — this makes that an invariant
    that's re-verified every round, not something that only gets fixed if
    the specific edit's own cascade happens to touch it.

    Reuses the exact same deterministic (no LLM) primitives a fresh
    `pipeline.run()` already self-heals with inside `run_group` —
    `fidelity.strip_orphan_scenes` (drops a stale scene entry scenes.json
    no longer has) then `fidelity.check_scene_alignment` (finds any scene
    scenes.json has that this stage's data doesn't) — reiterated via the
    same scoped `existing=`/`revise_keys=` mechanism a direct hand-edit
    already uses, not a second, parallel implementation. A stage with no
    on-disk artifact yet is left alone: it has nothing to be misaligned
    FROM, and will get full, unscoped coverage the first time it actually
    runs (see `draft_screenplay`/`plan_storyboard`'s own `scoped =
    revise_keys is not None and existing` check).

    `gate` (default `None` = no gating) is passed through to
    `_gate_stage_result` for each backfilled stage — see that function's
    docstring."""
    from .stages import REGISTRY, _load, _save_artifact, run_stage
    from .agents import fidelity

    current_scenes = _load(out, "scenes")
    if not current_scenes:
        return
    for dname in sorted(_SCENE_KEYED_STAGES - {"scenes"}):
        artifact_name = REGISTRY[dname].artifact()
        existing = _load(out, artifact_name)
        if not existing:
            continue
        stripped = fidelity.strip_orphan_scenes(dname, existing, current_scenes)
        if stripped is not existing:
            print(f"[reel]   {dname}: stripped orphan scene(s) no longer in scenes.json")
            _save_artifact(out, artifact_name, stripped)
            existing = stripped
        align = fidelity.check_scene_alignment(dname, existing, current_scenes)
        missing = {k for k in align["missing_scenes"] if isinstance(k, int)}
        if not missing:
            continue
        print(f"[reel]   {dname}: aligning to scenes.json — this stage was missing scene(s) "
             f"{sorted(missing)}, regenerating just those now")
        d_max_scenes = _effective_max_scenes(dname, max_scenes)
        d_duration_kwargs = _duration_kwargs(dname, out, target_duration)
        result = run_stage(dname, out=out, profile=profile, max_scenes=d_max_scenes,
                          existing=existing, revise_keys=missing, **d_duration_kwargs)
        _gate_stage_result(gate, dname, result, out=out, profile=profile,
                          existing=existing, revise_keys=missing,
                          max_scenes=d_max_scenes, duration_kwargs=d_duration_kwargs)


def _strip_removed_scenes(out, downstream: list[str], scenes_after: dict) -> None:
    """Propagate a scenes.json DELETION to every other scene-keyed artifact.

    `revision_merge.merge_by_key` (what every scoped agent call uses) can
    only ADD or REPLACE a key, never delete one — so a scene removed from
    scenes.json would otherwise leave a stale, orphaned entry behind in
    soundscape/visuals/cinematography/screenplay/storyboard forever. Reuses
    `fidelity.strip_orphan_scenes` — the exact same deterministic primitive
    `pipeline.run_group` already uses to self-heal scene-structure
    alignment during a fresh run — rather than a second implementation of
    the same "drop any entry whose scene_number scenes.json doesn't have"
    logic. Called BEFORE `_run_downstream_revision`, so that function's own
    `existing=` load already sees the cleaned-up artifact.

    `screenplay` additionally needs `screenplay.fountain` regenerated after
    a strip (deterministic — no LLM call — via the same `to_fountain` the
    normal `stages.run_stage("screenplay", ...)` path already calls after
    every save), since nothing else in this revision round will touch that
    file when the stage's own `revise_keys` scope is empty (see the
    "nothing to regenerate" skip above)."""
    from .stages import REGISTRY, _load, _save_artifact
    from .agents import fidelity
    from .agents.screenplay import to_fountain

    for dname in downstream:
        if dname not in _SCENE_KEYED_STAGES or dname == "scenes":
            continue
        artifact_name = REGISTRY[dname].artifact()
        current = _load(out, artifact_name)
        if current is None:
            continue
        stripped = fidelity.strip_orphan_scenes(dname, current, scenes_after)
        if stripped is current:
            continue   # no-op: this artifact had no orphaned scene entries
        if dname == "screenplay" and "drafted_count" in stripped:
            # Keep these two bookkeeping counts (used by to_fountain's own
            # "draft covers N of M scenes" notice) honest after a deletion —
            # cosmetic, but stale-by-one is an easy, needless confusion.
            stripped = dict(stripped)
            stripped["drafted_count"] = len(stripped.get("scenes", []))
            stripped["total_scenes"] = len(scenes_after.get("scenes", []))
        _save_artifact(out, artifact_name, stripped)
        print(f"[reel]   {dname}: removed deleted scene entry/entries")
        if dname == "screenplay":
            structure = _load(out, "structure") or {}
            source = _load(out, "source") or {}
            (out / "screenplay.fountain").write_text(
                to_fountain(source, structure, stripped), encoding="utf-8")


def _strip_removed_scenes_from_video_manifest(out, scenes_after: dict) -> None:
    """Deleting a scene leaves a stale entry in output/video/manifest.json —
    `pipeline._assemble_movie` stitches every scene the manifest lists, in
    order, so a leftover entry for a scene number scenes.json no longer has
    would still be included in movie.mp4. Strips it and re-assembles the
    movie so it reflects the current scene set. Does NOT delete the actual
    clip files on disk (non-destructive by design — an operator who deletes
    a scene can still find the old clips under output/video/scene_NN/ if
    they want them). Purely local bookkeeping (no Gemini/Veo call), so this
    runs unconditionally, independent of the `render` skip-by-default flag."""
    import json as _json
    from .pipeline import _assemble_movie

    manifest_path = out / "video" / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_numbers = {s.get("number") for s in scenes_after.get("scenes", [])}
    scenes_list = manifest.get("scenes", [])
    kept = [s for s in scenes_list if s.get("scene_number") in valid_numbers]
    if len(kept) == len(scenes_list):
        return
    removed_count = len(scenes_list) - len(kept)
    manifest["scenes"] = kept
    movie = _assemble_movie(manifest, out)
    manifest["movie"] = str(movie.relative_to(out)) if movie else None
    manifest_path.write_text(_json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reel]   scene_render: removed {removed_count} deleted scene(s) from the video "
         "manifest and re-assembled movie.mp4 (old clip files left on disk, not deleted)")


def _revise_source(out, *, edited_override: dict | None = None, auto_confirm: bool = False,
                   profile: str | None = None, max_scenes: int | None = 1,
                   target_duration: int | None = None, render_images: bool = False,
                   render_video: bool = False, gate=None) -> bool:
    """Revise the raw ingested story text — SCOPED to the scenes actually
    affected, when possible, instead of always falling back to a full regen.

    `artifact_diff.candidate_changed_scenes` (deterministic: which scenes'
    stored `source_excerpt` no longer appears verbatim in the new text) plus
    `reel.agents.revision.identify_source_text_changes` (LLM-confirmed, on
    `agent_profiles.revision` — the LARGEST local tier by default, since
    this needs to correlate a diff against every existing scene reliably)
    narrow the edit down to specific scene numbers. Once known, those
    numbers get funneled through the SAME machinery a direct `scenes.json`
    hand-edit already uses: `scenes` itself is re-run scoped via
    `existing=`/`revise_keys=`, then `_run_downstream_revision` cascades
    from there exactly like `_revise_one("scenes", ...)` would — one shared
    scoped-revision implementation, not a second one that could drift.

    Falls back to the original full-regen-of-every-stage behavior in two
    cases: no `scenes.json` exists yet to correlate the diff against (e.g.
    the very first revision on a fresh ingest), or the analysis itself
    reports the change as DRASTIC (implies a scene should be added or
    removed — this pipeline's v1 scope assumption, shared with every other
    scene-keyed artifact's diff, is that scene count/order stays stable
    across a scoped revision).

    `profile`/`target_duration` are the original run's inherited attributes
    (see `_revise_loop`'s docstring); `max_scenes` is deliberately NOT
    inherited from the original run — `_revise_loop` always passes `None`
    ("all") here, since a revision should never re-cap casting-image/video
    rendering to a prototype-scale `--max-scenes N` just because that's
    what the original run happened to use. All threaded into every
    `run_stage` call so a regen doesn't silently revert to `run_stage`'s
    own bare default of `max_scenes=1` either.

    `render_images`/`render_video` (each default `False`) independently gate
    `IMAGE_RENDER_STAGES`/`VIDEO_RENDER_STAGES` — see those constants'
    docstring."""
    from .stages import STAGES, _load, _save_artifact, run_stage, downstream_of
    from .agents.ingest import chunk_text
    from .gate import edit_in_editor
    from . import artifact_diff
    from .agents import revision as revision_agent

    current = _load(out, "source")
    if current is None:
        print("[reel] source.json doesn't exist yet — run `stage ingest SOURCE.txt` first")
        return False

    edited = edited_override if edited_override is not None else edit_in_editor("source", current)
    if edited is None:
        return False

    diff = artifact_diff.diff_source_text(current, edited)
    if not diff["changed"]:
        print("[reel] no changes detected — nothing to revise")
        return False

    new_text = edited.get("text", "")
    chunks = chunk_text(new_text)
    edited["chunks"] = chunks
    edited["chunk_count"] = len(chunks)
    edited["word_count"] = len(new_text.split())
    edited["char_count"] = len(new_text)

    current_scenes = _load(out, "scenes")
    analysis = None
    if current_scenes and current_scenes.get("scenes"):
        candidates = artifact_diff.candidate_changed_scenes(
            current.get("text", ""), new_text, current_scenes)
        unified_diff = artifact_diff.unified_source_diff(current.get("text", ""), new_text)
        rev_profile = profile or llm.agent_profile("revision")
        print(f"[reel] analyzing story-text change against "
             f"{len(current_scenes.get('scenes', []))} scene(s)  [{rev_profile}] …")
        analysis = revision_agent.identify_source_text_changes(
            unified_diff, current_scenes, candidates, profile=rev_profile)

    if analysis and not analysis.get("drastic") and analysis.get("changed_scene_numbers"):
        changed_scene_numbers = set(analysis["changed_scene_numbers"])
        print(f"[reel] evaluation — source text: scene(s) {sorted(changed_scene_numbers)} affected"
             + (f" — {analysis['summary']}" if analysis.get("summary") else ""))
        print(f"[reel]   {_render_status_message(render_images, render_video)}")
        if not (auto_confirm or input(
                f"[reel] proceed with scoped revision (scene(s) {sorted(changed_scene_numbers)} "
                "+ downstream)? [y/N] ").strip().lower() == "y"):
            print("[reel] revision cancelled")
            return False

        _save_artifact(out, "source", edited)
        scenes_duration_kwargs = _duration_kwargs("scenes", out, target_duration)
        new_scenes = run_stage("scenes", out=out, profile=profile,
                               existing=current_scenes, revise_keys=changed_scene_numbers,
                               **scenes_duration_kwargs)
        new_scenes = _gate_stage_result(gate, "scenes", new_scenes, out=out, profile=profile,
                                        existing=current_scenes, revise_keys=changed_scene_numbers,
                                        max_scenes=None, duration_kwargs=scenes_duration_kwargs)
        downstream = downstream_of("scenes")
        _run_downstream_revision(out, "scenes", downstream, changed_scene_numbers, new_scenes, {},
                                 profile=profile, max_scenes=max_scenes,
                                 target_duration=target_duration, render_images=render_images,
                                 render_video=render_video, gate=gate)
        _align_scene_keyed_stages(out, profile=profile, max_scenes=max_scenes,
                                  target_duration=target_duration, gate=gate)
        print(f"[reel] source revision applied — scoped to scene(s) "
             f"{sorted(changed_scene_numbers)}"
             + _render_skip_suffix(render_images, render_video))
        return True

    # Drastic (or no scenes.json to scope against yet) — full regen fallback.
    if analysis and analysis.get("drastic"):
        print(f"[reel] ⚠ {analysis.get('reason') or 'story-text change looks drastic'}")
    elif current_scenes is None or not current_scenes.get("scenes"):
        print("[reel] ⚠ no scenes.json yet to scope this change against")
    else:
        print(f"[reel] ⚠ {diff['reason']}")
    print("[reel]   falling back to a full regen of every stage (this may take a while)")
    print(f"[reel]   {_render_status_message(render_images, render_video)}")
    if not (auto_confirm or input("[reel] proceed? [y/N] ").strip().lower() == "y"):
        print("[reel] revision cancelled")
        return False

    prior_scene_count = len(current_scenes.get("scenes", [])) if current_scenes else None
    _save_artifact(out, "source", edited)
    for s in STAGES:
        if s.name == "ingest":
            continue
        if _render_stage_skipped(s.name, render_images, render_video):
            adjective, flag = ("image", "images") if s.name in IMAGE_RENDER_STAGES else ("video", "video")
            print(f"[reel]   {s.name}: skipped ({adjective} rendering disabled by default — "
                 f"pass --render-{flag}, or type 'render {flag} on' in the revise "
                 "loop, to include it)")
            continue
        extra = {"prior_scene_count": prior_scene_count} if s.name == "scenes" else {}
        s_max_scenes = _effective_max_scenes(s.name, max_scenes)
        s_duration_kwargs = _duration_kwargs(s.name, out, target_duration)
        result = run_stage(s.name, out=out, profile=profile, max_scenes=s_max_scenes,
                          **s_duration_kwargs, **extra)
        _gate_stage_result(gate, s.name, result, out=out, profile=profile,
                          existing=None, revise_keys=None,
                          max_scenes=s_max_scenes, duration_kwargs=s_duration_kwargs,
                          extra_kwargs=extra)
    _align_scene_keyed_stages(out, profile=profile, max_scenes=max_scenes,
                              target_duration=target_duration, gate=gate)
    print("[reel] source revision applied — every downstream stage regenerated"
         + _render_skip_suffix(render_images, render_video))
    return True


def _revise_one(stage_name: str, out, *, edited_override: dict | None = None,
                auto_confirm: bool = False, profile: str | None = None,
                max_scenes: int | None = 1, target_duration: int | None = None,
                render_images: bool = False, render_video: bool = False,
                gate=None) -> bool:
    """One revision round: edit `stage_name`'s current artifact via $EDITOR
    (or the raw source text if `stage_name == "source"`), figure out what
    actually changed, propose a downstream re-run plan (falling back to a
    full regen for a "drastic" change — see `artifact_diff`), confirm, and
    selectively re-run what's affected. Returns True if a revision was
    actually applied, False if cancelled or no change was made.

    Handles all three edit shapes directly, not just in-place modification:
    MODIFY (a changed value for an existing key) and ADD (a genuinely new
    scene number / character-or-location name) both flow through the normal
    `revise_keys` scoped-regen path (`revision_merge.merge_by_key` appends a
    new key on its own). DELETE is different — `merge_by_key` can only add
    or replace, never remove — so for `stage_name == "scenes"` specifically
    (the one artifact where deletion is common and well-scoped: scenes.json
    is the single source of truth every other scene-keyed artifact must
    mirror), a removed scene number is propagated by directly stripping it
    out of every downstream artifact (`_strip_removed_scenes`) and out of
    the video manifest (`_strip_removed_scenes_from_video_manifest`) right
    after the edit is saved, before the normal downstream cascade runs for
    whatever was also added/changed in the same edit.

    Deleting a NAME (a character/location from casting.json or
    characters.json) is also allowed — `artifact_diff.ARTIFACT_SHAPES`
    already sets `allow_remove=True` for both — but has no equivalent
    downstream-stripping step: those artifacts have no per-scene structure
    to reconcile against, so the deletion is simply saved as-is; anything
    downstream that still references the deleted name only stops seeing it
    in FUTURE regenerations, not retroactively cleaned up.

    `render_images`/`render_video` (each default `False`) independently gate
    `IMAGE_RENDER_STAGES`/`VIDEO_RENDER_STAGES` in the downstream re-run loop
    below — see those constants' docstring for why they're skipped by default.

    `edited_override`/`auto_confirm` are testing hooks — bypass the
    interactive $EDITOR / confirm prompt with a canned value, so this
    function (and thus the whole revise flow) can be driven directly without
    a TTY.

    `profile`/`max_scenes`/`target_duration` are the original run's
    inherited attributes (see `_revise_loop`'s docstring), threaded into
    every downstream `run_stage` call below via `_duration_kwargs` for the
    two stages that consume target_duration."""
    if stage_name in ("source", "ingest"):
        # "ingest" is the stage that PRODUCES source.json (Stage.produces=
        # "source") — routed to the same special handling as "source" itself
        # (chunk_indices/word_count/char_count recomputation) rather than the
        # generic path below, which has no idea those derived fields exist.
        return _revise_source(out, edited_override=edited_override, auto_confirm=auto_confirm,
                              profile=profile, max_scenes=max_scenes, target_duration=target_duration,
                              render_images=render_images, render_video=render_video, gate=gate)

    from .stages import REGISTRY, _load, _save_artifact, downstream_of
    from .gate import edit_in_editor
    from . import artifact_diff
    from .agents import revision as revision_agent

    stage = REGISTRY[stage_name]
    artifact_name = stage.artifact()
    current = _load(out, artifact_name)
    if current is None:
        print(f"[reel] {artifact_name}.json doesn't exist yet — run "
             f"`python -m reel.cli stage {stage_name}` first")
        return False

    edited = edited_override if edited_override is not None else edit_in_editor(stage_name, current)
    if edited is None:
        return False

    panel_targets: dict = {}   # {scene_number: [panel_numbers]} — storyboard only
    removed_keys: set = set()   # scene numbers / names DELETED from this artifact
    whole_file = stage_name in artifact_diff.WHOLE_FILE_ARTIFACTS
    if whole_file:
        drastic, reason, revise_keys = True, f"'{stage_name}' has no scene/name-keyed structure to scope by", None
        # Evaluation printed immediately — before any ripple-suggestion/
        # identity-check LLM calls below, and well before the final "plan:"
        # confirm prompt — so the operator sees WHAT was identified as soon
        # as it's known, not buried after other side effects.
        print(f"[reel] evaluation — {artifact_name}.json: whole-file artifact, no "
             "scene/name-keyed structure to scope by — always a full downstream regen")
    else:
        diff = artifact_diff.diff_artifact(stage_name, current, edited)
        drastic, reason = diff.drastic, diff.drastic_reason
        revise_keys = set(diff.changed) | set(diff.added)
        removed_keys = set(diff.removed)
        print(f"[reel] evaluation — {artifact_name}.json: "
             f"changed={sorted(diff.changed, key=str)}  "
             f"added={sorted(diff.added, key=str)}  "
             f"removed={sorted(diff.removed, key=str)}")
        if stage_name == "storyboard" and not drastic and revise_keys:
            nested = artifact_diff.diff_nested("storyboard", current, edited)
            for snum in revise_keys:
                nd = nested.get(snum)
                if nd and not nd.drastic:
                    changed_panels = sorted(set(nd.changed) | set(nd.added))
                    if changed_panels:
                        panel_targets[snum] = changed_panels

    if drastic:
        print(f"[reel] ⚠ drastic change: {reason}")
        print("[reel]   falling back to a full downstream regen (no scoped revision)")
        if not (auto_confirm or input("[reel] proceed? [y/N] ").strip().lower() == "y"):
            print("[reel] revision cancelled")
            return False
        revise_keys = None
    elif not revise_keys and not removed_keys:
        print("[reel] no changes detected — nothing to revise")
        return False
    else:
        if removed_keys:
            print(f"[reel] {sorted(removed_keys, key=str)} deleted from {artifact_name}.json")
            if stage_name == "scenes":
                print("[reel]   will be removed from every downstream artifact that "
                     "tracks scenes, and from the video manifest if rendered")

        if stage_name == "casting":
            by_name_old = {c.get("name"): c for c in current.get("casting", [])}
            by_name_new = {c.get("name"): c for c in edited.get("casting", [])}
            for name in sorted(revise_keys, key=str):
                old_c, new_c = by_name_old.get(name), by_name_new.get(name)
                if not old_c or not new_c:
                    continue   # genuinely new name — nothing to compare against
                old_desc = (old_c.get("character") or old_c).get("visual_prompt", "")
                new_desc = (new_c.get("character") or new_c).get("visual_prompt", "")
                is_drastic, ratio = revision_agent.is_drastic_identity_change(old_desc, new_desc)
                if is_drastic:
                    print(f"[reel] ⚠ '{name}': description changed a lot (similarity {ratio:.2f}) — "
                         "the OLD reference image will be preserved by default; re-render its "
                         "casting image manually if you actually want the new look reflected")

        if stage_name in _SCENE_KEYED_STAGES:
            changed_scene_numbers = sorted(k for k in revise_keys if isinstance(k, int))
            if changed_scene_numbers:
                source = _load(out, "source") or {}
                scenes_artifact = edited if stage_name == "scenes" else (_load(out, "scenes") or {})
                summary = f"stage '{stage_name}' scene(s) {changed_scene_numbers} were edited"
                ripple = revision_agent.suggest_ripple_scenes(
                    source.get("text", ""), scenes_artifact, changed_scene_numbers, summary)
                suggested = ripple.get("suggested_scenes", [])
                if suggested:
                    print("[reel] possible ripple effects (not applied automatically):")
                    for s in suggested:
                        print(f"[reel]   scene {s.get('scene_number')}: {s.get('reason', '')}")
                    if not auto_confirm:
                        extra = input("[reel] add any of these scene numbers to the revision? "
                                      "(comma-separated, or blank to skip): ").strip()
                        if extra:
                            try:
                                revise_keys |= {int(x.strip()) for x in extra.split(",") if x.strip()}
                            except ValueError:
                                print("[reel] couldn't parse that — skipping ripple additions")

    downstream = downstream_of(stage_name)
    plan_bits = []
    if revise_keys:
        plan_bits.append(f"revising: {sorted(revise_keys, key=str)}")
    if removed_keys:
        plan_bits.append(f"removing: {sorted(removed_keys, key=str)}")
    plan_desc = f" ({'; '.join(plan_bits)})" if plan_bits else " (full regen)"
    print(f"[reel] plan: save {artifact_name}.json{plan_desc}"
         + (f", then re-run: {', '.join(downstream)}" if downstream else ", nothing downstream"))
    if not (auto_confirm or input("[reel] proceed? [y/N] ").strip().lower() == "y"):
        print("[reel] revision cancelled")
        return False

    _save_artifact(out, artifact_name, edited)
    if removed_keys and stage_name == "scenes":
        _strip_removed_scenes(out, downstream, edited)
        _strip_removed_scenes_from_video_manifest(out, edited)
    _run_downstream_revision(out, stage_name, downstream, revise_keys, edited, panel_targets,
                             profile=profile, max_scenes=max_scenes,
                             target_duration=target_duration, render_images=render_images,
                             render_video=render_video, gate=gate)
    _align_scene_keyed_stages(out, profile=profile, max_scenes=max_scenes,
                              target_duration=target_duration, gate=gate)

    print(f"[reel] revision applied — {artifact_name}.json"
         + (f" and {len(downstream)} downstream stage(s)" if downstream else "") + " updated")
    return True


def _revise_loop(out, *, render_images: bool = False, render_video: bool = False) -> None:
    """The interactive stage-picker/edit/selective-rerun loop: pick any stage
    (or 'source' for the raw ingested text) to hand-edit in $EDITOR, review
    the proposed downstream re-run plan, and selectively apply it — repeat as
    many rounds as needed. Shared by the standalone `revise` command and the
    "revise now?" prompt offered right after a full pipeline run completes
    (see `_offer_revise`) — both just need to have already called
    `session.start(out, fresh=False)` before entering this loop, so it stays
    'running' across every round; it's only marked 'complete'/'paused' here
    when you type 'quit'/'exit' or 'pause' (or Ctrl-C).

    `render_images`/`render_video` (each default `False`) independently gate
    `IMAGE_RENDER_STAGES`/`VIDEO_RENDER_STAGES` for the whole session —
    skipped by default so text-only iteration doesn't pay their API cost
    every round. Each can be toggled separately mid-session, without
    restarting: 'render images on'/'render images off' for image rendering
    (casting/casting_images/moodboard_tiles — Gemini spend) and 'render
    video on'/'render video off' for video rendering (scene_render — Veo
    spend); 'render on'/'render off' remain a shorthand that toggles both at
    once. The current setting of both is echoed in the menu header.

    Restores the ORIGINAL run's attributes before any regeneration happens:
    `_restore_direction` re-applies the genre/moodboard creative-direction
    steering (a separate process invocation otherwise starts with none at
    all), and `_load_run_params` recovers the `--profile`/`--target-duration`
    this `--out` was actually run with, threaded into every `_revise_one`
    call below — without this, a revision silently reverted to each stage's
    own bare defaults (e.g. dropping a `--profile fast` override, or losing
    a `--target-duration` scene-count budget) instead of continuing to
    honor what the operator originally chose.

    `max_scenes` is the ONE inherited attribute deliberately NOT restored
    from `run_params.json` — `revise` always operates as if `--max-scenes
    all` (`None`) had been used, regardless of what the original run's
    `--max-scenes` was. A revision's whole point is to selectively update
    exactly the scenes a diff identifies as affected (see `_revise_one`'s
    evaluation printout); silently re-capping casting-image/video
    rendering to the original run's prototype-scale `--max-scenes N` would
    make an identified-as-affected scene beyond that cap invisible to the
    render stages for no reason a revision should ever need. Scoping still
    happens — via `revise_keys`, not `max_scenes` — so this doesn't widen
    what actually gets regenerated, only removes an unrelated, accidental
    cap on what CAN be.

    Builds a real `Gate.from_config(...)` (the same class a fresh
    `pipeline.run()` uses) and threads it into every `_revise_one` call, so
    each stage `revise` regenerates this round gets the identical review
    experience a fresh run's stage does — fidelity/genre score, feedback-
    driven re-run, view/edit-in-editor, auto-escalation, auto-approve
    timeout (see `_gate_stage_result`) — not just an auto-apply. A 'stop' at
    any of those gates raises `PipelineStopped`, caught here the same way
    Ctrl-C already is: the session is marked 'paused' and the loop exits."""
    from . import session
    from .gate import Gate
    from .stages import STAGES, names, _load

    _restore_direction(out)
    run_params = _load_run_params(out)
    profile = run_params.get("profile")
    max_scenes = None                      # always "all" — see docstring above
    target_duration = run_params.get("target_duration")
    gate = Gate.from_config(llm.config())

    try:
        while True:
            images_note = "ON" if render_images else "off"
            video_note = "ON" if render_video else "off"
            print(f"\n[reel] revise — {out}/  (session: {session.current(out)})  "
                 f"[render: images={images_note}, video={video_note}]")
            src_mark = "✓" if (out / "source.json").exists() else " "
            print(f"  [{src_mark}] source")
            for s in STAGES:
                if s.name == "ingest":
                    continue   # same artifact as "source" above (produces="source") — not a separate entry
                mark = "✓" if _load(out, s.artifact()) is not None else " "
                print(f"  [{mark}] {s.name}")
            choice = input("\npick a stage to edit ('quit'/'exit' to finish, 'pause' to stop "
                           "for now, 'render images on/off', 'render video on/off', or "
                           "'render on/off' for both, to toggle rendering): ").strip().lower()
            if not choice:
                continue
            if choice in ("quit", "q", "exit", "e"):
                session.finish(out, "complete")
                print(f"[reel] revision session complete → {out}/session.json")
                return
            if choice in ("pause", "p"):
                session.finish(out, "paused")
                print(f"[reel] revision session paused → {out}/session.json")
                return
            if choice in ("render images on", "render images:on", "render image on"):
                render_images = True
                print("[reel] image rendering enabled — casting/casting_images/moodboard_tiles "
                     "will run when affected")
                continue
            if choice in ("render images off", "render images:off", "render image off"):
                render_images = False
                print("[reel] image rendering disabled — casting/casting_images/moodboard_tiles "
                     "will be skipped")
                continue
            if choice in ("render video on", "render video:on"):
                render_video = True
                print("[reel] video rendering enabled — scene_render will run when affected")
                continue
            if choice in ("render video off", "render video:off"):
                render_video = False
                print("[reel] video rendering disabled — scene_render will be skipped")
                continue
            if choice in ("render on", "render:on", "render enable"):
                render_images = render_video = True
                print("[reel] image + video rendering enabled — both will run when affected")
                continue
            if choice in ("render off", "render:off", "render disable"):
                render_images = render_video = False
                print("[reel] image + video rendering disabled — both will be skipped")
                continue
            if choice != "source" and choice not in names():
                print(f"[reel] unknown stage {choice!r}")
                continue
            _revise_one(choice, out, profile=profile, max_scenes=max_scenes,
                       target_duration=target_duration, render_images=render_images,
                       render_video=render_video, gate=gate)
    except KeyboardInterrupt:
        session.finish(out, "paused")
        print("\n[reel] revision loop paused.")
    except PipelineStopped as e:
        session.finish(out, "paused")
        print(f"\n[reel] {e} — revision loop paused "
             "(completed stages this round stay saved).")


def _print_spend_summary(out) -> None:
    """Print an estimated $ spend readout from `<out>/logs/gemini_api.log`
    (see `spend.py`'s module docstring for the pricing-snapshot caveat).
    Best-effort — never raises, since this is purely informational and must
    never be the reason a run's actual completion/error path fails."""
    from . import spend as _spend
    try:
        result = _spend.estimate_run_cost(out)
        if result["priced_calls"] or result["unpriced_calls"]:
            print(f"[reel] estimated spend this run: {_spend.format_summary(result)}")
    except Exception:
        pass


def _offer_revise(out) -> None:
    """Right after a full pipeline run completes (video render included, if
    enabled), offer to jump straight into the revision loop without a
    separate `python -m reel.cli revise` invocation — or exit cleanly. Purely
    additive to that standalone command, which still works the same as
    always for revisiting a run later. Skipped entirely when `hitl.enabled`
    is false (unattended/batch runs) or there's no TTY (input() raises
    EOFError — the same "non-interactive means don't block" convention
    gate.py already uses), so a cron/CI run is never left waiting on a
    prompt. `pipeline.run()` already marked the session 'complete' on its own
    return by this point; choosing to revise here reopens it
    (`session.start(fresh=False)` reattaches and flips status back to
    'running') — declining leaves it exactly as `pipeline.run()` left it."""
    from . import llm as _llm
    if not bool(_llm.config().get("hitl", {}).get("enabled", True)):
        return
    try:
        choice = input("\n[reel] pipeline complete. Type 'revise' to make changes now, "
                       "or press Enter to exit: ").strip().lower()
    except EOFError:
        return
    if choice in ("revise", "r"):
        from . import session
        session.start(out, fresh=False)
        _revise_loop(out)


def _revise(argv: list[str]) -> int:
    """Standalone entry point: `python -m reel.cli revise --out DIR`.
    Reattaches to that --out's session and enters the same interactive
    revision loop `_offer_revise` can also drop straight into right after a
    fresh run finishes."""
    from pathlib import Path
    from . import session

    ap = argparse.ArgumentParser(prog="reel revise",
                                 description="revise a completed run: edit any stage, "
                                             "selectively re-run what's affected")
    ap.add_argument("--out", default="output")
    ap.add_argument("--render", action="store_true",
                    help="also re-run BOTH casting/image rendering and video rendering when "
                         "affected by an edit (both skipped by default — text-only iteration "
                         "is free; shorthand for --render-images --render-video)")
    ap.add_argument("--render-images", action="store_true",
                    help="re-run casting/casting_images/moodboard_tiles (Gemini image spend) "
                         "when affected by an edit — independent of --render-video; toggle "
                         "anytime inside the loop with 'render images on'/'render images off'")
    ap.add_argument("--render-video", action="store_true",
                    help="re-run scene_render (Veo video spend) when affected by an edit — "
                         "independent of --render-images; toggle anytime inside the loop with "
                         "'render video on'/'render video off'")
    a = ap.parse_args(argv)
    out = Path(a.out)

    if not out.exists():
        print(f"[reel] {out}/ doesn't exist — run the pipeline first")
        return 2

    session.start(out, fresh=False)
    _revise_loop(out, render_images=a.render or a.render_images,
                render_video=a.render or a.render_video)
    return 0


def _print_config_warnings() -> None:
    """Best-effort startup sanity check for config/models.yaml — see
    `llm.validate_config`'s own docstring for what it catches (unknown
    keys, a wrong type on a handful of consequential fields) and why it's
    deliberately shallow rather than a full schema. Never blocks — a
    warning here is a hint to check for a typo, not an error; every
    individual `.get(key, default)` call downstream already degrades
    gracefully on its own regardless of what this prints."""
    try:
        warnings = llm.validate_config()
    except Exception:
        return  # config() itself failing surfaces its own clear error later
    if warnings:
        print("[reel] config/models.yaml — possible issues:")
        for w in warnings:
            print(f"  ⚠ {w}")


def _cmd_spend(argv: list[str]) -> int:
    """`python -m reel.cli spend [--out DIR]` — print an estimated $ spend
    breakdown from `<out>/logs/gemini_api.log` at any time, not just right
    after a run — the log accumulates across every `--resume`/`revise`
    round against the same `--out`, so this reflects the run's TOTAL spend
    to date, not just the most recent invocation. See `spend.py`'s module
    docstring for the pricing-snapshot caveat (an estimate, not a
    reconciled bill)."""
    from pathlib import Path
    from . import spend as _spend

    ap = argparse.ArgumentParser(prog="reel spend",
                                 description="estimate Gemini/Veo API spend from gemini_api.log")
    ap.add_argument("--out", default="output")
    a = ap.parse_args(argv)
    out = Path(a.out)
    result = _spend.estimate_run_cost(out)
    print(f"[reel] {_spend.format_summary(result)}")
    if result["by_model"]:
        print("[reel] by model:")
        for model, bucket in sorted(result["by_model"].items(), key=lambda kv: -kv[1]["usd"]):
            print(f"    {model}: ~${bucket['usd']:.4f} ({bucket['calls']} call(s))")
    return 0


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    _print_config_warnings()
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "stages":
        return _list_stages()
    if argv and argv[0] == "spend":
        return _cmd_spend(argv[1:])
    if argv and argv[0] == "stage":
        return _run_stage(argv[1:])
    if argv and argv[0] == "render":
        return _render_video(argv[1:])
    if argv and argv[0] == "stitch":
        return _stitch(argv[1:])
    if argv and argv[0] == "gen-video":
        return _gen_video_prompt(argv[1:])
    if argv and argv[0] == "revise":
        return _revise(argv[1:])
    if argv and argv[0] == "veo-sync":
        return _veo_sync(argv[1:])

    ap = argparse.ArgumentParser(prog="reel", description=__doc__)
    ap.add_argument("source", nargs="?", help="path to source text (book/story/script)")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--max-scenes", type=_max_scenes_arg, default=_MAX_SCENES_UNSET,
                    help="how many scenes to RENDER — casting images and video "
                         "(default: 1, prototype), or 'all' for every scene; "
                         "every shot within each rendered scene is always rendered. "
                         "Design/planning stages (screenplay, storyboard, soundscape, "
                         "visuals, cinematography) always process every scene, "
                         "regardless of this flag. Omitted on a --resume run: "
                         "inherits whatever the run being resumed actually used, "
                         "not silently reset to the default")
    ap.add_argument("--profile", choices=["fast", "quality"], default=None,
                    help="force a single quality tier for every agent. Omitted on a "
                         "--resume run: inherits whatever the run being resumed "
                         "actually used")
    ap.add_argument("--genre", default=None,
                    help="force the adaptation's genre (e.g. 'noir thriller'); "
                         "overrides config genre.value. Omit to use config / auto-detect")
    ap.add_argument("--target-duration", type=int, default=None, dest="target_duration",
                    help="target total runtime of the rendered movie, in seconds "
                         "(default: config duration.target_seconds, currently 45) — "
                         "engine-independent guidance for scene/shot-count planning, "
                         "and the basis for each clip's requested render duration. "
                         "Omitted on a --resume run: inherits whatever the run being "
                         "resumed actually used")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed stages in --out and continue from the "
                         "first unfinished one (pair with a prior paused run)")
    ap.add_argument("--no-render", action="store_true", default=None, dest="no_render",
                    help="skip casting-image and video rendering for this run — "
                         "every design/planning stage (structure, characters, scenes, "
                         "soundscape, visuals, cinematography, screenplay, storyboard) "
                         "still runs normally, only the two stages that spend real "
                         "Gemini/Veo API quota are skipped. Equivalent to setting "
                         "config `image.enabled`/`video.enabled` to false, but scoped "
                         "to just this invocation. Omitted on a --resume run: inherits "
                         "whatever the run being resumed actually used")
    ap.add_argument("--gate-timeout", type=int, default=None, dest="gate_timeout",
                    help="override the review-gate auto-approve idle timeout, in "
                         "seconds, for this invocation only (default: config "
                         "hitl.timeout_seconds, currently 900; 0 = wait forever). "
                         "Does not change hitl.enabled or persist to config/models.yaml")
    ap.add_argument("--list-models", action="store_true",
                    help="show local model / profile status and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        return _list_models()
    if not args.source:
        ap.error("a SOURCE file is required (or use --list-models)")

    # Resolve --max-scenes/--profile inheritance: an explicit flag always
    # wins; an OMITTED flag on a --resume run inherits whatever the run being
    # resumed actually used (persisted below to output/run_params.json) —
    # otherwise resuming with the bare hint this command itself prints would
    # silently fall back to argparse's defaults (1 scene, no profile
    # override) instead of continuing with the original run's settings.
    prev_params = _load_run_params(args.out) if args.resume else {}
    if args.max_scenes is _MAX_SCENES_UNSET:
        if args.resume and "max_scenes" in prev_params:
            max_scenes = prev_params["max_scenes"]
            print(f"[reel] --max-scenes not given — inheriting "
                 f"{_scenes_label(max_scenes)} from the run being resumed")
        else:
            max_scenes = 1
    else:
        max_scenes = args.max_scenes
    profile = args.profile
    if profile is None and args.resume and prev_params.get("profile"):
        profile = prev_params["profile"]
        print(f"[reel] --profile not given — inheriting {profile!r} from the run being resumed")
    target_duration = args.target_duration
    if target_duration is None and args.resume and prev_params.get("target_duration"):
        target_duration = prev_params["target_duration"]
        print(f"[reel] --target-duration not given — inheriting {target_duration}s "
             "from the run being resumed")
    # `args.no_render` is None when the flag wasn't given at all (default=None,
    # distinct from the flag's own True when it WAS given) — same sentinel
    # pattern as --max-scenes above, needed here because a bare boolean
    # default of False couldn't tell "not given" apart from "given as off".
    if args.no_render is None:
        if args.resume and "render" in prev_params:
            render = prev_params["render"]
            print(f"[reel] --no-render not given — inheriting "
                 f"render={'on' if render else 'off'} from the run being resumed")
        else:
            render = True
    else:
        render = not args.no_render

    _save_run_params(args.out, max_scenes=max_scenes, profile=profile, genre=args.genre,
                     target_duration=target_duration, render=render)

    try:
        run(args.source, out_dir=args.out, max_scenes=max_scenes,
            profile_override=profile, resume=args.resume, genre=args.genre,
            target_duration_seconds=target_duration, render=render,
            gate_timeout_seconds=args.gate_timeout)
    except PipelineStopped as e:
        session.finish(args.out, "paused")
        print(f"\n[reel] paused at '{e.stage}'. Completed stages saved in {args.out}/.")
        resume_cmd = (f"python -m reel.cli {args.source} --out {args.out} --resume "
                     f"--max-scenes {_scenes_label(max_scenes)}")
        if profile:
            resume_cmd += f" --profile {profile}"
        if target_duration:
            resume_cmd += f" --target-duration {target_duration}"
        if not render:
            resume_cmd += " --no-render"
        print(f"[reel] resume:  {resume_cmd}")
        _print_spend_summary(args.out)
        return 0
    except KeyboardInterrupt:
        session.finish(args.out, "paused")
        print("\n[reel] interrupted. Completed stages are saved; "
              "resume with --resume.")
        _print_spend_summary(args.out)
        return 130
    except Exception:
        session.finish(args.out, "failed")
        _print_spend_summary(args.out)
        raise
    _print_spend_summary(args.out)
    _offer_revise(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
