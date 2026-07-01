"""Command-line entry point for the reel screenplay-material pipeline.

Usage:
    python -m reel.cli SOURCE.txt [--out DIR] [--max-scenes N] [--profile NAME] [--resume]
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
    python -m reel.cli veo-sync                  # refresh Veo prompt guide snapshot
        [--status]                               #   print cache status only (no fetch)

Run via the package so relative imports resolve: `python -m reel.cli ...`.

Each stage can be invoked on its own (`stage NAME`): it loads the inputs it needs
from prior `output/<input>.json` checkpoints (ingesting SOURCE on demand) and
writes its own artifact — re-run a single stage without the whole pipeline. (Stage
runs are direct, with no HITL gate; use `--feedback` to pass a revision note.)

The pipeline pauses for human review after each stage (approve with Enter, type
feedback to re-run that stage, or 'stop' to pause). Toggle this in
`config/models.yaml` under `hitl` (set `enabled: false` for unattended runs;
tune `timeout_seconds` for the auto-approve fallback).

Each approved stage is checkpointed to `output/<stage>.json`. After a pause
(typing 'stop', Ctrl-C) or a failure, re-run with `--resume` to reload the
finished stages and continue from the first one that isn't done.
"""
from __future__ import annotations

import argparse
import sys

from . import llm
from .pipeline import run, PipelineStopped


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
    ap.add_argument("--max-scenes", type=int, default=1)
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
    artificial caps. Works purely off existing artifacts; no LLM stage runs."""
    import json
    import shutil
    from pathlib import Path

    from . import fountain, i2v
    from .pipeline import _render_scene_frames

    ap = argparse.ArgumentParser(prog="reel render",
                                 description="render scene videos from screenplay.fountain + cinematography.json")
    ap.add_argument("--out", default="output")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="optional cap on scenes (default: all drafted scenes)")
    ap.add_argument("--max-shots", type=int, default=None,
                    help="optional cap on shots per scene (default: every action beat)")
    ap.add_argument("--fresh", action="store_true",
                    help="re-render existing clips (clears output/video first; "
                         "old clips are backed up to output/video_prev)")
    a = ap.parse_args(argv)
    out = Path(a.out)

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
    )
    (out / "storyboard.json").write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    nshots = sum(len(s["frames"]) for s in board["storyboard"])
    print(f"[reel] render plan: {len(board['storyboard'])} scene(s), {nshots} shot(s) "
          f"(story-defined, camera-directed from cinematography.json) → {out}/storyboard.json")

    if a.fresh and (out / "video").exists():
        backup = out / "video_prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(out / "video"), str(backup))
        print(f"[reel] cleared existing clips → backed up to {backup}/")

    manifest = _render_scene_frames(board, _load_json(out / "casting.json"), out, max_scenes=a.max_scenes)
    print(f"[reel] rendered {manifest.get('clips', 0)} new clip(s) → {out}/video/")
    if manifest.get("movie"):
        print(f"[reel] movie → {out}/{manifest['movie']}")
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

    # Pull config defaults, allow per-call overrides.
    cfg = i2v._cfg()
    model       = a.model        or cfg.get("model", "veo-3.1-fast-generate-preview")
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "stages":
        return _list_stages()
    if argv and argv[0] == "stage":
        return _run_stage(argv[1:])
    if argv and argv[0] == "render":
        return _render_video(argv[1:])
    if argv and argv[0] == "stitch":
        return _stitch(argv[1:])
    if argv and argv[0] == "gen-video":
        return _gen_video_prompt(argv[1:])
    if argv and argv[0] == "veo-sync":
        return _veo_sync(argv[1:])

    ap = argparse.ArgumentParser(prog="reel", description=__doc__)
    ap.add_argument("source", nargs="?", help="path to source text (book/story/script)")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--max-scenes", type=int, default=1,
                    help="how many scenes to draft AND render (default: 1, prototype); "
                         "every shot within each rendered scene is always rendered")
    ap.add_argument("--profile", choices=["fast", "quality"], default=None,
                    help="force a single quality tier for every agent")
    ap.add_argument("--genre", default=None,
                    help="force the adaptation's genre (e.g. 'noir thriller'); "
                         "overrides config genre.value. Omit to use config / auto-detect")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed stages in --out and continue from the "
                         "first unfinished one (pair with a prior paused run)")
    ap.add_argument("--list-models", action="store_true",
                    help="show local model / profile status and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        return _list_models()
    if not args.source:
        ap.error("a SOURCE file is required (or use --list-models)")

    try:
        run(args.source, out_dir=args.out, max_scenes=args.max_scenes,
            profile_override=args.profile, resume=args.resume, genre=args.genre)
    except PipelineStopped as e:
        print(f"\n[reel] paused at '{e.stage}'. Completed stages saved in {args.out}/.")
        print(f"[reel] resume:  python -m reel.cli {args.source} "
              f"--out {args.out} --resume")
        return 0
    except KeyboardInterrupt:
        print("\n[reel] interrupted. Completed stages are saved; "
              "resume with --resume.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
