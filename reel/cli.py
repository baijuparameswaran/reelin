"""Command-line entry point for the reel screenplay-material pipeline.

Usage:
    python -m reel.cli SOURCE.txt [--out DIR] [--max-scenes N] [--profile NAME] [--resume]
    python -m reel.cli --list-models             # show local model status
    python -m reel.cli stages                    # list pipeline stages + their inputs
    python -m reel.cli stage NAME [SOURCE.txt]   # run ONE stage independently

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
    ap.add_argument("--max-scenes", type=int, default=3)
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "stages":
        return _list_stages()
    if argv and argv[0] == "stage":
        return _run_stage(argv[1:])

    ap = argparse.ArgumentParser(prog="reel", description=__doc__)
    ap.add_argument("source", nargs="?", help="path to source text (book/story/script)")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--max-scenes", type=int, default=3,
                    help="how many scenes to draft AND render (default: 3); every "
                         "shot within each rendered scene is always rendered")
    ap.add_argument("--profile", choices=["fast", "quality"], default=None,
                    help="force a single quality tier for every agent")
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
            profile_override=args.profile, resume=args.resume)
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
