"""Command-line entry point for the reel screenplay-material pipeline.

Usage:
    python -m reel.cli SOURCE.txt [--out DIR] [--max-scenes N] [--profile NAME]
    python -m reel.cli --list-models     # show local model status

Run via the package so relative imports resolve: `python -m reel.cli ...`.
"""
from __future__ import annotations

import argparse
import sys

from . import llm
from .pipeline import run


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reel", description=__doc__)
    ap.add_argument("source", nargs="?", help="path to source text (book/story/script)")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--max-scenes", type=int, default=3,
                    help="how many scenes to fully draft (default: 3)")
    ap.add_argument("--profile", choices=["fast", "quality"], default=None,
                    help="force a single quality tier for every agent")
    ap.add_argument("--list-models", action="store_true",
                    help="show local model / profile status and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        return _list_models()
    if not args.source:
        ap.error("a SOURCE file is required (or use --list-models)")

    run(args.source, out_dir=args.out, max_scenes=args.max_scenes,
        profile_override=args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
