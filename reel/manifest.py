"""Print the set of models the agents depend on, for the updater script.

`python -m reel.manifest` emits one Ollama tag per line (preferred models, and
optionally fallbacks). The shell updater consumes this so model selection lives
in one place: config/models.yaml.

`--runnable-only` filters the list to models that fit within the detected GPU
VRAM + system RAM, so the update script only pulls models that can actually run.
"""
from __future__ import annotations

import argparse

from . import llm


def models(include_fallbacks: bool = False) -> list[str]:
    tags: list[str] = []
    for name in llm.config()["profiles"]:
        p = llm.get_profile(name)
        tags.append(p.model)
        if include_fallbacks:
            tags.extend(p.fallbacks)
    return list(dict.fromkeys(tags))


def runnable_models(include_fallbacks: bool = False) -> list[str]:
    """Models from the manifest that fit in detected GPU VRAM + system RAM."""
    vram = llm.gpu_vram_mb() or 0
    ram = llm.system_ram_mb()
    return [m for m in models(include_fallbacks) if llm.can_run_model(m, vram_mb=vram, ram_mb=ram)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-fallbacks", action="store_true")
    ap.add_argument("--min-ollama", action="store_true",
                    help="print the configured minimum Ollama version and exit")
    ap.add_argument("--runnable-only", action="store_true",
                    help="only print models that fit available GPU VRAM + RAM")
    ap.add_argument("--hardware", action="store_true",
                    help="print detected hardware summary and exit")
    args = ap.parse_args()
    if args.min_ollama:
        print(llm.config().get("min_ollama_version", "0.0.0"))
        return
    if args.hardware:
        print(llm.hardware_summary())
        return
    fn = runnable_models if args.runnable_only else models
    for tag in fn(args.include_fallbacks):
        print(tag)


if __name__ == "__main__":
    main()
