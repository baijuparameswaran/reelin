"""Print the set of models the agents depend on, for the updater script.

`python -m reel.manifest` emits one Ollama tag per line (preferred models, and
optionally fallbacks). The shell updater consumes this so model selection lives
in one place: config/models.yaml.
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
    # de-dup, preserve order
    seen: set[str] = set()
    return [t for t in tags if not (t in seen or seen.add(t))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-fallbacks", action="store_true")
    ap.add_argument("--min-ollama", action="store_true",
                    help="print the configured minimum Ollama version and exit")
    args = ap.parse_args()
    if args.min_ollama:
        print(llm.config().get("min_ollama_version", "0.0.0"))
        return
    for tag in models(args.include_fallbacks):
        print(tag)


if __name__ == "__main__":
    main()
