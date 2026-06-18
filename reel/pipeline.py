"""Pipeline orchestration for the screenplay-material phase.

Phase graph:

    ingest ─┬─▶ structure ──┐
            └─▶ characters ──┼─▶ scenes ─▶ screenplay ─▶ assemble
                             │
   (structure & characters run concurrently)

Note on parallelism: the two independent branches are submitted concurrently.
On this CPU-only host a single model serves requests sequentially, so they won't
*speed up* here — but the design is honest about the dependency graph and will
parallelize for real once a second model/GPU/remote backend is available.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agents.ingest import ingest
from .agents.structure import analyze_structure
from .agents.characters import extract_characters
from .agents.scenes import segment_scenes
from .agents.screenplay import draft_screenplay, to_fountain
from . import llm


def _log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


def run(
    input_path: str,
    out_dir: str = "output",
    max_scenes: int = 3,
    profile_override: str | None = None,
) -> dict:
    """Run the full screenplay-material phase and write artifacts to `out_dir`."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve which model each tier maps to, for the run manifest.
    fast_model = llm.resolve_model(llm.get_profile(profile_override or "fast"))
    quality_model = llm.resolve_model(llm.get_profile(profile_override or "quality"))
    _log(f"models — fast: {fast_model} | quality: {quality_model}")

    _log("1/5 ingest …")
    source = ingest(input_path)
    _log(f"      '{source['title']}' — {source['word_count']} words")

    parallel = llm.config().get("runtime", {}).get("max_parallel_agents", 1) > 1
    if parallel:
        _log("2/5 structure ‖ characters (concurrent) …")
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_struct = ex.submit(analyze_structure, source, profile_override)
            f_chars = ex.submit(extract_characters, source, profile_override)
            structure = f_struct.result()
            characters = f_chars.result()
    else:
        # Sequential on CPU/low-RAM: two different models can't co-reside.
        _log("2/5 structure → characters (sequential) …")
        structure = analyze_structure(source, profile_override)
        characters = extract_characters(source, profile_override)
    _log(f"      logline: {structure.get('logline', '(parse failed)')[:80]}")
    _log(f"      characters: {len(characters.get('characters', []))}")

    _log("3/5 scene segmentation …")
    scenes = segment_scenes(source, structure, profile=profile_override)
    _log(f"      {len(scenes.get('scenes', []))} scenes")

    _log(f"4/5 screenplay draft (first {max_scenes} scenes) …")
    draft = draft_screenplay(
        source, structure, characters, scenes,
        max_scenes=max_scenes, profile=profile_override,
    )

    _log("5/5 assemble artifacts …")
    fountain = to_fountain(source, structure, draft)
    project = {
        "title": source["title"],
        "source": source["source_path"],
        "word_count": source["word_count"],
        "structure": structure,
        "characters": characters,
        "scenes": scenes,
        "screenplay_draft": draft,
        "models": {"fast": fast_model, "quality": quality_model},
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    _write_json(out / "structure.json", structure)
    _write_json(out / "characters.json", characters)
    _write_json(out / "scenes.json", scenes)
    _write_json(out / "project.json", project)
    (out / "screenplay.fountain").write_text(fountain, encoding="utf-8")

    _log(f"done in {project['elapsed_seconds']}s → {out}/")
    return project


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
