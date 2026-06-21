"""Per-stage abstraction + registry for the reel pipeline.

Each stage of processing is declared once here as a `Stage`: its name, the input
artifacts it depends on (by artifact name, loaded from `output/<name>.json`), and
the function that runs it. This gives a uniform per-stage interface AND lets any
single stage be invoked **independently** with just its required inputs —

    from reel.stages import run_stage
    run_stage("scenes", out="output")          # loads source+structure checkpoints
    run_stage("structure", input_path="story.txt")   # ingests on demand

    python -m reel.cli stage scenes            # same, from the CLI
    python -m reel.cli stages                  # list stages + their inputs

— instead of re-running the whole pipeline. `run_stage` resolves each dependency
from a prior checkpoint (or ingests the source when needed), runs the stage
through the model abstraction, and saves its artifact. The full pipeline
(`reel.pipeline.run`) still orchestrates these same stages with the HITL gate,
concurrency, and resume.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .agents.casting import cast_characters
from .agents.characters import extract_characters
from .agents.cinematography import plan_cinematography
from .agents.fidelity import check_alignment
from .agents.ingest import ingest
from .agents.scenes import segment_scenes
from .agents.screenplay import draft_screenplay, to_fountain
from .agents.soundscape import design_soundscape
from .agents.storyboard import plan_storyboard
from .agents.structure import analyze_structure
from .agents.visuals import design_visuals


@dataclass
class Stage:
    name: str
    inputs: list[str]                       # required artifacts (output/<name>.json)
    run: Callable                           # run(ctx, *, out, profile, feedback, max_scenes)
    produces: str = ""                      # artifact name (defaults to `name`)
    optional: tuple[str, ...] = ()          # inputs used if present, skipped if absent
    desc: str = ""

    def artifact(self) -> str:
        return self.produces or self.name


# ── stage run callables — ctx provides each input artifact by name ───────────
# Signature is uniform: run(ctx, *, out, profile=None, feedback=None, max_scenes=3).

def _ingest(ctx, **_):
    return ingest(ctx["input_path"])

def _structure(ctx, *, profile=None, feedback=None, **_):
    return analyze_structure(ctx["source"], profile, feedback=feedback)

def _characters(ctx, *, profile=None, feedback=None, **_):
    return extract_characters(ctx["source"], profile, feedback=feedback)

def _scenes(ctx, *, profile=None, feedback=None, **_):
    return segment_scenes(ctx["source"], ctx["structure"], profile=profile, feedback=feedback)

def _casting(ctx, *, profile=None, feedback=None, **_):
    return cast_characters(ctx["structure"], ctx["characters"], profile, feedback=feedback)

def _soundscape(ctx, *, profile=None, feedback=None, **_):
    return design_soundscape(ctx["structure"], ctx["scenes"], profile, feedback=feedback)

def _visuals(ctx, *, profile=None, feedback=None, **_):
    return design_visuals(ctx["structure"], ctx["scenes"], profile, feedback=feedback)

def _cinematography(ctx, *, profile=None, feedback=None, **_):
    return plan_cinematography(ctx["structure"], ctx["scenes"], profile, feedback=feedback)

def _screenplay(ctx, *, profile=None, feedback=None, max_scenes=3, **_):
    return draft_screenplay(
        ctx["source"], ctx["structure"], ctx["characters"], ctx["scenes"],
        soundscape=ctx.get("soundscape"), visuals=ctx.get("visuals"),
        cinematography=ctx.get("cinematography"),
        max_scenes=max_scenes, profile=profile, feedback=feedback)

def _storyboard(ctx, *, profile=None, feedback=None, **_):
    return plan_storyboard(
        ctx["structure"], ctx["scenes"], ctx["casting"], ctx["soundscape"],
        ctx["visuals"], ctx["cinematography"], profile=profile, feedback=feedback)

def _casting_images(ctx, *, out, **_):
    from . import pipeline as P                      # lazy: avoid import cycle
    casting = ctx["casting"]
    P._render_casting_images(casting, Path(out))
    return casting

def _scene_render(ctx, *, out, **_):
    from . import pipeline as P
    return P._render_scene_frames(ctx["storyboard"], ctx["casting"], Path(out))

def _fidelity(ctx, *, profile=None, feedback=None, **_):
    source = ctx["source"]
    return check_alignment((source or {}).get("text", ""), ctx["screenplay_fountain"],
                           ctx.get("storyboard"), profile=profile, feedback=feedback)


# ── registry (declaration order = pipeline order) ────────────────────────────

STAGES: list[Stage] = [
    Stage("ingest", ["input_path"], _ingest, produces="source",
          desc="load & normalize the source text (deterministic)"),
    Stage("structure", ["source"], _structure, desc="logline, genre, themes, beats"),
    Stage("characters", ["source"], _characters, desc="character breakdown"),
    Stage("scenes", ["source", "structure"], _scenes, desc="numbered scene list"),
    Stage("casting", ["structure", "characters"], _casting, desc="actor/character casting"),
    Stage("soundscape", ["structure", "scenes"], _soundscape, desc="score / sound design"),
    Stage("visuals", ["structure", "scenes"], _visuals, desc="art production / look"),
    Stage("cinematography", ["structure", "scenes"], _cinematography, desc="shot list"),
    Stage("screenplay", ["source", "structure", "characters", "scenes"], _screenplay,
          optional=("soundscape", "visuals", "cinematography"),
          desc="Fountain draft (shots, attributed dialogue, V.O.)"),
    Stage("storyboard", ["structure", "scenes", "casting", "soundscape", "visuals",
                         "cinematography"], _storyboard, desc="per-moment storyboard"),
    Stage("casting_images", ["casting"], _casting_images, produces="casting",
          desc="render character representation images (image provider)"),
    Stage("scene_render", ["storyboard", "casting"], _scene_render, produces="scene_render",
          desc="render scenes frame-by-frame to video (video provider)"),
    Stage("fidelity", ["source", "screenplay_fountain"], _fidelity,
          optional=("storyboard",),
          desc="check the draft aligns with the original story (open text model)"),
]

REGISTRY: dict[str, Stage] = {s.name: s for s in STAGES}


def names() -> list[str]:
    return [s.name for s in STAGES]


# ── input resolution + independent invocation ────────────────────────────────

def _load(out: Path, name: str) -> dict | None:
    f = out / f"{name}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_input(dep: str, out: Path, input_path: str | None) -> object:
    """Resolve one dependency: a checkpoint artifact, the source text file, or the
    rendered screenplay. Raises with a clear message when it can't be found."""
    if dep == "input_path":
        if not input_path:
            raise ValueError("this stage needs a source file — pass input_path=… "
                             "(CLI: give the SOURCE argument)")
        return input_path
    if dep == "screenplay_fountain":
        f = out / "screenplay.fountain"
        if not f.exists():
            raise FileNotFoundError(f"missing {f} — run the 'screenplay' stage first")
        return f.read_text(encoding="utf-8")
    data = _load(out, dep)
    if data is not None:
        return data
    if dep == "source" and input_path:               # ingest on demand
        src = ingest(input_path)
        _save_artifact(out, "source", src)
        return src
    raise FileNotFoundError(
        f"missing input '{dep}' ({out}/{dep}.json) — run the '{dep}' stage first "
        f"(or `python -m reel.cli stage {dep}`)")


def _save_artifact(out: Path, name: str, data) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                      encoding="utf-8")


def run_stage(name: str, out: str | Path = "output", *, input_path: str | None = None,
              profile: str | None = None, feedback: str | None = None,
              max_scenes: int = 3, save: bool = True) -> dict:
    """Invoke a single stage independently. Loads each required input from its
    checkpoint in `out` (ingesting the source on demand), runs the stage, and
    writes its artifact. Returns the stage result."""
    if name not in REGISTRY:
        raise KeyError(f"unknown stage '{name}'. Known: {', '.join(names())}")
    stage = REGISTRY[name]
    outp = Path(out)
    ctx: dict = {}
    for dep in stage.inputs:
        ctx[dep] = _resolve_input(dep, outp, input_path)
    for dep in stage.optional:
        data = _load(outp, dep)
        if data is not None:
            ctx[dep] = data
    result = stage.run(ctx, out=outp, profile=profile, feedback=feedback, max_scenes=max_scenes)
    if save and isinstance(result, dict):
        _save_artifact(outp, stage.artifact(), result)
        if name == "screenplay":
            (outp / "screenplay.fountain").write_text(
                to_fountain(ctx["source"], ctx["structure"], result), encoding="utf-8")
    return result
