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
    python -m reel.cli revise [--out DIR]        # revise a completed/paused run: pick any stage
                                                 # (or the raw source text) to hand-edit, then
                                                 # selectively re-run what's actually affected —
                                                 # repeat rounds until you type 'quit'/'pause'
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
from .pipeline import run, PipelineStopped


def _max_scenes_arg(v: str) -> int | None:
    """--max-scenes value: an integer count, or the literal 'all' (case-insensitive)
    for every drafted scene, unbounded (None — downstream slicing `[:None]`
    naturally means "no cap" throughout the pipeline)."""
    if v.strip().lower() == "all":
        return None
    return int(v)


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
    artificial caps. Works purely off existing artifacts; no LLM stage runs."""
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

    manifest = _render_scene_frames(board, _load_json(out / "casting.json"), out,
                                    max_scenes=a.max_scenes,
                                    characters=_load_json(out / "characters.json"))
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


_SCENE_KEYED_STAGES = {"scenes", "soundscape", "visuals", "cinematography",
                       "screenplay", "storyboard"}
_NAME_KEYED_STAGES = {"characters", "casting"}


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
    scoped, rather than being silently skipped."""
    if revise_keys is None:
        return None
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


def _revise_source(out, *, edited_override: dict | None = None, auto_confirm: bool = False) -> bool:
    """Revise the raw ingested story text. A source-text edit is ALWAYS
    treated as drastic — every downstream stage's `chunk_indices`/
    `source_line` anchors are keyed to the exact prior text, and fine-
    grained diffing of raw prose into "which scene moved" is out of scope
    for v1 (see `artifact_diff.diff_source_text`'s docstring) — so this
    falls back to a full regen of every stage, replayed via `run_stage` in
    registry order (each stage reloads its inputs fresh from disk, so the
    edited `source.json` — and each newly regenerated artifact after it —
    is what every subsequent stage actually sees)."""
    from .stages import STAGES, _load, _save_artifact, run_stage
    from .agents.ingest import chunk_text
    from .gate import edit_in_editor
    from . import artifact_diff

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

    print(f"[reel] ⚠ {diff['reason']}")
    print("[reel]   falling back to a full regen of every stage (this may take a while,")
    print("[reel]   and will re-invoke image/video generation for casting + scene_render)")
    if not (auto_confirm or input("[reel] proceed? [y/N] ").strip().lower() == "y"):
        print("[reel] revision cancelled")
        return False

    _save_artifact(out, "source", edited)
    for s in STAGES:
        if s.name == "ingest":
            continue
        run_stage(s.name, out=out)
    print("[reel] source revision applied — every downstream stage regenerated")
    return True


def _revise_one(stage_name: str, out, *, edited_override: dict | None = None,
                auto_confirm: bool = False) -> bool:
    """One revision round: edit `stage_name`'s current artifact via $EDITOR
    (or the raw source text if `stage_name == "source"`), figure out what
    actually changed, propose a downstream re-run plan (falling back to a
    full regen for a "drastic" change — see `artifact_diff`), confirm, and
    selectively re-run what's affected. Returns True if a revision was
    actually applied, False if cancelled or no change was made.

    `edited_override`/`auto_confirm` are testing hooks — bypass the
    interactive $EDITOR / confirm prompt with a canned value, so this
    function (and thus the whole revise flow) can be driven directly without
    a TTY."""
    if stage_name in ("source", "ingest"):
        # "ingest" is the stage that PRODUCES source.json (Stage.produces=
        # "source") — routed to the same special handling as "source" itself
        # (chunk_indices/word_count/char_count recomputation) rather than the
        # generic path below, which has no idea those derived fields exist.
        return _revise_source(out, edited_override=edited_override, auto_confirm=auto_confirm)

    from .stages import REGISTRY, _load, _save_artifact, run_stage, downstream_of
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
    whole_file = stage_name in artifact_diff.WHOLE_FILE_ARTIFACTS
    if whole_file:
        drastic, reason, revise_keys = True, f"'{stage_name}' has no scene/name-keyed structure to scope by", None
    else:
        diff = artifact_diff.diff_artifact(stage_name, current, edited)
        drastic, reason = diff.drastic, diff.drastic_reason
        revise_keys = set(diff.changed) | set(diff.added)
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
    elif not revise_keys:
        print("[reel] no changes detected — nothing to revise")
        return False
    else:
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
    print(f"[reel] plan: save {artifact_name}.json"
         + (f" (revising: {sorted(revise_keys, key=str)})" if revise_keys else " (full regen)")
         + (f", then re-run: {', '.join(downstream)}" if downstream else ", nothing downstream"))
    if not (auto_confirm or input("[reel] proceed? [y/N] ").strip().lower() == "y"):
        print("[reel] revision cancelled")
        return False

    _save_artifact(out, artifact_name, edited)

    for dname in downstream:
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
            run_stage(dname, out=out)
            continue
        d_existing = _load(out, REGISTRY[dname].artifact())
        d_keys = _translate_revise_keys(stage_name, dname, revise_keys, edited)
        run_stage(dname, out=out, existing=d_existing, revise_keys=d_keys)

    print(f"[reel] revision applied — {artifact_name}.json"
         + (f" and {len(downstream)} downstream stage(s)" if downstream else "") + " updated")
    return True


def _revise_loop(out) -> None:
    """The interactive stage-picker/edit/selective-rerun loop: pick any stage
    (or 'source' for the raw ingested text) to hand-edit in $EDITOR, review
    the proposed downstream re-run plan, and selectively apply it — repeat as
    many rounds as needed. Shared by the standalone `revise` command and the
    "revise now?" prompt offered right after a full pipeline run completes
    (see `_offer_revise`) — both just need to have already called
    `session.start(out, fresh=False)` before entering this loop, so it stays
    'running' across every round; it's only marked 'complete'/'paused' here
    when you type 'quit'/'exit' or 'pause' (or Ctrl-C)."""
    from . import session
    from .stages import STAGES, names, _load

    try:
        while True:
            print(f"\n[reel] revise — {out}/  (session: {session.current(out)})")
            src_mark = "✓" if (out / "source.json").exists() else " "
            print(f"  [{src_mark}] source")
            for s in STAGES:
                if s.name == "ingest":
                    continue   # same artifact as "source" above (produces="source") — not a separate entry
                mark = "✓" if _load(out, s.artifact()) is not None else " "
                print(f"  [{mark}] {s.name}")
            choice = input("\npick a stage to edit ('quit'/'exit' to finish, "
                           "'pause' to stop for now): ").strip().lower()
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
            if choice != "source" and choice not in names():
                print(f"[reel] unknown stage {choice!r}")
                continue
            _revise_one(choice, out)
    except KeyboardInterrupt:
        session.finish(out, "paused")
        print("\n[reel] revision loop paused.")


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
    a = ap.parse_args(argv)
    out = Path(a.out)

    if not out.exists():
        print(f"[reel] {out}/ doesn't exist — run the pipeline first")
        return 2

    session.start(out, fresh=False)
    _revise_loop(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path

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
    if argv and argv[0] == "revise":
        return _revise(argv[1:])
    if argv and argv[0] == "veo-sync":
        return _veo_sync(argv[1:])

    ap = argparse.ArgumentParser(prog="reel", description=__doc__)
    ap.add_argument("source", nargs="?", help="path to source text (book/story/script)")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--max-scenes", type=_max_scenes_arg, default=1,
                    help="how many scenes to RENDER — casting images and video "
                         "(default: 1, prototype), or 'all' for every scene; "
                         "every shot within each rendered scene is always rendered. "
                         "Design/planning stages (screenplay, storyboard, soundscape, "
                         "visuals, cinematography) always process every scene, "
                         "regardless of this flag")
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
        session.finish(args.out, "paused")
        print(f"\n[reel] paused at '{e.stage}'. Completed stages saved in {args.out}/.")
        print(f"[reel] resume:  python -m reel.cli {args.source} "
              f"--out {args.out} --resume")
        return 0
    except KeyboardInterrupt:
        session.finish(args.out, "paused")
        print("\n[reel] interrupted. Completed stages are saved; "
              "resume with --resume.")
        return 130
    except Exception:
        session.finish(args.out, "failed")
        raise
    _offer_revise(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
