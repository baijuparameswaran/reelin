"""Fidelity agent: does the generated screenplay/storyboard still tell the
original story — and does every stage's own scene-keyed data still line up
with scenes.json, the structure the pipeline established at the start?

Two distinct checks, deliberately kept separate:

  * `check_stage`/`check_alignment`/`score_pipeline` — STORY fidelity: does
    this stage's output still tell the same story as the original source
    text? A qualitative judgment call (drift, omissions, invented material),
    so it runs on the OPEN models (Ollama) via `reel.models.text`, per the
    project policy that Gemini is used only for image + video generation.

  * `check_scene_alignment`/`strip_orphan_scenes` — SCENE-STRUCTURE
    alignment: does this stage's own per-scene data actually correspond,
    scene-number for scene-number, to scenes.json — the specific upstream
    INPUT every scene-keyed stage (soundscape/visuals/cinematography/
    screenplay/storyboard) is declared to depend on (`reel.stages.STAGES`)?
    This is NOT a judgment call — a stage missing a scene scenes.json has,
    or carrying a stale scene scenes.json no longer has (e.g. left over from
    before a `revise` edit), is an objective structural bug, not a quality
    tradeoff. So it's deterministic (no LLM, reusing `artifact_diff`'s
    scene/name-keying knowledge) and wired into `pipeline.run()` to
    self-heal automatically — reiterate the affected stage (scoped to just
    the misaligned scene numbers, via the same `existing=`/`revise_keys=`
    mechanism the `revise` CLI command uses) BEFORE the operator ever sees
    the review gate, rather than just advising them to fix it manually the
    way a low fidelity/genre score does.
"""
from __future__ import annotations

import json

from .. import artifact_diff
from .. import models

SYSTEM = (
    "You are an exacting, top-tier script editor and story-continuity "
    "checker. You compare an adaptation against its source and judge fidelity "
    "honestly — crediting what is preserved and flagging what is dropped, "
    "invented, or contradicted. You always respond with valid JSON and "
    "nothing else."
)

PROMPT = """\
Compare the ORIGINAL STORY with its adapted SCREENPLAY and SHOT LIST. Judge how
faithfully the adaptation preserves the original story's premise, characters,
beats, and outcome.

Do NOT:
- credit a beat as "covered" if the adaptation only gestures at it without the
  story's actual content
- flag an addition or contradiction that isn't genuinely unsupported by /
  inconsistent with the ORIGINAL STORY — reasonable adaptive craft is not drift
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present, no extra top-level keys, no missing keys:
{{
  "logline_alignment": "does the adaptation's through-line match the story's?",
  "covered_beats": ["story beat that is preserved", "..."],
  "omissions": ["meaningful element of the story that is missing", "..."],
  "additions": ["element invented by the adaptation that is NOT in the story", "..."],
  "contradictions": ["anything that changes or contradicts the story's facts/outcome", "..."],
  "character_fidelity": "are the characters and their roles consistent with the story?",
  "fidelity_score": 0,
  "verdict": "aligned | mostly aligned | drifting | misaligned",
  "summary": "2-3 sentence overall assessment",
  "recommendations": ["concrete fix to improve fidelity", "..."]
}}

`fidelity_score` is 0-100 (100 = a faithful adaptation). Be specific and concise;
judge only against what the ORIGINAL STORY actually contains.

ORIGINAL STORY:
{story}

ADAPTED SCREENPLAY (Fountain):
{screenplay}

SHOT LIST / STORYBOARD (JSON, may be partial):
{storyboard}
"""


STAGE_PROMPT = """\
You are checking ONE stage of an adaptation pipeline for fidelity to the ORIGINAL
STORY. The '{stage}' stage produced the OUTPUT below (it transforms the story into
{stage} material). Judge whether it stays consistent with the original story.

Do NOT:
- flag ordinary adaptive craft (added dialogue, visual/sound detail, camera
  choices) as drift — only real premise/character/beat/outcome divergence counts
- invent a drift/omission/contradiction item that isn't actually traceable to
  the OUTPUT vs. the ORIGINAL STORY
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present, no extra top-level keys, no missing keys:
{{
  "stage": "{stage}",
  "consistent": true,
  "fidelity_score": 0,
  "drift": ["element that diverges from / isn't supported by the story"],
  "omissions": ["essential story element this stage should keep but dropped"],
  "contradictions": ["anything that changes or contradicts the story's facts/outcome"],
  "verdict": "aligned | mostly aligned | drifting | misaligned",
  "summary": "1-2 sentence assessment"
}}

`fidelity_score` is 0-100 (100 = faithful). Some invention is normal in adaptation
(added dialogue, visual/sound detail, camera choices) — do NOT flag reasonable
craft. Only flag real drift: changing or contradicting the story's premise,
characters, key beats, or outcome, or dropping its essentials. Judge only against
what the ORIGINAL STORY actually contains.

ORIGINAL STORY:
{story}

'{stage}' STAGE OUTPUT (JSON):
{artifact}
"""


def check_stage(stage: str, artifact: dict, story_text: str,
                profile: str | None = None, feedback: str | None = None) -> dict:
    """Consistency check for a SINGLE stage's output against the original story.
    Runs on the open models (Ollama) via the model abstraction."""
    prompt = STAGE_PROMPT.format(
        stage=stage,
        story=(story_text or "")[:8000],
        artifact=json.dumps(artifact or {}, ensure_ascii=False)[:8000],
    )
    raw = models.text(prompt, system=SYSTEM,
                      profile=profile or models.agent_profile("fidelity"),
                      as_json=True, feedback=feedback)
    report = models.safe_json(raw)
    report.setdefault("stage", stage)
    return report


def _verdict_for(score: float) -> str:
    return ("aligned" if score >= 85 else "mostly aligned" if score >= 70
            else "drifting" if score >= 50 else "misaligned")


def score_pipeline(reports: dict) -> dict:
    """Aggregate the per-stage fidelity scores into ONE pipeline score.

    Definition: each stage reports `fidelity_score` in 0-100 (100 = faithful to the
    original story). The overall pipeline score is

        overall = round(0.5 * mean(stage scores) + 0.5 * min(stage scores))

    — i.e. half the average quality and half the weakest stage, because one badly
    drifting stage breaks story consistency for everything downstream. Verdict
    bands: >=85 aligned, 70-84 mostly aligned, 50-69 drifting, <50 misaligned.
    """
    scores = [r.get("fidelity_score") for r in reports.values()
              if isinstance(r.get("fidelity_score"), (int, float))]
    if not scores:
        return {"overall_score": None, "verdict": "unknown",
                "checked": list(reports), "drifting_stages": []}
    mean = sum(scores) / len(scores)
    overall = round(0.5 * mean + 0.5 * min(scores))
    drifting = sorted(s for s, r in reports.items()
                      if r.get("verdict") in ("drifting", "misaligned")
                      or (isinstance(r.get("fidelity_score"), (int, float))
                          and r["fidelity_score"] < 70))
    return {
        "overall_score": overall,
        "verdict": _verdict_for(overall),
        "mean_score": round(mean, 1),
        "min_score": min(scores),
        "checked": list(reports),
        "drifting_stages": drifting,
    }


def check_alignment(
    story_text: str,
    screenplay_fountain: str,
    storyboard: dict | None = None,
    profile: str | None = None,
    feedback: str | None = None,
) -> dict:
    """Compare the screenplay/storyboard to the original story; return the report.
    Runs on the open models (Ollama) via the unified model abstraction."""
    prompt = PROMPT.format(
        story=(story_text or "")[:8000],
        screenplay=(screenplay_fountain or "")[:12000],
        storyboard=json.dumps(storyboard or {}, ensure_ascii=False)[:6000],
    )
    raw = models.text(prompt, system=SYSTEM,
                      profile=profile or models.agent_profile("fidelity"),
                      as_json=True, feedback=feedback)
    return models.safe_json(raw)


def check_scene_alignment(stage: str, artifact: dict, scenes: dict) -> dict:
    """Deterministic (no LLM) scene-structure alignment check: does `stage`'s
    own scene-keyed data match scenes.json's ACTUAL scene list — the
    specific upstream input every scene-keyed stage is declared to depend
    on — with no MISSING scenes (in scenes.json, absent from this artifact)
    and no ORPHAN scenes (present here, absent from scenes.json — e.g. a
    stale leftover from before an earlier `revise` edit to scenes.json)?

    Reuses `artifact_diff.ARTIFACT_SHAPES`/`SCENE_KEYED_ARTIFACTS` (the same
    scene-keying knowledge the revision agent's diffing already relies on)
    so there's exactly one place per-artifact key/field mappings live.
    `scenes` itself and any non-scene-keyed artifact (structure, characters,
    casting, moodboard) always report aligned — there's nothing meaningful
    to check them against."""
    if stage not in artifact_diff.SCENE_KEYED_ARTIFACTS or stage == "scenes":
        return {"stage": stage, "aligned": True, "missing_scenes": [], "orphan_scenes": []}
    list_field, key_fn, *_ = artifact_diff.ARTIFACT_SHAPES[stage]
    expected = {s.get("number") for s in scenes.get("scenes", [])}
    actual = {key_fn(e) for e in artifact.get(list_field, [])}
    missing = sorted((k for k in (expected - actual) if k is not None), key=str)
    orphan = sorted((k for k in (actual - expected) if k is not None), key=str)
    return {
        "stage": stage,
        "aligned": not missing and not orphan,
        "missing_scenes": missing,
        "orphan_scenes": orphan,
    }


def strip_orphan_scenes(stage: str, artifact: dict, scenes: dict) -> dict:
    """Deterministically drop any per-scene entry whose scene_number isn't in
    scenes.json's actual set. This is the one part of "reiterate as needed"
    that reiteration via `existing=`/`revise_keys=` can't do on its own —
    that mechanism (see `reel.revision_merge.merge_by_key`) only ever adds
    or replaces keys, never deletes one, so a stale orphan scene has to be
    filtered out directly rather than "regenerated away". No-op (returns
    `artifact` unchanged, same object) when there's nothing to strip."""
    if stage not in artifact_diff.SCENE_KEYED_ARTIFACTS or stage == "scenes":
        return artifact
    list_field, key_fn, *_ = artifact_diff.ARTIFACT_SHAPES[stage]
    expected = {s.get("number") for s in scenes.get("scenes", [])}
    current = artifact.get(list_field, [])
    kept = [e for e in current if key_fn(e) in expected]
    if len(kept) == len(current):
        return artifact
    artifact = dict(artifact)
    artifact[list_field] = kept
    return artifact
