"""Fidelity agent: does the generated screenplay/storyboard still tell the
original story?

After the creative pipeline has transformed the source text through structure →
scenes → screenplay → shot list/storyboard, drift can creep in (invented
characters, dropped beats, changed outcomes). This agent compares the final
screenplay (and the shot list/storyboard) back against the **original story** and
reports how faithfully the adaptation preserves it — covered beats, omissions,
inventions, contradictions, and an overall verdict.

Runs on the OPEN models (Ollama) via `reel.models.text`, per the project policy
that Gemini is used only for image + video generation.
"""
from __future__ import annotations

import json

from .. import models

SYSTEM = (
    "You are a script editor and story-continuity checker. You compare an "
    "adaptation against its source and judge fidelity honestly — crediting what "
    "is preserved and flagging what is dropped, invented, or contradicted. You "
    "always respond with valid JSON and nothing else."
)

PROMPT = """\
Compare the ORIGINAL STORY with its adapted SCREENPLAY and SHOT LIST. Judge how
faithfully the adaptation preserves the original story's premise, characters,
beats, and outcome.

Respond with JSON in exactly this shape:
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

Respond with JSON in exactly this shape:
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
