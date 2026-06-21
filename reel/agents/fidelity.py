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
