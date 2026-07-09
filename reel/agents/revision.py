"""Revision agent: supports the standalone `revise` CLI flow (`reel.cli`'s
`revise` command), which lets an operator hand-edit any completed stage's
JSON (or the raw ingested story text) and then selectively re-run only what's
actually affected downstream.

Two narrowly-scoped responsibilities, kept separate on purpose (see each
function's docstring for why):

  * `suggest_ripple_scenes` — an LLM call (open model, via `reel.models.text`,
    never Gemini, per the project's provider policy) that, given a
    deterministic direct diff (`reel.artifact_diff`) of what changed, flags
    any OTHER scene that might need re-examination as a narrative ripple
    effect. Advisory only — the CLI presents this as accept/reject, never
    auto-applies it.

  * `is_drastic_identity_change` — a deterministic (no LLM) heuristic that
    flags when an edited character/location description has drifted enough
    from what a locked casting entry was rendered from that the existing
    reference image might no longer match. Runs on the same "cheap,
    reproducible content fingerprint" philosophy as `pipeline._content_hash`
    — this needs to fire automatically and consistently every time,
    including in `hitl.enabled: false` unattended contexts, which an LLM call
    is a poor fit for (slower, non-deterministic, and the signal needed —
    "did the input text change a lot" — doesn't need model judgment).
"""
from __future__ import annotations

import difflib
import json

from .. import models

SYSTEM = (
    "You are a script continuity analyst reviewing a targeted revision to one "
    "part of an adaptation. Given exactly what changed and the full original "
    "story, you flag any OTHER scene that might need re-examination because it "
    "depends on what changed — a plot detail, a prop, a relationship, a piece "
    "of information one scene sets up and another pays off. You are advisory "
    "only: you never claim certainty, and you always respond with valid JSON."
)

RIPPLE_PROMPT = """\
A revision was made to scene(s) {changed_scene_numbers} of this story. Below is
the full original story text, the CURRENT scene list (after the edit), and a
summary of exactly what changed in the edited scene(s).

Your job: suggest any OTHER scene numbers (not already in the changed set) that
might need re-examination as a RIPPLE EFFECT of this change — e.g. a plot detail
established in the changed scene that a later scene depends on or pays off, a
prop/object introduced there that reappears, a piece of information a character
learns there that they act on elsewhere. Do NOT suggest a scene just because it
mentions the same characters or location in passing — only suggest it if the
SPECIFIC thing that changed plausibly affects that scene's content.

Respond with JSON in exactly this shape:
{{
  "suggested_scenes": [
    {{"scene_number": 7, "reason": "one sentence - what depends on the change"}}
  ],
  "confidence": "low | medium | high"
}}

If nothing else plausibly depends on this change, return an empty
"suggested_scenes" list — do not force suggestions.

ORIGINAL STORY:
{story}

CURRENT SCENE LIST:
{scenes}

WHAT CHANGED (scene {changed_scene_numbers}):
{change_summary}
"""


def suggest_ripple_scenes(story_text: str, scenes: dict, changed_scene_numbers: list,
                          change_summary: str, profile: str | None = None) -> dict:
    """Returns {"suggested_scenes": [{"scene_number", "reason"}], "confidence"}.
    Never auto-applied — the revise CLI presents this list for the operator to
    accept/reject by scene number. Runs on the OPEN model via models.text
    (never Gemini). Best-effort: a malformed/empty model response degrades to
    an empty suggestion list via `models.safe_json`, never raises."""
    prompt = RIPPLE_PROMPT.format(
        story=(story_text or "")[:8000],
        scenes=json.dumps(
            [{k: s[k] for k in ("number", "slugline", "summary") if k in s}
             for s in scenes.get("scenes", [])],
            ensure_ascii=False,
        ),
        changed_scene_numbers=list(changed_scene_numbers),
        change_summary=(change_summary or "")[:2000],
    )
    raw = models.text(prompt, system=SYSTEM,
                      profile=profile or models.agent_profile("revision"), as_json=True)
    result = models.safe_json(raw)
    result.setdefault("suggested_scenes", [])
    result.setdefault("confidence", "low")
    return result


def is_drastic_identity_change(old_description: str, new_description: str,
                               threshold: float = 0.55) -> tuple[bool, float]:
    """Heuristic: `difflib.SequenceMatcher` ratio over the *input* character/
    location description (the characters.json description/appearance/traits
    text, or the scenes.json-derived location summary — whichever fed the
    original casting call for this name), NOT the casting agent's own
    generated `visual_prompt` (comparing LLM-generated prose is noisier; the
    input text is the actual signal of "did the source material change
    enough to plausibly imply a different look").

    ratio() >= threshold: similar enough — treat as a normal detail edit.
    ratio() < threshold: drastic — the caller should warn and preserve the
    old casting entry/image by default rather than silently re-casting.

    threshold=0.55 is a starting heuristic (tune from real use — see
    config `revision.identity_drift_threshold`, not hardcoded forever)."""
    ratio = difflib.SequenceMatcher(None, old_description or "", new_description or "").ratio()
    return ratio < threshold, ratio
