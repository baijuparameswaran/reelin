"""Critique agent: a second-pass, neutral review of a stage's own output
against the REQUEST that governed it — distinct from both fidelity (does
this stay TRUE TO THE SOURCE STORY) and genre (does this fit the chosen
GENRE). This asks a narrower, third question: given what this stage was
actually asked to do (its SYSTEM role + PROMPT rules template — the
governing instructions, not the fully-interpolated text, which would
include the whole source story and isn't needed to judge craft quality),
is the response a genuinely strong, complete piece of craft, or does it
show the kind of gaps a single generation pass can miss?

Wired into `pipeline._gated` (see that function's docstring) so every
creative pipeline stage gets ONE automatic critique-and-refine round —
using the SAME `rerun_fn`/`feedback` mechanism a human's typed gate
feedback already uses — before the operator ever sees the review gate.
The pre-critique raw response is preserved on disk as
`output/<stage>.0.json` (see `pipeline.run_group`) so the operator can
always see what the very first pass actually produced, regardless of what
the critique changed.

Runs on the OPEN model via `reel.models.text` — neutral (`steer=False`),
never Gemini, matching every other grader in this codebase (fidelity.py,
genre.py's `enforce_stage`). Best-effort: a malformed or missing response
fails safe to `{"verdict": "solid", ...}` rather than forcing an extra
regeneration round on top of an already-flaky critique call — the whole
point of this step is to improve output, not to become a new source of
flakiness.
"""
from __future__ import annotations

import json

from .. import models

SYSTEM = (
    "You are one of the sharpest script and story editors in the industry, "
    "doing a second-pass critique of another writer's work before it goes to "
    "the director for review. You are blunt but constructive: you only flag "
    "GENUINE gaps, weaknesses, or missed opportunities — never nitpick a "
    "stylistic choice just because you'd have made a different one. You "
    "always respond with valid JSON and nothing else."
)

PROMPT = """\
A creative pipeline stage called "{stage_name}" was asked to do the following.

STAGE ROLE:
{stage_role}

GOVERNING PROMPT TEMPLATE (the rules/schema that shaped the response — shown \
as the template itself, not the fully-filled-in version, since the specific \
story content isn't the point here):
{stage_rules}

It produced this response:
{response}

Critique this response as a second-pass editor: does it genuinely fulfill the \
stage role and rules above, or does it show gaps, missed opportunities, \
generic or underdeveloped content, or a rule it followed only superficially? \
Only flag REAL issues that would make the response genuinely stronger if \
fixed — do not invent nitpicks for the sake of having something to say, and \
do not flag a stylistic choice just because you'd have made a different one. \
If the response is already solid, say so plainly.

Do NOT:
- invent a nitpick just to have something to say, or flag a stylistic choice
  you'd have simply made differently
- mark a response "needs_improvement" without at least one concrete,
  actionable issue in "issues"
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present, no extra top-level keys, no missing keys, no commentary:
{{
  "verdict": "solid" or "needs_improvement",
  "issues": ["specific, actionable gap or weakness — empty list if solid", "..."],
  "improvement_note": "a single paragraph of direct, actionable feedback to \
the original writer telling them exactly what to fix or strengthen — empty \
string if verdict is solid"
}}
"""


def critique_stage(stage_name: str, stage_role: str, stage_rules: str,
                   response: dict, profile: str | None = None) -> dict:
    """One neutral critique pass over `response`. Returns `{"verdict",
    "issues", "improvement_note"}` — `verdict` is always one of "solid" /
    "needs_improvement" (a malformed model response, or one missing
    `verdict` entirely, defaults to "solid" rather than forcing a refine
    round on a flaky critique call itself)."""
    prompt = PROMPT.format(
        stage_name=stage_name,
        stage_role=(stage_role or "").strip(),
        stage_rules=(stage_rules or "").strip()[:6000],
        response=json.dumps(response or {}, ensure_ascii=False, indent=2)[:8000],
    )
    raw = models.text(prompt, system=SYSTEM,
                      profile=profile or models.agent_profile("critique"),
                      as_json=True)
    report = models.safe_json(raw)
    if not isinstance(report, dict):
        report = {}
    verdict = report.get("verdict")
    if verdict not in ("solid", "needs_improvement"):
        report["verdict"] = "solid"
    report.setdefault("issues", [])
    report.setdefault("improvement_note", "")
    return report
