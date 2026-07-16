"""Revision agent: supports the standalone `revise` CLI flow (`reel.cli`'s
`revise` command), which lets an operator hand-edit any completed stage's
JSON (or the raw ingested story text) and then selectively re-run only what's
actually affected downstream.

Three narrowly-scoped responsibilities, kept separate on purpose (see each
function's docstring for why):

  * `identify_source_text_changes` — an LLM call that scopes a raw
    story-TEXT edit down to the specific scene numbers it actually affects,
    given a compact deterministic diff (`reel.artifact_diff.
    unified_source_diff`) and a deterministic candidate pre-filter
    (`reel.artifact_diff.candidate_changed_scenes`) rather than the whole
    story twice. Runs on `agent_profiles.revision` — deliberately the
    LARGEST available local tier (`quality_high`, see config/models.yaml)
    since this needs to correlate a diff against every existing scene's
    content reliably, not just pattern-match. Fails SAFE: any malformed,
    empty, or ambiguous response is treated as `drastic` (falls back to a
    full downstream regen) rather than silently under-scoping.

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

IDENTIFY_SYSTEM = (
    "You are one of the industry's sharpest script continuity analysts, "
    "determining the SCOPE of a hand-edit to a story's raw text — which "
    "existing scene numbers it actually affects, and whether the edit can be "
    "handled in place or requires adding/removing a scene entirely. You are "
    "precise and conservative: when genuinely unsure, you say so rather than "
    "guessing. You always respond with valid JSON."
)

IDENTIFY_CHANGES_PROMPT = """\
A hand-edit was made to the raw story text this film adaptation is built
from. Below is a DIFF showing exactly what changed ("-" lines were removed,
"+" lines were added, unmarked lines are unchanged surrounding context), and
the CURRENT scene list this story was already segmented into (each scene's
number, a short verbatim anchor from the OLD text, and what it covers).

Your job: decide which scene NUMBERS are actually affected by this change —
their content changed, was removed, or a genuinely new event was added with
no home in the current scene list — and whether this is a SCOPED edit
(existing scene numbering stays stable; the listed scenes just need their
content refreshed) or DRASTIC (the diff implies a scene should be ADDED or
REMOVED, which this revision flow doesn't support in a scoped way — the
caller falls back to a full regen instead).

DETERMINISTIC PRE-FILTER (scenes whose exact anchor text is no longer found
verbatim in the new text) — a HINT, not a certainty: the surrounding prose
may have only been reworded without the scene's actual content changing, so
don't include a scene here just because the pre-filter flagged it if your
own reading says its content is materially unchanged; conversely a real
content change your reading catches should be included even if the
pre-filter missed it: {candidates}

Do NOT:
- include a scene number just because it's mentioned or adjacent to the
  change — only genuinely different content counts
- call something drastic just because the wording changed a lot — only a
  genuinely ADDED or REMOVED event is drastic
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present, no extra top-level keys, no missing keys:
{{
  "drastic": false,
  "reason": "if drastic: one sentence why (a new event with no matching scene, or a removed event a scene depends on); empty string otherwise",
  "changed_scene_numbers": [2, 5],
  "summary": "one sentence describing what actually changed, for the operator to review before confirming"
}}

CURRENT SCENE LIST:
{scenes}

DIFF (old text -> new text):
{diff}

Before you respond, re-check against the diff above (long diffs push early
rules out of recent context — re-verify against what you just read, not
just what you remember from the pre-filter hint):
- Every number in `changed_scene_numbers` is a scene whose own content is
  genuinely different — not a scene merely mentioned or adjacent to the
  change.
- `drastic` is true whenever the diff implies a scene should be ADDED or
  REMOVED — never force a genuinely new or deleted event into an existing
  scene number just to avoid saying drastic.
"""

SYSTEM = (
    "You are one of the industry's sharpest script continuity analysts, "
    "reviewing a targeted revision to one part of an adaptation. Given "
    "exactly what changed and the full original story, you flag any OTHER "
    "scene that might need re-examination because it depends on what changed "
    "— a plot detail, a prop, a relationship, a piece of information one "
    "scene sets up and another pays off. You are advisory only: you never "
    "claim certainty, and you always respond with valid JSON."
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

Do NOT:
- suggest a scene just because it shares a character or location in passing —
  only suggest it if the SPECIFIC thing that changed plausibly affects it
- force a suggestion when nothing else plausibly depends on this change
- add commentary, preamble, or markdown code fences before or after the JSON

Respond with ONLY a single JSON object matching EXACTLY this shape — every
key present, no extra top-level keys, no missing keys:
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


def identify_source_text_changes(unified_diff: str, scenes: dict, candidates: list[int],
                                 profile: str | None = None) -> dict:
    """Scope a raw story-text edit to the scene numbers it actually affects.

    `unified_diff` (from `reel.artifact_diff.unified_source_diff`) and
    `candidates` (from `reel.artifact_diff.candidate_changed_scenes`) are
    both already-computed, deterministic inputs — this function's only job
    is the part that genuinely needs judgment: confirming/refining the
    candidate set against the actual diff content, and deciding whether the
    change is a scoped in-place edit or DRASTIC (implies a scene should be
    added or removed — out of scope for a scoped revision, same v1 "scene
    count/order stays stable" assumption `reel.artifact_diff`'s
    `ARTIFACT_SHAPES` already encodes for direct `scenes.json` edits).

    Runs on `agent_profiles.revision` (default `quality_high` — the largest
    local tier, per config/models.yaml) via `models.text` (neutral, never
    Gemini). FAILS SAFE: `drastic` defaults to `True` and
    `changed_scene_numbers` defaults to empty if the model's response is
    malformed/empty, and every returned number is sanitized against the
    actual scene numbers that exist — an invented or stale number can never
    leak into the caller's revise_keys. The caller should treat a
    `drastic=True` (or empty `changed_scene_numbers`) result as "fall back
    to a full downstream regen", the same safety net a scoped `scenes.json`
    hand-edit already falls back to for a genuinely drastic change."""
    scene_list = json.dumps(
        [{k: s[k] for k in ("number", "source_line", "summary") if k in s}
         for s in scenes.get("scenes", [])],
        ensure_ascii=False, indent=2,
    )
    prompt = IDENTIFY_CHANGES_PROMPT.format(
        candidates=candidates if candidates else
                  "(none — deterministic check found no scene's anchor text missing)",
        scenes=scene_list,
        diff=(unified_diff or "")[:8000],
    )
    raw = models.text(prompt, system=IDENTIFY_SYSTEM,
                      profile=profile or models.agent_profile("revision"), as_json=True)
    result = models.safe_json(raw)
    result.setdefault("drastic", True)
    result.setdefault("reason", "")
    result.setdefault("summary", "")
    result.setdefault("changed_scene_numbers", [])

    valid_numbers = {s.get("number") for s in scenes.get("scenes", [])}
    result["changed_scene_numbers"] = sorted(
        n for n in (result.get("changed_scene_numbers") or [])
        if isinstance(n, int) and n in valid_numbers
    )
    if not result["changed_scene_numbers"] and not result.get("reason"):
        result["drastic"] = True
        result["reason"] = "model returned no valid changed scene numbers — failing safe"
    return result


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
