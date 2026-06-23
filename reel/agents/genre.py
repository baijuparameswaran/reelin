"""Genre agent: pick the adaptation's genre and keep every stage aligned to it.

The genre is the creative throughline — it shapes structure, casting, palette,
score, camera, and dialogue. This agent owns it end to end:

  * `resolve_genre` — fix the genre once, from an explicit choice (CLI/config) or,
    when set to "auto", inferred from the storyline itself.
  * `guidance` — a compact directive injected into every creative stage's prompt so
    generation *leans into* the genre's conventions (steering).
  * `enforce_stage` — after a stage runs, score how well its output honors the
    genre and flag anything off-genre, so the operator (or a re-run) can correct it
    (enforcement). Mirrors the fidelity agent's per-stage check.
  * `score_pipeline` — aggregate the per-stage genre-alignment scores into one.

Runs on the OPEN models (Ollama) via `reel.models.text`, per the project policy
that Gemini is used only for image + video generation.
"""
from __future__ import annotations

import json

from .. import models

SYSTEM = (
    "You are a genre-savvy showrunner and development executive. You identify a "
    "story's most fitting genre and hold every department to its conventions — "
    "tone, structure, imagery, sound, pacing, dialogue. You always respond with "
    "valid JSON and nothing else."
)

DETERMINE_PROMPT = """\
Decide the single best GENRE for adapting the STORY below into a film, and spell
out the conventions every department should honor. {hint}

Respond with JSON in exactly this shape:
{{
  "genre": "primary genre (one or two words)",
  "subgenre": "optional more specific genre, or empty",
  "tone": "the mood/register this genre sets",
  "conventions": ["concrete genre convention to honor", "..."],
  "visual_language": "palette/lighting/framing hallmarks of this genre",
  "sound_language": "score/sound hallmarks of this genre",
  "pacing": "how this genre paces scenes and tension",
  "dialogue_style": "how characters speak in this genre",
  "rationale": "1-2 sentences on why this genre fits THIS story"
}}

Choose a genre the story can actually support — judge only from what the STORY
contains. Be specific and concise.

STORY:
{story}
"""

ENFORCE_PROMPT = """\
You are checking ONE stage of a film-adaptation pipeline for alignment to the
chosen GENRE. The '{stage}' stage produced the OUTPUT below. Judge whether it
honors the genre's tone and conventions.

GENRE DIRECTION:
{genre}

Respond with JSON in exactly this shape:
{{
  "stage": "{stage}",
  "genre": "{genre_name}",
  "aligned": true,
  "genre_score": 0,
  "off_genre": ["element that fights the genre's tone/conventions"],
  "missing_conventions": ["genre convention this stage should use but doesn't"],
  "verdict": "on-genre | mostly on-genre | off-genre | wrong-genre",
  "summary": "1-2 sentence assessment"
}}

`genre_score` is 0-100 (100 = fully on-genre). Reasonable creative latitude is
fine — only flag real tonal/convention drift away from the genre. Judge the
OUTPUT against the GENRE DIRECTION above.

'{stage}' STAGE OUTPUT (JSON):
{artifact}
"""


def determine_genre(story_text: str, *, explicit: str | None = None,
                    profile: str | None = None, feedback: str | None = None) -> dict:
    """Infer (or, with `explicit`, flesh out) the genre + its conventions from the
    storyline. Runs on the open models via the model abstraction."""
    hint = (f"The genre has already been chosen as '{explicit}' — keep that genre "
            f"and describe how it should shape this story." if explicit else
            "Infer the genre that best fits the story.")
    prompt = DETERMINE_PROMPT.format(hint=hint, story=(story_text or "")[:8000])
    raw = models.text(prompt, system=SYSTEM,
                      profile=profile or models.agent_profile("genre"),
                      as_json=True, feedback=feedback)
    spec = models.safe_json(raw)
    if explicit and not spec.get("genre"):
        spec["genre"] = explicit
    spec.setdefault("source", "explicit" if explicit else "auto")
    return spec


def resolve_genre(source_text: str, *, explicit: str | None = None,
                  config_value: str | None = None, structure: dict | None = None,
                  profile: str | None = None) -> dict:
    """Fix the genre for the run. Priority: explicit (CLI) > config value > auto
    from the storyline. A concrete name (explicit/config, anything but "auto") is
    elaborated into a full convention spec; "auto"/unset infers from the story.

    `structure` (if available) seeds the inference with the story's own genre hint.
    """
    chosen = (explicit or "").strip() or None
    if not chosen:
        cv = (config_value or "").strip().lower()
        if cv and cv != "auto":
            chosen = config_value.strip()
    story = source_text or ""
    if structure and structure.get("genre") and not chosen:
        # let the story's own structural read seed (not override) the inference
        story = f"[structure's genre read: {structure['genre']}]\n\n{story}"
    return determine_genre(story, explicit=chosen, profile=profile)


def guidance(spec: dict | None) -> str:
    """Compact steering directive to prepend to creative stages' prompts."""
    if not spec or not spec.get("genre"):
        return ""
    g = spec.get("genre", "")
    sub = spec.get("subgenre")
    name = f"{g} ({sub})" if sub else g
    bits = [f"GENRE: {name}."]
    if spec.get("tone"):
        bits.append(f"Tone: {spec['tone']}.")
    convs = spec.get("conventions") or []
    if convs:
        bits.append("Honor these genre conventions: " + "; ".join(str(c) for c in convs[:5]) + ".")
    for key, label in (("visual_language", "Visual language"),
                       ("sound_language", "Sound"),
                       ("pacing", "Pacing"),
                       ("dialogue_style", "Dialogue")):
        if spec.get(key):
            bits.append(f"{label}: {spec[key]}.")
    return ("Keep the whole adaptation within this genre. " + " ".join(bits)).strip()


def enforce_stage(stage: str, artifact: dict, spec: dict, *,
                  profile: str | None = None, feedback: str | None = None) -> dict:
    """Score how well a single stage's output honors the chosen genre. Runs on the
    open models (a neutral grader — never steered by the genre direction itself)."""
    prompt = ENFORCE_PROMPT.format(
        stage=stage,
        genre=guidance(spec) or json.dumps(spec or {}, ensure_ascii=False)[:1500],
        genre_name=(spec or {}).get("genre", ""),
        artifact=json.dumps(artifact or {}, ensure_ascii=False)[:8000],
    )
    raw = models.text(prompt, system=SYSTEM,
                      profile=profile or models.agent_profile("genre"),
                      as_json=True, feedback=feedback)
    report = models.safe_json(raw)
    report.setdefault("stage", stage)
    report.setdefault("genre", (spec or {}).get("genre", ""))
    # Small models sometimes omit the verdict — derive it from the score so the
    # gate readout is never blank.
    score = report.get("genre_score")
    if not report.get("verdict") and isinstance(score, (int, float)):
        report["verdict"] = _verdict_for(score)
    return report


def _verdict_for(score: float) -> str:
    return ("on-genre" if score >= 85 else "mostly on-genre" if score >= 70
            else "off-genre" if score >= 50 else "wrong-genre")


def score_pipeline(reports: dict) -> dict:
    """Aggregate per-stage genre-alignment scores into one pipeline genre score:
    `overall = round(0.5*mean + 0.5*min)` of the 0-100 stage scores (the weakest
    stage caps genre consistency). Verdict bands match `_verdict_for`."""
    scores = [r.get("genre_score") for r in reports.values()
              if isinstance(r.get("genre_score"), (int, float))]
    if not scores:
        return {"overall_score": None, "verdict": "unknown",
                "checked": list(reports), "off_genre_stages": []}
    mean = sum(scores) / len(scores)
    overall = round(0.5 * mean + 0.5 * min(scores))
    off = sorted(s for s, r in reports.items()
                 if r.get("verdict") in ("off-genre", "wrong-genre")
                 or (isinstance(r.get("genre_score"), (int, float)) and r["genre_score"] < 70))
    return {
        "overall_score": overall,
        "verdict": _verdict_for(overall),
        "mean_score": round(mean, 1),
        "min_score": min(scores),
        "checked": list(reports),
        "off_genre_stages": off,
    }
