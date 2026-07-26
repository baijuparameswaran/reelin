"""reel.models — unified AI-model abstraction and provider policy.

Single source of truth for which provider serves each modality:

  TEXT  (creative stages)           → OPEN models (Ollama via reel.llm) by
                                     default. A stage may opt into the HOSTED
                                     frontier profile (`provider: gemini`, see
                                     reel.gemini.generate_text) via
                                     `agent_profiles` in config; that profile
                                     degrades back to a local one when no
                                     Gemini key is set.
  TEXT  (grader/checker stages)     → OPEN models, ALWAYS — see `text()` below.
  IMAGE (character representation)  → Gemini if GEMINIAPIKEY is set, else the
                                     configured OPEN backend (diffusers/auto1111).
  VIDEO (scene image-to-video)      → Gemini Veo if a key is set, else the OPEN
                                     backend (diffusers/comfyui).

Image/video backends resolve the `auto` setting in config (reel.imagegen / reel.i2v):
`auto` → gemini when a key exists, else the `open_backend`.

Gemini text is a deliberate, narrow exception to what used to be an absolute
("no Gemini text path exists by design"), taken because the frontier tier
needs no new credential or server-side setup — it reuses the key image/video
already use — and because ONE stage (scenes) was genuinely capped by local
context length rather than merely trading off against it.

The creative-vs-grader split above is the load-bearing part. It already
existed along one axis (`steer=False` keeps graders neutral); the hosted tier
adds a second axis along the same seam, because a grader running on the same
model that authored the artifact would be marking its own homework — and
because graders fire on every stage, so hosting them would multiply spend for
the check rather than the work.
"""
from __future__ import annotations

from . import llm

# Re-export the open-text helpers so an agent needs only `from .. import models`.
agent_profile = llm.agent_profile
with_feedback = llm.with_feedback
safe_json = llm.safe_json
config = llm.config


# ── TEXT — always open models (Ollama) ───────────────────────────────────────

def local_profile(profile: str | None) -> str | None:
    """Force `profile` down to an OPEN (Ollama) one, since graders must stay
    local — see this module's docstring for why. A hosted profile reaching a
    grader would be a configuration accident (e.g. `agent_profiles.fidelity:
    frontier`, or a hosted profile named as some tier's default), not an
    intent this module should honor silently, so it's corrected here rather
    than at each of the ~4 grader call sites. It also keeps the "graders judge
    neutrally" property meaningful once a creative stage can run on Gemini:
    the grader must not be the same model that wrote the artifact.

    Deterministic and fail-open: an unknown or unreadable profile is returned
    untouched for `llm.generate` to resolve or reject as it normally would —
    this function's job is to strip a hosted provider, not to second-guess
    profile resolution."""
    if profile is None:
        return None
    try:
        p = llm.get_profile(profile)
    except Exception:
        return profile
    if p.provider == "ollama":
        return profile
    print(f"[reel] grader asked for hosted profile '{profile}' — using "
          f"'{p.fallback_profile}' instead (graders stay local and neutral)")
    return p.fallback_profile


def text(prompt: str, *, system: str | None = None, profile: str | None = None,
         as_json: bool = False, feedback: str | None = None, steer: bool = False) -> str:
    """Open-model text generation for grader/checker agents.

    `steer=False` by design: fidelity and genre-enforcement agents must judge
    neutrally, not under the creative steering direction. Creative agents call
    `llm.generate` directly, which steers.

    Also pinned to a LOCAL provider via `local_profile` — the second half of
    the same independence rule; see this module's docstring."""
    if feedback:
        prompt = llm.with_feedback(prompt, feedback)
    return llm.generate(prompt, profile=local_profile(profile), system=system,
                        as_json=as_json, steer=steer)
