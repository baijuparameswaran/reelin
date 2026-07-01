"""reel.models — unified AI-model abstraction and provider policy.

Single source of truth for which provider serves each modality:

  TEXT  (every LLM stage)          → OPEN models, always (Ollama via reel.llm).
                                     Never Gemini — grader/checker agents that
                                     import this module judge neutrally, without
                                     the creative steering direction.
  IMAGE (character representation)  → Gemini if GEMINIAPIKEY is set, else the
                                     configured OPEN backend (diffusers/auto1111).
  VIDEO (scene image-to-video)      → Gemini Veo if a key is set, else the OPEN
                                     backend (diffusers/comfyui).

Image/video backends resolve the `auto` setting in config (reel.imagegen / reel.i2v):
`auto` → gemini when a key exists, else the `open_backend`.
"""
from __future__ import annotations

from . import llm

# Re-export the open-text helpers so an agent needs only `from .. import models`.
agent_profile = llm.agent_profile
with_feedback = llm.with_feedback
safe_json = llm.safe_json
config = llm.config


# ── TEXT — always open models (Ollama) ───────────────────────────────────────

def text(prompt: str, *, system: str | None = None, profile: str | None = None,
         as_json: bool = False, feedback: str | None = None, steer: bool = False) -> str:
    """Open-model text generation for grader/checker agents.

    `steer=False` by design: fidelity and genre-enforcement agents must judge
    neutrally, not under the creative steering direction. Creative agents call
    `llm.generate` directly, which steers."""
    if feedback:
        prompt = llm.with_feedback(prompt, feedback)
    return llm.generate(prompt, profile=profile, system=system, as_json=as_json, steer=steer)
