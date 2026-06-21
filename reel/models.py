"""reel.models — unified AI-model abstraction and provider policy.

Single source of truth for which provider serves each modality, so agents talk to
*this* module instead of a specific backend:

  TEXT  (every LLM stage)          -> OPEN models, always (Ollama via reel.llm).
                                      Never Gemini.
  IMAGE (character representation)  -> Gemini if GEMINIAPIKEY is set, else the
                                      configured OPEN backend (diffusers/auto1111).
  VIDEO (scene image-to-video)      -> Gemini Veo if a key is set, else the OPEN
                                      backend (diffusers/comfyui).

Gemini is confined to image + video *generation* by design — text reasoning
(structure, characters, scenes, screenplay, storyboard, fidelity, …) always runs
on the local open models. Image/video backends resolve the `auto` setting in
config (see reel.imagegen / reel.i2v): `auto` -> gemini when a key exists, else
the `open_backend`.
"""
from __future__ import annotations

from pathlib import Path

from . import gemini, i2v, imagegen, llm

# Re-export the open-text helpers so an agent needs only `from .. import models`.
agent_profile = llm.agent_profile
with_feedback = llm.with_feedback
safe_json = llm.safe_json
config = llm.config


# ── TEXT — always open models (Ollama) ───────────────────────────────────────

def text(prompt: str, *, system: str | None = None, profile: str | None = None,
         as_json: bool = False, feedback: str | None = None) -> str:
    """Open-model text generation. The one entry point for every text stage."""
    if feedback:
        prompt = llm.with_feedback(prompt, feedback)
    return llm.generate(prompt, profile=profile, system=system, as_json=as_json)


def text_json(prompt: str, **kw) -> dict:
    return llm.safe_json(text(prompt, as_json=True, **kw))


def text_provider() -> str:
    return "open:ollama"


# ── IMAGE — Gemini if key, else open ─────────────────────────────────────────

def image_provider() -> str:
    return imagegen.backend()              # resolves 'auto' → gemini|open

def image_available() -> bool:
    return imagegen.available()

def generate_image(prompt: str, out_path) -> bool:
    return imagegen.generate_image(prompt, Path(out_path))

def generate_image_from(init_path, prompt: str, out_path, strength=None) -> bool:
    return imagegen.generate_image_from(init_path, prompt, Path(out_path), strength)


# ── VIDEO — Gemini Veo if key, else open ─────────────────────────────────────

def video_provider() -> str:
    return i2v.backend()                   # resolves 'auto' → gemini|open

def video_available() -> bool:
    return i2v.available()

def generate_clip(images, prompt: str, out_path) -> bool:
    return i2v.generate_clip(images, prompt, Path(out_path))


def providers() -> dict:
    """Resolved provider per modality (for logging / introspection)."""
    return {"text": text_provider(), "image": image_provider(), "video": video_provider()}
