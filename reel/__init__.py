"""reel — a multi-modal agentic pipeline that turns source material
(book / short story / script) into production-ready creative assets.

Iteration 1 implements the *screenplay-material* phase: ingest raw text and
produce a structural analysis, character breakdown, casting (actor vs. character,
optionally rendered to images), scene list, the creative design (soundscape /
visuals / cinematography), a per-moment storyboard, and a Fountain-formatted
screenplay draft — using locally-run open LLMs via Ollama, with a human-in-the-loop
review gate after each stage. The character representation is rendered to an image
via the Google Gemini API (`reel.imagegen` + `reel.gemini`); an optional next
phase renders storyboard frames into video via Veo (image-to-video, seeded by the
character image, with continuity) in `reel.i2v`. All rendering is optional and
best-effort — the text pipeline runs without it, and image/video need a Gemini API
key (env `GEMINIAPIKEY`).
"""

__version__ = "0.1.0"
