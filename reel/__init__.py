"""reel — a multi-modal agentic pipeline that turns source material
(book / short story / script) into production-ready creative assets.

Implements the screenplay-material phase: ingest raw text and produce a structural
analysis, character breakdown, casting (one character image per role via Gemini),
scene list, the creative design (soundscape / visuals / cinematography), a
per-moment storyboard (every panel is a Veo-aligned video prompt), and a
Fountain-formatted screenplay draft — using locally-run open LLMs via Ollama, with
a human-in-the-loop review gate after each stage.

An optional render phase converts the storyboard into video clips via Gemini Veo
(`reel.i2v`), seeded by character images for identity continuity, and stitches them
into a single movie. All rendering is optional and best-effort — the text pipeline
runs without it; image/video require a Gemini API key (env `GEMINIAPIKEY`).
"""

__version__ = "0.1.0"
