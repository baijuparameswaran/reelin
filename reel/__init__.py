"""reel — a multi-modal agentic pipeline that turns source material
(book / short story / script) into production-ready creative assets.

Iteration 1 implements the *screenplay-material* phase: ingest raw text and
produce a structural analysis, character breakdown, casting, scene list, the
creative design (soundscape / visuals / cinematography), a per-moment
storyboard, and a Fountain-formatted screenplay draft — using locally-run open
LLMs via Ollama, with a human-in-the-loop review gate after each stage.
"""

__version__ = "0.1.0"
