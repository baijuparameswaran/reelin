"""The screenplay-material agents (iteration 1).

Each agent is a small, single-responsibility function that takes structured
input and returns structured output, talking to local models via `reel.llm`.
The pipeline (`reel.pipeline`) wires them together, running independent agents
concurrently where possible and gating each stage for human review.

Agents, in dependency order:
  ingest         — load & normalize raw source text (deterministic, no LLM)
  structure      — logline, genre, themes, tone, three-act beats
  characters     — cast breakdown incl. animals/birds/creatures, each defined
                   individually (undetailed background masses → one "group")
  casting        — locks each character's on-screen visual form (image-ready),
                   humans, animals, and groups alike
  scenes         — numbered, filmable scene list
  soundscape     — background score / sound design per scene
  visuals        — art production (color, lighting, props) per scene
  cinematography — director of photography: shot list per scene
  storyboard     — fuses casting + visuals + cinematography + soundscape into a
                   visual image per moment, each with emotional & audio attributes
  screenplay     — Fountain-formatted draft pages, informed by every design above

Every LLM agent accepts an optional `feedback` argument; the pipeline's review
gate passes operator notes back through it to re-run a stage on request.
"""
