# reel

> Project context for Claude Code, auto-loaded into every session here.
> This file is a slim entry point — the detail lives in three focused,
> imported files. Keep **PROGRESS.md**'s Current state and Session log
> current so work carries across sessions; keep **ARCHITECTURE.md** and
> **AGENTS.md** current when a change is structural rather than a status
> update.

@ARCHITECTURE.md
@AGENTS.md
@PROGRESS.md

## Overview
`reel` is a POC for a **multi-modal, agentic pipeline** that ingests source
material (book / script / short story) and navigates the phases of adaptation
(some in parallel) to produce a movie / reel / web-series episode. Built on
**locally-run open LLMs**, developed in slow, steady iterations.

**Iteration 1 (done, thin slice working):** the agent set converts raw text into
*screenplay material* plus a full creative design — **genre** (one genre for the
film: explicit/config/auto-from-story), structure, characters, a **moodboard**
(film-wide visual-tone bible), casting (locked visual form), scenes, soundscape
(score), visuals (art production), cinematography (camera), a per-moment
storyboard fusing *all* artifacts, and a Fountain draft. Two cross-cutting agents
are set once and shape every stage: **genre** and **moodboard STEER** every
creative stage (their direction is injected into each prompt), while **genre** and
**fidelity GRADE** every stage (per-stage alignment shown at the gate; graders
judge neutrally). A human-in-the-loop gate reviews/iterates each stage.

## Where things live
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — stack, file layout, hardware
  constraints, and every established convention/decision (provider policy,
  fidelity/genre grading, steering, the revision agent, casting/rendering
  model, etc.). Read this to understand *how the system is built*.
- **[AGENTS.md](AGENTS.md)** — one reference table + notes for every module
  in `reel/agents/`: what it takes, what it produces, whether it's steered/
  graded/scoped-revisable/prompt-sandwiched. Read this to understand *what
  a specific agent does*.
- **[PROGRESS.md](PROGRESS.md)** — Current state (what's true right now) and
  the append-only Session log (newest entry at the top). Read this to
  understand *what's happened and what's next*.
