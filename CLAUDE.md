# reel

> Project context for Claude Code, auto-loaded into every session here. Keep the
> **Current state** and **Session log** sections current so work carries across
> sessions.

## Overview
`reel` is a POC for a **multi-modal, agentic pipeline** that ingests source
material (book / script / short story) and navigates the phases of adaptation
(some in parallel) to produce a movie / reel / web-series episode. Built on
**locally-run open LLMs**, developed in slow, steady iterations.

**Iteration 1 (done, thin slice working):** the agent set converts raw text into
*screenplay material* plus a full creative design — structure, characters,
casting (locked visual form), scenes, soundscape (score), visuals (art
production), cinematography (camera), a per-moment storyboard fusing all four,
and a Fountain draft. A human-in-the-loop gate reviews/iterates each stage.

## Stack & layout
- **Runtime:** Python 3.12 (`.venv/`), local LLMs via **Ollama**. Sole 3rd-party
  dep is PyYAML; the Ollama client is stdlib `urllib` (in `reel/llm.py`).
- `reel/` — package. `llm.py` (model-agnostic client + profile/fallback +
  `with_feedback`), `pipeline.py` (orchestration + per-stage gates), `gate.py`
  (human-in-the-loop review gate), `cli.py` (entry), `manifest.py` (model list
  for the updater), `agents/` (ingest, structure, characters, casting, scenes,
  soundscape, visuals, cinematography, storyboard, screenplay).
- `config/models.yaml` — model profiles, per-agent profile map, `hitl` gate
  knobs, runtime knobs.
- `scripts/` — `update-models.sh` (cadence), `install-cron.sh`, `model-updates.log`.
- `samples/` — bundled test story. `output/` — generated artifacts (gitignored).
- Entry points via `Makefile`: `setup`, `demo`, `run`, `models`, `update`,
  `update-all`, `install-cron`. Run pipeline as `python -m reel.cli`.

## Hardware reality (binding constraint)
Host `Priya-Laptop`, WSL2/Ubuntu 24.04. **No NVIDIA GPU (Intel iGPU only) →
CPU-only inference.** WSL RAM raised to **~12 GB** (`MemTotal` ≈ 12 GB; 16 GB
laptop) via `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=12GB`) — up from the
original 7.6 GB cap. 4 GB swap. 872 GB disk.
- Usable models ~3–8B. An 8B q5 model (~5.7 GB) now loads without swapping, so
  the `quality` tier is comfortable. `num_ctx` still **4096** (raising to 8192
  is now plausible at 12 GB but untested — would reduce prompt truncation).
- Inference is slow (few tok/s). An early 5-stage sample run took hours of wall
  clock (inflated by laptop sleep); the pipeline is now 10 stages, so expect
  longer. Treat runs as batch jobs — and note HITL gates add operator wait time
  unless `hitl.enabled: false` (or rely on the `timeout_seconds` auto-approve).
- **WSL memory has been raised to ~12 GB** (`%UserProfile%\.wslconfig`: `[wsl2]`
  / `memory=12GB`, then `wsl --shutdown`), so the `quality` 8B tier no longer
  OOMs. If you ever revert to the 7.6 GB cap, the 8B tier gets tight again.
- **Slow stages & timeouts:** generation streams, so `runtime.request_timeout_seconds`
  (now **600 s**) is the max gap *to the next token*, not a total cap. The first
  token is the long pole — it includes model (re)load + CPU prefill — and is
  worst for the heavy `storyboard` stage (largest prompt). If it still trips,
  raise the knob (or set 0 = wait forever) and `--resume`.

## Conventions & decisions
- **Model-agnostic by design:** agents pick a *profile* (`fast`/`quality`), never
  a model name. Preferred = **Qwen3 4B / 8B**; auto-fallback to installed models
  (qwen2.5, llama3:8b, mistral, phi3) so the pipeline always runs.
- **`runtime.max_parallel_agents: 1`** here — one CPU model serves sequentially
  and two different models can't co-reside in RAM. Raise on GPU/multi-model hosts
  to actually parallelize the independent branches: structure ‖ characters,
  scenes ‖ casting, and soundscape ‖ visuals ‖ cinematography.
- **Human-in-the-loop (`hitl` in models.yaml):** every LLM stage gates for review
  — approve, or type feedback to re-run that stage (`agent(..., feedback=...)`
  appends notes via `llm.with_feedback`). `enabled: false` for batch/unattended;
  `timeout_seconds` auto-approves on idle. Parallel branches compute together,
  then gate sequentially. Each stage also writes its own `output/<stage>.json`.
- **Streaming + timeouts:** `llm.generate` streams tokens, so
  `runtime.request_timeout_seconds` (config) is an *inactivity* window, not a
  total-time cap — slow CPU stages (notably `storyboard`) finish as long as
  tokens keep flowing. Pipeline writes each stage's `output/<stage>.json` on
  approval, so a late timeout/crash never loses earlier work.
- **Pause / resume:** type `stop` (or Ctrl-C) at any gate to pause — approved
  stages stay checkpointed in `output/`. `--resume` (CLI) / `resume=True`
  (`pipeline.run`) reloads them and continues from the first unfinished stage.
  `run_group()` in `pipeline.py` is the checkpoint-aware stage runner (load → or
  compute concurrently → gate → save); stop raises `PipelineStopped`, caught in
  `cli.main`. A stage interrupted mid-flight is never half-saved — it re-runs.
- On this host prefer `--profile fast` (one model, no 5 GB reloads between agents).
- Update cadence lives in `scripts/update-models.sh` (pull + version-check +
  smoke test + log), wired weekly/monthly via `make install-cron`, runnable
  on-demand via `make update`. **WSL caveat:** cron may not run unless enabled;
  fall back to Windows Task Scheduler calling the script, or run `make update`.
- Version control: git, branch `main`.

## Current state
- **Status:** Core screenplay-material slice (ingest→structure/characters→scenes→
  screenplay) was validated end-to-end on the sample story earlier. Since then the
  pipeline was extended with **casting, soundscape, visuals, cinematography,
  storyboard, and a human-in-the-loop gate** — these are import/unit smoke-tested
  but **not yet run fully end-to-end** (a complete run is slow on this host). Next
  full run should confirm all 10 stages produce clean JSON + coherent output.
- **Blocked-on (external):** **Ollama 0.6.5 is too old to pull Qwen3** — needs
  upgrade. Requires user's sudo: run in your terminal →
  `curl -fsSL https://ollama.com/install.sh | sh`, then `make update` pulls Qwen3.
  Until then the pipeline runs on the installed qwen2.5/llama3 fallbacks.
- **Recommended next user action:** (1) upgrade Ollama (Qwen3); (2) run a full
  end-to-end pass to validate the expanded pipeline. *(WSL RAM already raised to
  ~12 GB — done.)* The storyboard stage timed out on a run at the old 300 s
  inactivity window (slow time-to-first-token); raised default to 600 s.
- **Next up (next iterations):** input chunking for long texts (currently
  truncated at ~12k chars); draft *all* scenes not just first N; richer ingest
  (PDF/EPUB/.fdx); render storyboard `image_prompt`s through an image model; then
  the *next phase* of the larger pipeline (shot list / edit / etc.).

## Session log
- 2026-06-17 — Extended the pipeline well past the thin slice. Added agents:
  **casting** (locks each character's on-screen visual form, image-ready),
  **soundscape** (background score per scene/moment), **visuals** (art production:
  color/light/props), **cinematography** (DP shot list), and **storyboard**
  (fuses casting + visuals + cinematography + soundscape into a visual image per
  moment, each with emotional & audio attributes). Enriched the character agent
  with appearance/voice/mannerisms and write per-character files. Added a
  **human-in-the-loop review gate** (`gate.py` + `hitl` config): every LLM stage
  is approve-or-iterate, with a timeout auto-approve; all agents grew a `feedback`
  param fed via `llm.with_feedback`. Independent branches run concurrently
  (scenes ‖ casting; soundscape ‖ visuals ‖ cinematography). Refactored shared
  `MAX_CHARS`, manifest de-dup, screenplay date import. Updated all docs
  (README, CLAUDE.md, package docstrings, Makefile). Import/unit smoke-tested.
  First full run reached stage 9/10 then the storyboard call hit the hard-coded
  900s socket timeout (heavy prompt + quality 8B on CPU) and crashed, losing the
  run. Fixed: `llm.generate` now **streams** (timeout became a per-token
  inactivity window, not a total cap), timeout is configurable
  (`runtime.request_timeout_seconds`), timeouts raise a clear message, and the
  pipeline **writes each stage's artifact on approval** so a late failure keeps
  earlier work. Streaming verified live against the daemon. Then added
  **pause/resume**: `stop` at a gate (or Ctrl-C) pauses with checkpoints intact;
  `--resume` reloads completed `output/<stage>.json` and continues from the first
  unfinished stage (new checkpoint-aware `run_group` in pipeline.py;
  `PipelineStopped` handled in cli). Stop→resume cycle verified with stubs.
- 2026-06-16/17 — Built iteration 1: full screenplay-material agent pipeline
  (ingest→structure/characters→scenes→screenplay), model-agnostic Ollama client
  with profile fallback, model-update cadence script + cron installer, sample +
  docs. Probed hardware (CPU-only, 7.6 GB) and adapted (num_ctx 4096, sequential
  agents). Validated end-to-end with qwen2.5/llama3 fallbacks. Found Ollama 0.6.5
  too old for Qwen3 (upgrade pending — needs user sudo).
- 2026-06-16 — Initialized repo and continuity scaffolding (git, CLAUDE.md).
