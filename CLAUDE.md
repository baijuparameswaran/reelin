# reel

> Project context for Claude Code, auto-loaded into every session here. Keep the
> **Current state** and **Session log** sections current so work carries across
> sessions.

## Overview
`reel` is a POC for a **multi-modal, agentic pipeline** that ingests source
material (book / script / short story) and navigates the phases of adaptation
(some in parallel) to produce a movie / reel / web-series episode. Built on
**locally-run open LLMs**, developed in slow, steady iterations.

**Iteration 1 (done, thin slice working):** the first agent set converts raw text
into *screenplay material* — structure, characters, scenes, and a Fountain draft.

## Stack & layout
- **Runtime:** Python 3.12 (`.venv/`), local LLMs via **Ollama**. Sole 3rd-party
  dep is PyYAML; the Ollama client is stdlib `urllib` (in `reel/llm.py`).
- `reel/` — package. `llm.py` (model-agnostic client + profile/fallback logic),
  `pipeline.py` (orchestration), `cli.py` (entry), `manifest.py` (model list for
  the updater), `agents/` (ingest, structure, characters, scenes, screenplay).
- `config/models.yaml` — model profiles, per-agent profile map, runtime knobs.
- `scripts/` — `update-models.sh` (cadence), `install-cron.sh`, `model-updates.log`.
- `samples/` — bundled test story. `output/` — generated artifacts (gitignored).
- Entry points via `Makefile`: `setup`, `demo`, `run`, `models`, `update`,
  `update-all`, `install-cron`. Run pipeline as `python -m reel.cli`.

## Hardware reality (binding constraint)
Host `Priya-Laptop`, WSL2/Ubuntu 24.04. **No NVIDIA GPU (Intel iGPU only) →
CPU-only inference.** WSL RAM capped at **7.6 GB** (16 GB laptop). 872 GB disk.
- Caps usable models at ~3–8B; `num_ctx` capped at **4096** to fit a 7B in RAM
  (8192 OOMs: "requires 8.3 GiB > 8.2 available").
- Inference is slow (few tok/s). The sample run's 5 stages took hours of wall
  clock (inflated by laptop sleep). Treat runs as batch jobs.
- **`quality` tier (8B) is tight/may OOM at current RAM.** Recommended fix: raise
  WSL memory to ~12 GB — create/edit `%UserProfile%\.wslconfig` on Windows:
  `[wsl2]` / `memory=12GB`, then `wsl --shutdown` and reopen.

## Conventions & decisions
- **Model-agnostic by design:** agents pick a *profile* (`fast`/`quality`), never
  a model name. Preferred = **Qwen3 4B / 8B**; auto-fallback to installed models
  (qwen2.5, llama3:8b, mistral, phi3) so the pipeline always runs.
- **`runtime.max_parallel_agents: 1`** here — one CPU model serves sequentially
  and two different models can't co-reside in RAM. Raise on GPU/multi-model hosts
  to actually parallelize structure ‖ characters.
- On this host prefer `--profile fast` (one model, no 5 GB reloads between agents).
- Update cadence lives in `scripts/update-models.sh` (pull + version-check +
  smoke test + log), wired weekly/monthly via `make install-cron`, runnable
  on-demand via `make update`. **WSL caveat:** cron may not run unless enabled;
  fall back to Windows Task Scheduler calling the script, or run `make update`.
- Version control: git, branch `main`.

## Current state
- **Status:** Iteration 1 complete and validated end-to-end on the sample story
  (all 5 artifacts produced, JSON parsed clean, Fountain draft coherent).
- **Blocked-on (external):** **Ollama 0.6.5 is too old to pull Qwen3** — needs
  upgrade. Requires user's sudo: run in your terminal →
  `curl -fsSL https://ollama.com/install.sh | sh`, then `make update` pulls Qwen3.
  Until then the pipeline runs on the installed qwen2.5/llama3 fallbacks.
- **Recommended next user action:** (1) upgrade Ollama, (2) bump WSL RAM to 12 GB.
- **Next up (next iterations):** input chunking for long texts (currently
  truncated at ~12k chars); draft *all* scenes not just first N; richer ingest
  (PDF/EPUB/.fdx); then the *next phase* of the larger pipeline (storyboard /
  shot list / etc.).

## Session log
- 2026-06-16/17 — Built iteration 1: full screenplay-material agent pipeline
  (ingest→structure/characters→scenes→screenplay), model-agnostic Ollama client
  with profile fallback, model-update cadence script + cron installer, sample +
  docs. Probed hardware (CPU-only, 7.6 GB) and adapted (num_ctx 4096, sequential
  agents). Validated end-to-end with qwen2.5/llama3 fallbacks. Found Ollama 0.6.5
  too old for Qwen3 (upgrade pending — needs user sudo).
- 2026-06-16 — Initialized repo and continuity scaffolding (git, CLAUDE.md).
