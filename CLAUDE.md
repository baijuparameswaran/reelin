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
- **Runtime:** Python 3.12 (`.venv/`), local LLMs via **Ollama**. Core 3rd-party
  dep is PyYAML; the Ollama client is stdlib `urllib` (in `reel/llm.py`). Image
  rendering adds **optional** deps (`diffusers`/`torch`/etc., see
  `requirements-image.txt`) — the text pipeline runs without them.
- `reel/` — package. `llm.py` (model-agnostic client + profile/fallback +
  `with_feedback`), `pipeline.py` (orchestration + per-stage gates), `gate.py`
  (human-in-the-loop review gate), `imagegen.py` (pluggable text-to-image +
  img2img backend; default backend **Gemini**), `gemini.py` (Google Gemini REST
  helpers — image generation + Veo video, stdlib urllib, API key from env), `i2v.py`
  (pluggable image-to-video backend — default **Gemini Veo**; also diffusers
  LTX/Wan or a remote endpoint), `stock.py` (free CC stock-photo lookup —
  *currently unused*: kept for the diffusers actor-reference workflow), `cli.py`
  (entry), `manifest.py` (model list for the updater), `agents/` (ingest,
  structure, characters, casting, scenes, soundscape, visuals, cinematography,
  storyboard, screenplay).
- `config/models.yaml` — model profiles, per-agent profile map, `hitl` gate
  knobs, `image` block (backend/model — default Gemini), `video` block
  (image-to-video backend/model — default Gemini Veo), runtime knobs.
- **Gemini API key** (for image/video): read from env `GEMINIAPIKEY` (or
  `GEMINI_API_KEY`/`GOOGLE_API_KEY`). Without it, image/video stages no-op
  gracefully with a hint; the text pipeline is unaffected.
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
- **Casting still keeps the actor vs. character data model** (an `actor` block —
  the performer's own, role-independent features — and a `character` block — that
  actor aged/costumed/styled into the role), but **image generation is limited to
  the character representation only**: exactly one image per character
  (`output/casting/<name>.png`) from `character.visual_prompt`, via the **Gemini
  image API** (`imagegen` backend `gemini`, default model `gemini-3.1-flash-image`).
  This character image is the **identity reference** handed to the video stage. No
  actor render, no stock photo (the old Openverse → actor → character img2img chain
  is retired; `stock.py` is dormant). Best-effort: no API key → skip with a hint.
- **Scene rendering = image-to-video (next phase):** after storyboard + screenplay,
  the pipeline renders each storyboard frame as a **video clip** via `i2v`
  (`pipeline._render_scene_frames` → `output/video/`), default backend **Gemini
  Veo** (`veo-3.1-fast-generate-preview`). The **character representation image**
  is the reference: the first frame of a scene is seeded from the in-frame
  character's `output/casting/<name>.png` (identity); later frames are seeded from
  the previous frame's last image for **continuity within a scene** (scene boundary
  = reset = cut). No intermediate stills — image generation is reserved for the
  character. Pluggable (`video` block): `gemini`/`veo`, `diffusers` (LTX/Wan on a
  GPU via `pipeline_class`), `comfyui`/`http`, or `none`. Best-effort: no API key
  → skip with a hint.
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
- **Image + video now use the Gemini API.** Image generation is **limited to the
  character representation** — one `output/casting/<name>.png` per character via
  the Gemini image API (`gemini-3.1-flash-image`) — and that image is the identity
  reference for video. Scene rendering uses **Veo** (`reel/i2v.py` backend
  `gemini`) image-to-video, seeded by the character image and chained for
  continuity. Shared REST helper: `reel/gemini.py` (stdlib urllib, key from env
  `GEMINIAPIKEY`). **Code wired + imports verified; NOT yet run live** — needs
  `GEMINIAPIKEY` exported (currently unset), so both stages no-op with a hint.
  The earlier local sd-turbo + Openverse-stock + img2img chain is retired
  (`stock.py` now dormant; diffusers/auto1111/LTX backends remain as options).
- **Blocked-on (external):** **Ollama 0.6.5 is too old to pull Qwen3** — needs
  upgrade. Requires user's sudo: run in your terminal →
  `curl -fsSL https://ollama.com/install.sh | sh`, then `make update` pulls Qwen3.
  Until then the pipeline runs on the installed qwen2.5/llama3 fallbacks.
- **Recommended next user action:** (1) upgrade Ollama (Qwen3); (2) run a full
  end-to-end pass to validate the expanded pipeline. *(WSL RAM already raised to
  ~12 GB — done.)* The storyboard stage timed out on a run at the old 300 s
  inactivity window (slow time-to-first-token); raised default to 600 s.
- **Scene rendering (image-to-video) via Veo — wired, not run live.** `i2v.py`
  (backend `gemini`) + `pipeline._render_scene_frames` render each storyboard
  frame as a clip seeded by the character image, chained for continuity. Needs
  `GEMINIAPIKEY`. Per-scene clip assembly (ffmpeg concat) is not done yet. The
  diffusers LTX/Wan and comfyui/http backends remain as GPU/remote alternatives.
- **Next up (next iterations):** export `GEMINIAPIKEY` and do a full live run
  (character images → Veo scene clips → per-scene concat); input chunking for long
  texts (currently truncated at ~12k chars); draft *all* scenes not just first N;
  richer ingest (PDF/EPUB/.fdx); then the *next phase* (shot list / edit / sound
  mix / final cut).

## Session log
- 2026-06-21 — **Switched image + video to the Google Gemini API; image gen
  limited to the character representation.** Added `reel/gemini.py` (stdlib-urllib
  REST helpers per ai.google.dev docs: image `…:generateContent` with
  `responseModalities:[TEXT,IMAGE]`; Veo `…:predictLongRunning` + operation poll +
  video download; key from env `GEMINIAPIKEY`/`GEMINI_API_KEY`/`GOOGLE_API_KEY`).
  `imagegen` gained a `gemini` backend (default; model `gemini-3.1-flash-image`);
  `i2v` gained a `gemini`/Veo backend (default; `veo-3.1-fast-generate-preview`).
  Simplified `pipeline._render_casting_images` to render **only the character
  image** per character (dropped the stock→actor→character img2img chain; removed
  the stock import + actor-query helpers; `stock.py` now dormant).
  `_render_scene_frames` is now video-only: each storyboard frame → a Veo clip
  seeded by the in-frame character's representation image, chained for continuity
  (no intermediate stills). Config `image`/`video` blocks default to gemini/veo.
  Imports verified; both stages degrade gracefully with a clear hint when no API
  key is set (it currently isn't). **Not yet run live** (needs `GEMINIAPIKEY`).
  *(Uncommitted at time of writing.)*
- 2026-06-20 (later) — **Scene rendering scaffold (image-to-video).** Added
  `i2v.py`: pluggable, model-agnostic image-to-video backend (`diffusers` for
  LTX-Video/Wan/CogVideoX on a GPU via `pipeline_class`; `comfyui`/`http` to
  offload to a remote GPU; `none`), best-effort + GPU-gated. Added
  `pipeline._render_scene_frames`, run after storyboard+screenplay: renders each
  storyboard frame to a still (reusing `imagegen`, identity-anchored on casting
  images) then animates it into a clip, **chaining clips from the previous frame's
  last image for continuity** (scene boundary = reset = cut); writes
  `output/video/scene_NN/frame_MM.{png,mp4}` + `manifest.json`, idempotent. New
  `video` config block. Web-searched current open I2V SOTA (June 2026): LTX-2,
  Wan 2.x, HunyuanVideo 1.5 — LTX/Wan are the efficient picks; all need a GPU.
  Verified the stills+continuity path on CPU (2-frame Edith scene; frame 2 img2img
  from frame 1 — same person, continuous look); clips correctly skipped (no GPU).
  *(Uncommitted at time of writing.)*
- 2026-06-20 — **Image rendering for casting, with actor/character identity.**
  Added `imagegen.py` (pluggable text-to-image: `diffusers` / `auto1111` / none,
  best-effort, lazy-imported optional deps) and wired it into the casting stage
  (`pipeline._render_casting_images`) — renders a portrait per character from its
  `visual_prompt`, idempotent, paths linked back into `casting.json`. Verified it
  actually generates correct images from the sample `casting.json` (sd-turbo on
  CPU). Then **restructured casting into actor vs. character**: each entry now has
  an `actor` block (invented, role-independent performer features + neutral
  prompt) and a `character` block (age/costume/mannerism/defining_feature +
  transformed prompt); updated `casting.py` schema, `storyboard.py` (reads the
  character block), and the pipeline summary/render. Added **`stock.py`** — free
  CC stock-photo lookup via **Openverse** (no API key, modification-allowed
  licenses, attribution captured) — and **img2img** in `imagegen` so the render
  chain is **stock reference → actor → character**, keeping the same real face
  throughout (verified: same person carried across all three on the sample cast;
  `CREDITS.json` written). Stock photo is used as a *reference* to generate the
  actor (grounded but AI-made/license-clean), `use_as: direct` to use it as-is.
  Relaxed framing from forced full-length to open portrait (img2img inherits the
  reference's framing, so fighting it hurt quality). New `image` config block
  (backend/model/size/img2img/stock knobs) + `requirements-image.txt` (optional
  torch/diffusers). Sample render run: 4 images, ~50 s/image. Storyboard-frame
  rendering still pending. *(All this work is currently uncommitted.)*
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
