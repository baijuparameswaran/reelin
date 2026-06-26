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
*screenplay material* plus a full creative design — **genre** (one genre for the
film: explicit/config/auto-from-story), structure, characters, a **moodboard**
(film-wide visual-tone bible), casting (locked visual form), scenes, soundscape
(score), visuals (art production), cinematography (camera), a per-moment
storyboard fusing *all* artifacts, and a Fountain draft. Two cross-cutting agents
are set once and shape every stage: **genre** and **moodboard STEER** every
creative stage (their direction is injected into each prompt), while **genre** and
**fidelity GRADE** every stage (per-stage alignment shown at the gate; graders
judge neutrally). A human-in-the-loop gate reviews/iterates each stage.

## Stack & layout
- **Runtime:** Python 3.12 (`.venv/`), local LLMs via **Ollama**. Core 3rd-party
  dep is PyYAML; the Ollama client is stdlib `urllib` (in `reel/llm.py`). Image
  rendering adds **optional** deps (`diffusers`/`torch`/etc., see
  `requirements-image.txt`) — the text pipeline runs without them.
- `reel/` — package. `models.py` (**unified AI-model abstraction + provider
  policy** — the front door for text/image/video), `stages.py` (**per-stage
  registry + `run_stage` for independent invocation**), `llm.py` (open-model
  client: model-agnostic Ollama + profile/fallback + `with_feedback` + the global
  creative-**direction** steering hook `set_direction`), `pipeline.py`
  (orchestration + per-stage gates), `gate.py`
  (human-in-the-loop review gate), `imagegen.py` (pluggable text-to-image +
  img2img backend; default backend **Gemini**), `gemini.py` (Google Gemini REST
  helpers — image generation + Veo video, stdlib urllib, API key from env), `i2v.py`
  (pluggable image-to-video backend — default **Gemini Veo**; also diffusers
  LTX/Wan or a remote endpoint), `stock.py` (free CC stock-photo lookup —
  *currently unused*: kept for the diffusers actor-reference workflow), `cli.py`
  (entry), `manifest.py` (model list for the updater), `fountain.py` (Fountain
  parser + screenplay→storyboard/shot builder for rendering; `to_storyboard` folds
  cinematography camera grammar into the render plan), `agents/` (ingest, **genre**,
  structure, **moodboard**, characters, casting, scenes, soundscape, visuals,
  cinematography, storyboard, screenplay, fidelity).
- `config/models.yaml` — model profiles, per-agent profile map, `hitl` gate
  knobs, `genre` block (value/steer/enforce/min_score), `moodboard` block
  (enabled/steer), `fidelity` block, `image` block (backend/model — default
  Gemini), `video` block (image-to-video backend/model — default Gemini Veo),
  runtime knobs.
- **Gemini API key** (for image/video): read from env `GEMINIAPIKEY` (or
  `GEMINI_API_KEY`/`GOOGLE_API_KEY`). Without it, image/video stages no-op
  gracefully with a hint; the text pipeline is unaffected.
- `scripts/` — `update-models.sh` (cadence), `install-cron.sh`, `model-updates.log`.
- `samples/` — bundled test story. `output/` — generated artifacts (gitignored).
- Entry points via `Makefile`: `setup`, `demo`, `run`, `models`, `update`,
  `update-all`, `install-cron`. Run pipeline as `python -m reel.cli`.

## Hardware reality (binding constraint)
Host `Priya-Laptop`, WSL2/Ubuntu 24.04. **NVIDIA GeForce RTX 2070 Super, 8 GB
VRAM** (confirmed via `nvidia-smi`). WSL RAM ~12 GB (`MemTotal` ≈ 12 GB; 16 GB
laptop) via `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=12GB`). 4 GB swap.
872 GB disk.
- Installed models and VRAM fit: `qwen3:4b` (2.5 GB, full GPU), `qwen3:8b`
  (5.2 GB, full GPU), `gemma3:12b` (8.1 GB, partial offload ~85% GPU).
- **GPU requires the official Ollama installer, not the snap.** The snap package
  (v0.24.0) uses strict confinement that blocks `/dev/nvidia*` access — models
  fall back to 100% CPU. Install via:
  `curl -fsSL https://ollama.com/install.sh | sh`
  Verify with `ollama ps` — look for non-zero "Size VRAM" after loading a model.
- **With GPU:** first-token latency ~2–5 s (vs ~30–60 s on CPU). `request_timeout_seconds`
  now **120 s** (was 600 s). **Intel NPU is not accessible from WSL2** — no
  `/dev/accel`, no OpenVINO; GPU is the only accelerator usable by Ollama.
- **23 GB RAM** enables GPU+CPU split for models larger than VRAM: qwen3:14b
  (~8.9 GB, ~90% GPU) and qwen3:30b (~19 GB, ~42% GPU + ~11 GB CPU RAM).
- Profile model assignments: `fast`=qwen3:4b (100% GPU, ~80 tok/s), `quality`=qwen3:8b
  (100% GPU, ~40 tok/s), `thinking`=qwen3:14b (~90% GPU, ~25 tok/s, num_ctx 16384),
  `quality_high`=qwen3:30b (42% GPU + CPU, ~8 tok/s, best quality escalation target).
- `max_parallel_agents: 2` — GPU holds two 4B or one 8B+4B simultaneously.

## Conventions & decisions
- **Model-agnostic by design:** agents pick a *profile* (`fast`/`quality`), never
  a model name. Preferred = **Qwen3 4B / 8B**; auto-fallback to installed models
  (qwen2.5, llama3:8b, mistral, phi3) so the pipeline always runs.
- **Provider policy (single source of truth: `reel/models.py`):** **Gemini is used
  ONLY for image + video generation, and only if `GEMINIAPIKEY` is set; every
  text/LLM stage always uses the local OPEN models (Ollama).** Image/video config
  backends are `auto` → resolve to `gemini` when a key exists, else the
  `open_backend` (diffusers/comfyui). `reel.models` is the front door
  (`text()`/`generate_image()`/`generate_clip()` + `providers()`); `reel.llm` is
  the open-text engine behind `models.text`. No Gemini text path exists by design.
- **Story-fidelity at every stage (`reel/agents/fidelity.py`, open model):** each
  stage's output is scored against the **original story text** by
  `fidelity.check_stage` (`fidelity_score` 0-100, drift / omissions /
  contradictions, verdict) → `output/fidelity/<stage>.json`. The score is computed
  **before the review gate** and shown in the gate readout (`story fidelity:
  DRIFTING 62/100 ⚠ below 70 — consider re-running with feedback`, with the top
  drift items), so the operator can decide to approve or re-iterate the stage on
  the spot; `_gated` returns `(result, report)`, the threshold is config
  `fidelity.min_score` (default 70). `fidelity.score_pipeline` aggregates the
  per-stage scores into one **pipeline score**:
  `overall = round(0.5·mean + 0.5·min)` of the per-stage scores (the weakest stage
  caps consistency); verdict bands >=85 aligned / 70-84 mostly aligned / 50-69
  drifting / <50 misaligned → `output/fidelity.json` + `project.json`. Runs on the
  OPEN models (never Gemini); toggle with config `fidelity.per_stage`. Best-effort
  (a failed check never blocks the pipeline). `check_alignment` remains for a
  holistic screenplay+storyboard-vs-story check.
- **Creative direction = genre + moodboard STEER every stage (`reel/agents/genre.py`,
  `reel/agents/moodboard.py`, open models):** two cross-cutting agents are fixed
  once and shape the whole run via ONE shared steering hook. **Genre** is resolved
  up front (priority: `--genre` flag > config `genre.value` > **auto-inferred from
  the storyline**) → `output/genre.json`; **moodboard** runs right after structure
  (film-wide visual-tone bible: palette/light/texture/atmosphere/influences + render-
  ready `tiles`, the tiles **capped to `max_scenes`** so they match the scenes
  actually rendered) → `output/moodboard.json`. The pipeline composes
  `genre.guidance()+moodboard.guidance()` and calls **`llm.set_direction()`** (a
  process-wide directive prepended to the *system* message of steered generations).
  Creative agents call `llm.generate` directly → they get the direction; the
  **graders** (fidelity, genre-enforcement) call `models.text`, which passes
  `steer=False`, so they judge **neutrally**. **Genre also ENFORCES** per stage:
  `genre.enforce_stage` scores genre alignment (`genre_score`, off_genre, verdict —
  verdict back-filled from the score if the model omits it) shown at the gate next
  to fidelity; aggregate `genre.score_pipeline` → `output/genre/<stage>.json` +
  `output/genre_alignment.json`. `_gated` now returns `(result, fidelity_report,
  genre_report)`. Config `genre.{value,steer,enforce,min_score}` +
  `moodboard.{enabled,steer}`. Per policy these run on OPEN models (never Gemini) —
  **except** the moodboard's reference `tiles` are **rendered to images** via the
  image backend (`_render_moodboard_tiles` → Gemini when keyed, else open image
  backend) into `output/moodboard/tile_NN.png`, palette+lighting appended for
  coherence; that's image generation (policy-consistent), the moodboard *spec* still
  comes from the open text models. Standalone stage `moodboard_tiles`.
- **Storyboard + screenplay capture FULL detail (they drive video):** the screenplay
  agent now also gets **casting** (a "locked on-screen look" block per character, so
  action stays true to what's rendered); the storyboard agent's `_scene_bundles`
  fuses the COMPLETE detail of every artifact — cast look + voice/mannerism +
  casting `visual_prompt`/image, full visuals (filter, visual_moments, emotional_fn),
  full soundscape (sound_events, emotional_fn), full camera (framing, coverage,
  transition, per-shot emotional_fn), scene purpose, AND the screenplay's own written
  shots + attributed dialogue — and the prompt requires each `image_prompt` to be a
  self-contained, render-ready video prompt. Pipeline passes screenplay←`casting`,
  storyboard←`characters`+`draft`+`genre`.
- **Standalone video render (`python -m reel.cli render [--fresh]`):** builds a
  camera-directed render plan from `screenplay.fountain`+`cinematography.json` (every
  drafted scene, every shot — NO caps by default) via `fountain.to_storyboard`
  (cinematography camera grammar folded into each Veo prompt), then renders clips
  with `i2v` — no LLM stage runs. `gemini.generate_video` retries HTTP 429/5xx AND
  transient Veo **operation** errors (codes 8/13/14) with backoff (preview tier
  rate-limits hard).
- **Per-stage abstraction + independent invocation (`reel/stages.py`):** every
  stage of processing is declared once as a `Stage` (name, the input artifacts it
  depends on, the agent it runs, what it `produces`). The registry lets the
  pipeline treat stages uniformly AND lets any single stage be run on its own with
  just its required inputs — `run_stage("scenes", out=…)` /
  `python -m reel.cli stage scenes` — resolving each dependency from a prior
  `output/<input>.json` checkpoint (ingesting the source on demand), running it
  through the model abstraction, and writing its artifact. `python -m reel.cli
  stages` lists stages + inputs. Stage runs are direct (no HITL gate; pass
  `--feedback` for a revision note). `reel.pipeline.run` still orchestrates the
  same stages with the gate, concurrency, and resume; `run()` now also
  checkpoints `source.json` so source-dependent stages are independently runnable.
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
- **Status:** The pipeline now runs **genre → structure/characters → moodboard →
  scenes/casting → soundscape/visuals/cinematography → screenplay → storyboard →
  render**, with a human-in-the-loop gate per stage plus per-stage **fidelity** and
  **genre** scoring. The earlier creative stages were validated end-to-end on the
  sample; the newest additions (genre, moodboard, steering, full-detail storyboard/
  screenplay, standalone render) are **import/byte-compile/offline-verified and the
  genre agent is live-verified** (auto-detect + guidance + enforce on the sample),
  but a full single end-to-end run with everything on is still slow/pending on this
  CPU host.
- **Qwen3 installed** (`qwen3:8b`, `qwen3:4b`) — the old "Ollama too old for Qwen3"
  blocker is resolved; both profiles resolve to Qwen3.
- **Gemini image + video are live-verified** (key in `~/.bashrc` as `GEMINIAPIKEY`).
  Image generation is **limited to the character representation** (one
  `output/casting/<name>.png` per character); Veo (`reel/i2v.py`) renders scene
  clips seeded by that image, chained for continuity. **Veo preview tier rate-limits
  hard (429)** and occasionally returns transient operation errors (code 13) — the
  `gemini.py` client now **retries both with backoff**. After rendering, clips are
  **stitched into one `output/video/movie.mp4`** (`i2v.stitch` → ffmpeg concat
  demuxer, stream-copy with re-encode fallback; native audio preserved) — done
  automatically at the end of `_render_scene_frames`, or on demand via
  `python -m reel.cli stitch`.
- **Genre + moodboard steer the whole run; both grade-able stages stay neutral.**
  Verified the steering exemption (creative `llm.generate` gets the direction;
  `models.text` graders don't). Moodboard `tiles` are capped to `max_scenes`.
- **Recommended next user action:** Enable GPU by replacing the snap Ollama:
  `! curl -fsSL https://ollama.com/install.sh | sh` (then re-pull models with
  `ollama pull qwen3:4b && ollama pull qwen3:8b && ollama pull gemma3:12b`).
  Verify GPU with `ollama ps` — "PROCESSOR" column should show "GPU" or "GPU+CPU".
  Then run `make run` / `make demo` for a full end-to-end pass.
- **Next up (next iterations):** auto-render moodboard `tiles` (opt-in, like casting
  images); per-scene clip assembly (ffmpeg concat); input chunking for long texts
  (truncated ~12k chars); draft *all* scenes not just first N; richer ingest
  (PDF/EPUB/.fdx); then the *next phase* (edit / sound mix / final cut).

## Session log
- 2026-06-25 — **Better models for 23 GB RAM + 8 GB VRAM.** Intel NPU confirmed
  inaccessible from WSL2 (no `/dev/accel`, no OpenVINO). With 22 GB available RAM,
  models larger than VRAM are now viable via GPU+CPU split. Upgraded `thinking`
  profile: qwen3:8b → **qwen3:14b** (~8.9 GB, ~90% GPU, num_ctx 16384, think:true)
  — synthesis stages (storyboard/screenplay/structure) benefit most from extra
  params + CoT. Upgraded `quality_high`: gemma3:12b → **qwen3:30b** (~19 GB,
  42% GPU + 11 GB CPU RAM, think:true) — best available escalation target on this
  hardware. `fast`/`quality` unchanged (qwen3:4b/8b already perfectly sized).
  Pull needed: `ollama pull qwen3:14b && ollama pull qwen3:30b`.
- 2026-06-25 — **GPU enablement + model tuning.** Identified that the Ollama snap
  package (v0.24.0) uses strict confinement blocking `/dev/nvidia*` access, causing
  100% CPU inference despite an RTX 2070 Super (8 GB VRAM) being present. Fixed
  `config/models.yaml`: uncommented and enabled `num_gpu: -1` (full GPU offload);
  raised `num_ctx` from 4096 → 8192 for fast/quality/thinking profiles (VRAM
  headroom); kept quality_high at 4096 (partial offload for gemma3:12b); raised
  `max_parallel_agents` 1 → 2 (GPU can hold two models concurrently); lowered
  `request_timeout_seconds` 600 → 120 (GPU first-token ~2–5 s vs 30–60 s on CPU).
  Updated CLAUDE.md hardware section (was "Intel iGPU only → CPU-only", now
  documents the actual RTX 2070). **User action required** to take effect:
  replace the snap with the official Ollama installer (see Recommended next action).


- 2026-06-22 — **Genre + moodboard agents, full-detail storyboard/screenplay,
  standalone video render.** Added two cross-cutting agents that are fixed once and
  shape every stage. **`reel/agents/genre.py`**: one genre per run (priority
  `--genre` > config `genre.value` > auto-from-story), `guidance()` for steering,
  `enforce_stage()` per-stage alignment score, `score_pipeline()` aggregate.
  **`reel/agents/moodboard.py`**: film-wide visual-tone bible from structure+genre
  (palette/light/texture/atmosphere/influences + render-ready `tiles` **capped to
  `max_scenes`**), runs right after structure. **Steering hook**: `llm.set_direction`
  prepends a process-wide directive to *steered* generations; `models.text` passes
  `steer=False` so the fidelity/genre **graders judge neutrally** (creative agents
  call `llm.generate` directly → steered). Pipeline composes genre+moodboard into the
  direction (`apply_direction()`), runs the moodboard as a gated stage, and adds a
  per-stage **genre** check beside fidelity (`_gated` → 3-tuple; genre verdict
  back-filled from score). Config: `genre`/`moodboard` blocks + `genre`/`moodboard`
  profiles. CLI `--genre`. **Full-detail capture (these drive video):** screenplay
  agent now also takes **casting** (locked on-screen look block); storyboard
  `_scene_bundles` fuses the COMPLETE detail of every artifact incl. the
  screenplay's own shots+dialogue, and demands self-contained render-ready
  `image_prompt`s (pipeline passes screenplay←casting, storyboard←characters+draft+
  genre). **Standalone render**: `python -m reel.cli render [--fresh]` builds a
  camera-directed plan from `screenplay.fountain`+`cinematography.json` (all scenes/
  shots, no caps; `fountain.to_storyboard` folds cinematography camera grammar) and
  renders via Veo. **Gemini backoff**: `generate_video` retries HTTP 429/5xx + Veo
  op errors (8/13/14). Verified: byte-compile, all imports, steering exemption,
  tile cap, stage placement; genre live on the sample (Drama, melancholic, concrete
  conventions). Docs (README flow diagram + sections, CLAUDE.md) updated. Full
  single end-to-end run still pending (slow CPU). *(Committed.)*
- 2026-06-21 (later 6) — **Scenes capped, shots never.** `--max-scenes` (demo: 2)
  now limits drafting AND rendering to that many SCENES, but **every shot within a
  rendered scene is always rendered**: `_render_scene_frames` gained `max_scenes`
  (slices scenes, iterates all frames) and `run` passes it; `fountain.to_storyboard`
  default `max_shots=None` → all action beats become shots; the storyboard agent
  prompt now requires one frame per camera shot (cover every shot, no merge/drop).
  Verified: fountain build emits all shots per scene (5, was capped 3). Docs +
  Makefile/CLI help updated.
- 2026-06-21 (later 5) — **Show fidelity score at the review gate.** Moved the
  per-stage fidelity check to run *before* the HITL gate; `_gated` now computes the
  score for each candidate result and folds it into the gate readout (verdict +
  score/100, a "⚠ below N — consider re-running" hint under config
  `fidelity.min_score`=70, and the top drift items), then returns
  `(result, report)`. So the operator sees the story-consistency score when
  deciding to approve vs. re-iterate a stage. Verified the readout formatting +
  tuple flow (mock report; no Gemini). Docs updated (README, CLAUDE.md).
- 2026-06-21 (later 4) — **Per-stage story-fidelity + defined score.** Generalized
  the fidelity agent: `check_stage(stage, artifact, story_text)` checks any one
  stage's output against the original story (open model) → per-stage report
  (`fidelity_score` 0-100, drift/omissions/contradictions, verdict).
  `score_pipeline` defines the aggregate **pipeline score = round(0.5·mean +
  0.5·min)** of per-stage scores (verdict bands 85/70/50). Wired into
  `pipeline.run`: after each approved stage, `check_consistency` writes
  `output/fidelity/<stage>.json` and logs `verdict score/100`; assemble aggregates
  to `output/fidelity.json` + `project.json`. Config `fidelity.per_stage` (default
  true). Verified live on the open model: `check_stage("structure")` → 85 "mostly
  aligned" with concrete drift vs the lighthouse story; `score_pipeline` math
  checked. No Gemini (policy). Per-stage checks add one open-model call per stage
  (slow on CPU) — toggle off for speed.
- 2026-06-21 (later 3) — **Per-stage abstraction + independent invocation.** Added
  `reel/stages.py`: each stage declared as a `Stage` (name, input artifacts,
  run callable, `produces`) in a `REGISTRY`; `run_stage(name, …)` invokes any one
  stage standalone — resolving deps from `output/<input>.json` (ingesting SOURCE on
  demand), running through the model abstraction, saving the artifact. CLI gained
  `stage NAME [SOURCE]` and `stages` (list). `pipeline.run` now checkpoints
  `source.json`. Validated **without Gemini**: `ingest` ran end-to-end standalone;
  all 13 stages' input deps resolve from existing checkpoints (load-only, no agent
  runs). NB: the render stages (`casting_images`, `scene_render`) hit the image/
  video provider when actually run — keep them out of no-Gemini validation.
- 2026-06-21 (later 2) — **Unified model abstraction + provider policy.** Added
  `reel/models.py` as the single front door (text/image/video + `providers()`),
  enforcing: **Gemini ONLY for image + video (and only if `GEMINIAPIKEY` set);
  all text stages on the local open models.** Image/video config backends are now
  `auto` (resolve gemini-if-key-else-`open_backend`) in `imagegen.backend()` /
  `i2v.backend()`. Fixed the policy violation: the **fidelity agent now runs on
  open models** (was using Gemini) via `models.text`; removed `gemini.generate_text`
  (no Gemini text path by design). Added `fidelity` to `agent_profiles` (quality).
  Verified routing: with key → image/video=gemini; without → image/video=diffusers;
  text always open. *(Existing text agents still import `reel.llm` directly — the
  open engine behind `models.text`; migrating their call sites to `models.text` is
  optional polish, not yet done.)*
- 2026-06-21 (later) — **Live Gemini render verified; screenplay shots/V.O.;
  fidelity check.** Got the Gemini APIs working live (key in `~/.bashrc` as
  `GEMINIAPIKEY`; note `.bashrc`'s non-interactive early-return means a plain
  `source` in a non-interactive shell won't load it — eval the export line).
  **Image fix:** the v1 `:generateContent` image endpoint rejects extra
  `generationConfig` (responseModalities/responseFormat/imageConfig) — send only
  the documented minimal `contents/parts` body. Rendered all 4 character images
  live (`gemini-3.1-flash-image`, photoreal, ~13 s each). **Veo fix:** the seed
  image must be `bytesBase64Encoded` (NOT `inlineData`, which Veo rejects).
  Rendered scene clips live with `veo-3.1-fast-generate-preview` (native audio),
  seeded by character images, continuity via ffmpeg tail-frame; **Veo preview
  tier rate-limits (429)** so batches need backoff/spacing (one clip of six was
  dropped to 429). **Added `reel/fountain.py`** (parse `screenplay.fountain` →
  scenes/shots with attributed dialogue; `to_storyboard` folds visuals+soundscape
  audio into Veo prompts). **Enhanced the screenplay agent** to emit structured
  JSON: numbered **shots** (shot_type + action), **attributed dialogue**
  (speaker + modifier O.S./CONT'D + parenthetical), and **voice-over** provisions;
  `scene_to_fountain` renders proper Fountain (`!SHOT n — TYPE`, `NAME (V.O.)`).
  **Added a fidelity agent** (`reel/agents/fidelity.py` + `gemini.generate_text`):
  compares the final screenplay/storyboard against the original story → covered
  beats / omissions / additions / contradictions / score / verdict. Ran it
  (Gemini) on the lighthouse story — it correctly flagged drift: the source is
  Edith's *secret solitary* choice to abandon the light, but the draft has her
  *openly directing the crew*. **Not yet wired into `pipeline.run`**; screenplay
  agent's new structured output not yet re-run on Ollama. *(Uncommitted at write.)*
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
