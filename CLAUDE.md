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
- `reel/` — package. `models.py` (**provider policy + open-text front door** —
  grader agents import this; `text()` is the only public function, always routes
  to Ollama with `steer=False` so graders stay neutral), `stages.py` (**per-stage
  registry + `run_stage` for independent invocation**), `llm.py` (open-model
  client: model-agnostic Ollama + profile/fallback + `with_feedback` + the global
  creative-**direction** steering hook `set_direction`), `pipeline.py`
  (orchestration + per-stage gates), `gate.py` (human-in-the-loop review gate),
  `imagegen.py` (pluggable text-to-image; backends: **Gemini** default, diffusers,
  auto1111), `gemini.py` (Google Gemini REST helpers — image generation + Veo
  video, stdlib urllib, API key from env), `i2v.py` (pluggable image-to-video;
  default **Gemini Veo**; also diffusers LTX/Wan or a remote endpoint; Veo
  pre-flight prompt verifier via `veo_guide.verify_prompt`), `veo_guide.py` (Veo
  prompting guide snapshot + staleness check + `verify_prompt` — validates every
  prompt against the five-element guide before Gemini API submission; manual refresh
  via `python -m reel.cli veo-sync`), `cli.py` (entry), `manifest.py` (model list
  for the updater), `fountain.py` (Fountain parser + screenplay→storyboard/shot
  builder for rendering; `to_storyboard` folds cinematography camera grammar into
  Veo-aligned prompts using strict element ordering and Veo vocabulary), `agents/`
  (ingest, **genre**, structure, **moodboard**, characters, casting, scenes,
  soundscape, visuals, cinematography, storyboard, screenplay, fidelity).
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
Host `dev-host`, WSL2/Ubuntu 24.04. **NVIDIA GeForce RTX 2070 Super, 8 GB
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
  (100% GPU, ~40 tok/s), `synthesis`=qwen3:14b (~90% GPU, ~25 tok/s, num_ctx 16384),
  `quality_high`=qwen3:30b (42% GPU + CPU, ~8 tok/s, best quality escalation target).
  Reasoning traces (`think`) are **disabled** for all profiles — set `runtime.think:
  true` (or per-profile `think: true`) to re-enable (~2× slower but higher quality).
- `max_parallel_agents: 2` — GPU holds two 4B or one 8B+4B simultaneously.

## Conventions & decisions
- **Model-agnostic by design:** agents pick a *profile* (`fast`/`quality`), never
  a model name. Preferred = **Qwen3 4B / 8B**; auto-fallback to installed models
  (qwen2.5, llama3:8b, mistral, phi3) so the pipeline always runs.
- **Provider policy (single source of truth: `reel/models.py`):** **Gemini is used
  ONLY for image + video generation, and only if `GEMINIAPIKEY` is set; every
  text/LLM stage always uses the local OPEN models (Ollama).** Image/video config
  backends are `auto` → resolve to `gemini` when a key exists, else the
  `open_backend` (diffusers/comfyui). `reel.models` exposes only `text()` for
  grader agents; image/video are invoked directly via `reel.imagegen` and `reel.i2v`
  from `pipeline.py`. `reel.llm` is the open-text engine behind `models.text`. No
  Gemini text path exists by design.
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
- **Casting data model:** each character entry has an `actor` block (performer's
  own features) and a `character` block (that actor aged/costumed into the role).
  **Image generation renders the character only** — exactly one image per
  character (`output/casting/<name>.png`) from `character.visual_prompt`, via the
  **Gemini image API** (`imagegen` backend `gemini`, default model
  `gemini-2.5-flash-image`). This character image is the **identity seed** for
  Veo. No actor render, no stock photo lookup — the old Openverse → actor →
  character img2img chain and `stock.py` have been removed. Best-effort: no API
  key → skip with a hint. **Locations are cast the same way** (`kind: "location"`
  entries, seeded from `scenes.py`'s per-scene `location` field, one cast entry
  per distinct place regardless of scene count) but have no `actor` layer — just
  a `character.visual_prompt` showing the space's own architecture/decor, the
  inverse of a character portrait's people/scene isolation. `casting` therefore
  now runs *after* `scenes` (needs its scene→location mapping), not concurrently.
- **Scene rendering = image-to-video:** after storyboard + screenplay, the pipeline
  renders each storyboard panel as a **video clip** via `i2v`
  (`pipeline._render_scene_frames` → `output/video/`), default backend **Gemini
  Veo** (`veo-3.1-fast-generate-preview`; `veo-3.1-generate-preview`/`-lite-` also
  available — see `video.model` in `config/models.yaml` for the tradeoffs). The
  first panel of each scene is seeded from the in-frame character's
  `output/casting/<name>.png` (identity anchor); later panels use the previous
  clip's last frame for **continuity** (scene boundary = cut); idempotent by a
  content hash of prompt+seed (`pipeline._stale`), so a HITL-feedback-revised
  prompt re-renders instead of silently staying stale. **Every Veo prompt is
  verified** against the guide before submission (`veo_guide.verify_prompt` —
  checks the five required elements + audio cue formatting; issues logged as
  warnings, never blocking); audio cues themselves (ambient/SFX labels,
  voice-over/off-screen clarity, background-music directive, cross-clip voice
  consistency) are **constructed** by `veo_guide`'s helper functions
  (`dialogue_cue`/`ambient_cue`/`sfx_cue`/`music_directive`/
  `no_subtitles_directive`), not hand-rolled inline, so a future non-Veo backend
  can supply an equivalent module with the same call shape. Pluggable (`video`
  block): `gemini`/`veo`, `diffusers` (LTX/Wan on a GPU), `comfyui`/`http`, or
  `none`. Best-effort: no API key → skip with a hint.
- On this host prefer `--profile fast` (one model, no 5 GB reloads between agents).
- Update cadence lives in `scripts/update-models.sh` (pull + version-check +
  smoke test + log), wired weekly/monthly via `make install-cron`, runnable
  on-demand via `make update`. **WSL caveat:** cron may not run unless enabled;
  fall back to Windows Task Scheduler calling the script, or run `make update`.
- Version control: git, branch `main`.

## Current state
- **Status:** The pipeline runs **genre → structure/characters → moodboard →
  scenes/casting → soundscape/visuals/cinematography → screenplay → storyboard →
  render**, with a human-in-the-loop gate per stage plus per-stage **fidelity** and
  **genre** scoring. **Live-verified end-to-end on this host**, including a full
  video render + stitch to `output/video/movie.mp4`, using `--resume` across
  several sessions on the bundled sample story.
- **Qwen3 installed** (`qwen3:8b`, `qwen3:4b`). `agent_profiles.scenes` is now
  `quality` (qwen3:8b) — live comparison against `fast` (qwen3:4b) on the sample
  story showed 4b mis-numbering/dropping 1-2 scenes per run (partly a matching
  bug, now fixed — see below) vs. 8b capturing all scenes correctly.
- **`google-genai` SDK now actually installed** (`pip install google-genai`,
  currently `2.10.0`) — previously silently absent despite being in
  `requirements.txt`, so every Gemini/Veo call was going through the raw urllib
  REST fallback all along. Installing it surfaced real SDK-shape bugs (fixed):
  `generate_videos`/`GenerateVideosConfig` are plural in this SDK version (code
  assumed singular); `prompt` is a top-level `generate_videos()` kwarg, not a
  config field; `Video.from_file()`/`files.download()` are keyword-only
  (`location=`/`file=`); `enhance_prompt` isn't a documented Veo 3.1 parameter at
  all (removed); `person_generation`'s allowed value is fixed by generation
  **mode** per the official parameter table — `"allow_all"` for text-to-video/
  extension, `"allow_adult"` for image-to-video — not by model tier (was
  wrongly omitted entirely in one pass, now set conditionally). None of this was
  live-load-tested end-to-end after the fixes (deliberately, to avoid spending
  API quota mid-investigation) — only offline signature-binding + Pydantic
  config validation against the real installed SDK; worth a live smoke test on
  the next real run.
- **Gemini image + video live-verified** (key managed via `python -m reel.secrets`).
  One character image per character (`output/casting/<name>.png`,
  `gemini-2.5-flash-image`); Veo renders scene clips, default
  `veo-3.1-fast-generate-preview` (also live-tested this session at the
  standard `veo-3.1-generate-preview` tier and 1080p — both confirmed working,
  reverted to `fast` by preference). Veo preview tier rate-limits (429) +
  transient op errors — `gemini.py` retries both with backoff. Clips stitched
  into `output/video/movie.mp4` automatically or via `python -m reel.cli stitch`.
- **Render steps are idempotent by content hash, not just file existence**
  (`pipeline._content_hash`/`_stale`) — a HITL-feedback-revised prompt
  (`--resume` or a standalone `stage ... --feedback` rerun) now correctly
  re-renders a stale casting image/moodboard tile/scene clip instead of being
  skipped because the file already existed.
- **Scene-segmentation fidelity guard no longer silently drops scenes** —
  `segment_scenes`' `source_line`-not-found check (which strips scenes the
  model may have hallucinated) now returns what it dropped and why, surfaced at
  the gate and in the stage log, instead of leaving unexplained gaps in the
  scene numbering. Also fixed: the match was whitespace-sensitive (source text
  keeps literal `\n` line-wraps; model-written quotes use normal spacing),
  wrongly flagging real, verbatim quotes as hallucinated.
- **Veo prompts are strictly Veo-guide-aligned** (five-element order: Subject →
  Action → Style → Camera → Focus/Ambiance; verified before every Gemini API
  call via `veo_guide.verify_prompt`, issues/warnings logged, never blocking).
  Audio cues (ambient/SFX labels, voice-over/off-screen clarity, background-music
  directive, no-subtitles, cross-clip voice consistency) are now **constructed**
  by dedicated `veo_guide` helper functions rather than hand-rolled inline in
  `pipeline.py`, so a future non-Veo backend can supply an equivalent module.
  Guide snapshot cached in `config/veo_guide_snapshot.json`; refresh manually
  with `python -m reel.cli veo-sync`.
- **Locations are now cast alongside characters, incrementally (Increments 1–4
  of 5 done; video wiring paused on purpose — see below).** `scenes.py` gives
  each scene an explicit `location` field (the plain name of its physical
  setting, identical across every scene set there, independent of DAY/NIGHT —
  a location need not recur to get one). `casting.py` casts each distinct
  location (`kind: "location"`, no `actor` layer — just a locked
  `character.visual_prompt` showing the space's own architecture/decor with no
  people/action/scene-mood baked in, the inverse of a character portrait's
  isolation rule). Because casting now needs scenes' scene→location mapping,
  **`casting` runs after `scenes`, not concurrently with it** (`pipeline.run`'s
  stage 3/4 split). `_render_casting_images` renders one reference image per
  location the same way it does per character (kind-agnostic, same hash
  invalidation); the active-names cap that limits rendering to `--max-scenes`
  now includes each capped scene's location too. `screenplay.py` and
  `storyboard.py` both got a locked-location block fed into their prompts (so
  action/`image_prompt`s stay consistent with the rendered reference) and a
  fix for a real bug: their "no characters listed → fall back to every cast
  entry" paths would have treated a location as a character (wrong fields,
  possibly wrong identity seed). Same class of bug fixed in `fountain.py`'s
  `_resolve_character` (a location's name commonly appears in its own scenes'
  action text, which would otherwise win the character-identity match).
  `visuals.py`/`soundscape.py`/`cinematography.py` all gained the `location`
  field in their scene input plus a rule that scenes sharing a location share
  its base color/lighting, ambient bed, and coverage layout. **Increment 5
  (wiring location + multi-character references into Veo's `reference_images`
  for a scene's opening frame) is intentionally not started** — paused per
  explicit instruction while this alignment pass across screenplay/storyboard/
  visuals/soundscape/cinematography/fountain was done first. Live-tested
  through `casting`→`casting_images` (both location portraits rendered
  correctly, isolated from character/scene content) and `soundscape`/`visuals`/
  `cinematography`/`screenplay` (screenplay's action independently converged on
  the same architectural details as the rendered location image — "mosaic tile
  backsplash", "exposed brick walls" — without being told the exact wording,
  confirming the locked-location block is doing its job); `storyboard` was
  still running (synthesis profile, 5 scenes sequential) when this note was
  written and not yet confirmed end-to-end.
- **Open investigation (not yet implemented):** multi-character reference
  images for Veo (`reference_images`, up to 3 `ASSET` images, mutually
  exclusive with continuity per-call — this is Increment 5 above) and for
  Gemini image gen (`refs`, up to 20 images, no such exclusivity but not wired
  through `imagegen.py` yet); Kling 3.0 as a possible alternate provider for
  cross-scene subject-locked generation, not yet vetted against its real API.
  Full notes in memory (`veo_character_consistency` — see auto-memory for this
  project).
- **Recommended next action:** Enable GPU — replace the snap Ollama:
  `! curl -fsSL https://ollama.com/install.sh | sh`
  Then re-pull: `ollama pull qwen3:4b && ollama pull qwen3:8b`.
  Verify: `ollama ps` → "PROCESSOR" should show GPU or GPU+CPU.
- **Next up:** confirm the still-running `storyboard` live test (bundle
  location/cast data, panel consistency with the rendered location image);
  Increment 5 — wire location + character references into Veo's
  `reference_images` for a scene's opening frame; live smoke-test the SDK
  video-call fixes on a real run; moodboard tile auto-render (opt-in); richer
  ingest (PDF/EPUB/.fdx); draft all scenes (not just first N); edit / sound mix
  / final cut phase.

## Session log
- 2026-07-04 — **Locations cast alongside characters (Increments 1–4 of a
  5-step plan; video wiring deliberately paused).** Extended the "lock a
  visual identity, render one reference image" treatment already used for
  characters to scene *locations* (a bar, a stadium, a café), so the
  environment stays visually consistent across every scene set there — the
  same problem casting already solved for people. Built incrementally,
  confirming each step live before the next: (1) `scenes.py` gained an
  explicit `location` field per scene, the plain name of the physical
  setting, identical across every scene sharing a real place (independent of
  DAY/NIGHT slugline formatting) — a location need not recur to get one, since
  even a single scene benefits from consistency across its own panels; (2)+(3)
  `casting.py` collects the distinct `location` values from scenes.json (pure
  dedup, no LLM call) and casts each one as a `kind: "location"` entry — no
  `actor` layer, just a locked `character.visual_prompt` — inverting the
  character-portrait isolation rule (a location reference *should* show its
  architecture/decor, just with no people/action/scene-mood baked in, so it
  stays a clean background plate); (4) `_render_casting_images` needed no
  change (already kind-agnostic — just reads `character.visual_prompt`), but
  the `--max-scenes`-capped active-names filter in `pipeline.run` was missing
  locations entirely, so a rendered scene's location portrait would've been
  silently skipped — fixed. Because casting now depends on scenes' scene→
  location mapping, **`casting` runs after `scenes`, not concurrently** —
  `pipeline.run`'s stage 3/10 (scenes) and 4/10 (casting) split, previously one
  concurrent "3–4/10" group. Live-verified: `stage casting` correctly produced
  both character entries (still isolated, gender-fixed from the prior session)
  and location entries (architecture-only, e.g. "exposed brick walls...
  empty of people... no specific time of day"); `stage casting_images`
  rendered all four to `output/casting/*.png` with correct `image_path`s.
  **User then asked whether screenplay/storyboard were aligned to the new
  format — they weren't**, surfacing a real audit: `screenplay.py`'s and
  `storyboard.py`'s "no characters listed → fall back to every cast entry"
  paths would have treated a location as a character (wrong fields, possibly
  wrong video identity seed); same bug in `fountain.py`'s `_resolve_character`
  (a location's own name commonly appears in its own scenes' action text,
  which would otherwise win the character-identity match ahead of the actual
  character). All three fixed to exclude `kind: "location"` from
  character-shaped fallbacks/matching. Added a genuine locked-location
  reference into both prompts: `screenplay.py` gained `_scene_location_brief`
  (a "Locked setting (NAME): <visual_prompt>" block per scene) and
  `storyboard.py`'s `_scene_bundles` gained a `location` key (name +
  visual_prompt + reference_image) with a prompt rule to keep every panel's
  `image_prompt` consistent with it. `visuals.py`/`soundscape.py`/
  `cinematography.py` all gained `location` in their scene input plus a rule
  that scenes sharing a location share its base color/lighting, ambient bed,
  and shot coverage layout (only mood/specific events should vary). Live-
  verified `soundscape`/`visuals`/`cinematography`/`screenplay` end-to-end on
  the current sample story — screenplay's action for the bar scene
  independently converged on "mosaic tile backsplash" and "exposed brick
  walls" (the location's own locked `visual_prompt`, not restated to it) and
  correctly used mostly voice-over per the earlier session's economy rules;
  `storyboard` was still running (synthesis profile, 5 scenes sequential — a
  long real-world wait, confirmed genuinely working via climbing
  `llama-server` CPU time/memory rather than hung) when this entry was
  written, not yet confirmed. **Increment 5 — wiring location + character
  references into Veo's `reference_images` for a scene's opening frame — was
  explicitly NOT started**, per direct instruction to pause video rendering
  and do this alignment pass first. A stray `casting.json`/`casting/` at the
  repo root (dated well before this session's work) was noticed but left
  alone, not part of this change.
- 2026-07-03 — **google-genai SDK actually installed + real API bugs fixed;
  Veo audio-cue construction moved into veo_guide.py; scenes profile bumped;
  model tier explored.** Installed `google-genai` (was missing despite being in
  `requirements.txt` — every prior "live-verified" Gemini/Veo call had actually
  been going through the raw urllib REST fallback, not the SDK). This surfaced
  several real API-shape bugs in `gemini.py`, all fixed and verified **offline**
  against the installed SDK (signature binding + Pydantic config construction —
  deliberately not live-called, per an explicit "don't burn Gemini funds"
  instruction mid-session): `client.models.generate_video` → `generate_videos`
  and `types.GenerateVideoConfig` → `GenerateVideosConfig` (this SDK version
  uses plural naming); the image-to-video branch had `prompt` inside
  `GenerateVideosConfig` (rejected — pydantic `extra_forbidden`), moved to the
  top-level `generate_videos()` kwarg where it belongs; `types.Video.from_file()`
  and `client.files.download()` are keyword-only (`location=`/`file=`), not
  positional; `enhance_prompt` is confirmed absent from the official Veo API
  parameter table for any 3.1 variant (removed outright, not tier-specific);
  `person_generation`'s allowed value is fixed by generation **mode** per that
  same table — `"allow_all"` for text-to-video/extension, `"allow_adult"` for
  image-to-video/interpolation/reference-images — NOT by model tier as
  originally assumed; an earlier pass in this same session had wrongly removed
  the field entirely after a live 400 (the value sent was just wrong for the
  mode tested), now set conditionally per branch. Also live-tested (before the
  "stop burning funds" instruction landed) that `resolution` — previously
  silently dropped by the SDK path since `_generate_video_sdk` never received
  or forwarded it — now genuinely reaches the API: confirmed 1920×1080 output
  with an intact AAC audio track at the standard `veo-3.1-generate-preview`
  tier, disproving the "1080p drops audio" concern as it applies to our direct
  API path (that risk is specific to Flow-UI export/upscale, not native
  `generateVideos` calls). **Separately:** moved the Veo-specific audio-cue
  construction (dialogue attribution, voice-over/off-screen phrasing,
  ambient/SFX labels, background-music directive, no-subtitles) out of
  `pipeline._panel_video_prompt` and into new `veo_guide.py` helper functions
  (`dialogue_cue`/`ambient_cue`/`sfx_cue`/`music_directive`/
  `no_subtitles_directive`) — confirmed via live doc fetches that Veo has no
  reserved keyword for voice-over or off-screen dialogue or for suppressing
  background music (every source, including ours previously, has to fill this
  gap with plain sentences), so VO/O.S. lines now get an explicit sentence
  instead of a bare parenthetical tag that risked Veo lip-syncing a
  should-be-off-camera line to an on-screen face. **Also this session:**
  live-compared `qwen3:4b` (fast) vs `qwen3:8b` (quality) for the `scenes`
  stage on the bundled sample story — 8b captured all scenes correctly where
  4b dropped/mis-numbered 1-2 (compounded by a genuine bug: `segment_scenes`'
  source-line match was whitespace-sensitive, so a real quote spanning a hard
  line-wrap in the source text got wrongly flagged as hallucinated and
  silently dropped — fixed by normalizing whitespace before matching, and
  separately, dropped scenes are no longer silently discarded at all: shown at
  the gate + logged with the reason). Promoted `agent_profiles.scenes` to
  `quality`. **Model tier:** live-confirmed via `ListModels` which Veo models
  this key can reach (`-generate-preview` / `-fast-generate-preview` /
  `-lite-generate-preview`, no non-preview GA tier); briefly bumped
  `video.model` to the standard tier to test quality + unlock
  `continuity_mode: extend` (Veo 3.1's native video-to-video scene extension,
  added earlier as `gemini.extend_video`), then reverted to
  `veo-3.1-fast-generate-preview` by preference. **Open investigation, not
  implemented:** whether multi-character reference images (Veo's
  `reference_images`, up to 3 `ASSET` type, confirmed mutually exclusive with
  continuity per-call; Gemini image gen's `refs`, up to 20, no such
  exclusivity but not wired through `imagegen.py`) or an alternate provider
  (Kling 3.0, unvetted marketing claims of single-call multi-scene
  subject-locking) could better solve character-integrity-across-shots —
  full notes saved to project memory rather than acted on.
- 2026-07-02 — **Veo audio-continuity best practices + hash-based render
  invalidation.** Two independent fixes. (1) **Render steps now invalidate on
  feedback, not just file-existence.** `_render_casting_images`,
  `_render_moodboard_tiles`, `_render_scene_frames` were idempotent purely by
  `path.exists()`, so a HITL-feedback-revised `visual_prompt`/panel prompt
  (via `--resume` or a standalone `stage casting --feedback` rerun) silently
  left a stale image/clip on disk. Added `_content_hash()` (sha256 over prompt
  text + seed-image bytes) + `_stale()` (compares a `.hash` sidecar) — a
  revised prompt now forces a re-render; an unchanged one still skips.
  Pre-existing renders without a sidecar are accepted as baseline (no surprise
  re-render burn on upgrade) and backfilled. `_render_scene_frames` hashes the
  seed too, so a regenerated earlier frame's new tail image cascades
  invalidation to every later frame in that scene (continuity chain). Also
  fixed a latent bug found in the process: the "already rendered, skip"
  branch never updated `prev_tail`, so a resumed run with some frames on disk
  would reset later new frames to the character anchor instead of chaining
  continuity forward. (2) **Veo audio continuity.** `_panel_video_prompt`
  (`reel/pipeline.py`) now appends a single trailing `Audio:` block after the
  five-element visual prompt: ambient is kept strictly separate from
  score/music (previously merged, which let a hallucinated score piggyback on
  the ambient cue); explicit `"No background music or score."` is asserted by
  default whenever the storyboard's `audio_overview.score_cue` is empty (Veo
  has no memory across separate clip generations, so it can otherwise
  hallucinate a score mid-scene); ambient is auto-anchored to `"dry acoustics,
  no echo"` unless it already specifies reverb/echo, so room tone doesn't
  drift shot to shot; dialogue cues now fold in the speaking character's
  `characters.voice` description (`voice_index`, new param, sourced from
  `characters.json` — the manual substitute for Veo's lack of cross-generation
  voice cloning) and end with `"No subtitles or on-screen caption text."`.
  All three toggle via new config `video.audio.{no_background_music,room_tone,
  no_subtitles}` (default on). `_render_scene_frames` gained a `characters`
  param to build `voice_index`; threaded through all three call sites
  (`pipeline.run`, `stages.py` `scene_render` — added `characters` as an
  optional input, `cli.py`'s standalone `render` command). Also added
  **`gemini.extend_video()`** — Veo 3.1's native video-to-video scene
  extension (feeds the previous *clip*, not just its last frame, so ambient/
  music audio carries forward across the cut, not just the visual); wired in
  as `video.continuity_mode: extend` (default stays `seed`, the proven
  image-seed path) with automatic fallback to `seed` on any failure — marked
  experimental in config comments since only non-"fast" veo-3.1-*-preview
  models are confirmed (via web search of ai.google.dev) to support it, and
  it requires the input to itself be Veo-generated. Verified everything via
  byte-compile + stubbed unit tests (no live API calls): hash skip/revise/
  cascade behavior, backward-compat with pre-existing un-hashed renders,
  full `_render_scene_frames` prompt assembly incl. voice_index and
  `prev_clip` threading, and `veo_guide.verify_prompt` still passes the new
  prompt shape. Not yet exercised against the live Gemini API in this session.
- 2026-06-30 — **Veo guide alignment, Veo prompt verifier, dead code cleanup.**
  Aligned all Veo video prompts to the official guide's five-element order
  (Subject → Action → Style → Camera → Focus/Ambiance) and three audio cue types
  (ambient noise / SFX / quoted dialogue) across all code paths. `fountain.py`
  `_camera()` now emits Veo vocabulary without labels ("Camera:/Framing:"); `_av()`
  returns a 3-tuple separating ambient from SFX; `to_storyboard()` assembles
  elements in strict guide order. `_VEO_FOCUS_FOUNTAIN` and `_VEO_FOCUS` extended
  to cover natural language shot-type keys (e.g. "close-up", "wide shot") not just
  abbreviations. `cinematography.py` agent prompt updated to Veo vocabulary
  (bird's eye view, worms eye, dolly in/out, etc.). `storyboard.py` agent prompt
  updated: camera angle enum uses Veo terms, `sound` field now documents the
  ambient | SFX two-part format. `_panel_video_prompt()` in `pipeline.py` splits
  `panel.sound` on `" | "` into ambient vs. SFX; `audio_overview` feeds the
  scene-level ambient. Added `veo_guide.verify_prompt()` — checks a prompt against
  the five required elements and audio formatting; called in `i2v._gen_gemini()`
  before every Gemini API call and in `cli._gen_video_prompt()` for direct
  `gen-video` commands. Auto-sync at pipeline startup removed (was noise);
  `veo-sync` is manual-only. **Dead code removed:** `stock.py` (Openverse lookup,
  fully dormant), `imagegen._gen_diffusers_img2img` + `_I2I` + `can_img2img` +
  `generate_image_from` (img2img chain retired), `models.py` public image/video
  wrappers (now only `text()` + re-exports), `veo_guide.sync_if_stale`. All
  imports verified clean.

- 2026-06-27 — **Disabled reasoning traces + GPU-aware model selection.** Set
  `runtime.think: false` and `think: false` on the `thinking` and `quality_high`
  profiles — all stages now answer directly without a CoT preamble (~2× faster per
  stage; re-enable with `runtime.think: true` or a per-profile override). Added
  GPU-aware model selection to `reel/llm.py`: `gpu_vram_mb()` + `system_ram_mb()` +
  `can_run_model(tag)` check each model's quantized weight size against available
  VRAM + 80% of system RAM; `resolve_model` prefers models that fit. `manifest.py`
  gained `--runnable-only` (only emit models that fit hardware) and `--hardware`
  (print summary). `scripts/update-models.sh` now logs detected hardware, defaults
  to pulling only runnable models, and accepts `--no-gpu-filter` to override.
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
