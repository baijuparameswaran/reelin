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
  prompting guide snapshot + staleness check + `verify_prompt` — checks every
  prompt has a Subject/Action/Style/Camera/Focus-ish keyword present (order-
  agnostic; distinct from `pipeline.py`'s five-PART assembly ORDER, see
  Conventions & decisions below) before Gemini API submission; manual refresh
  via `python -m reel.cli veo-sync`), `cli.py` (entry), `manifest.py` (model list
  for the updater), `fountain.py` (Fountain parser + screenplay→storyboard/shot
  builder for rendering; `to_storyboard` folds cinematography camera grammar into
  Veo-aligned prompts using strict element ordering and Veo vocabulary),
  `session.py` (**session identity** — one story-to-video run's id, persisted to
  `output/session.json`; see Conventions & decisions below), `artifact_diff.py`
  (**deterministic keyed-list diff** — no LLM; compares two versions of a
  stage's JSON artifact to find which scene numbers / character-or-location
  names / panel numbers actually changed, and flags a scene-count/reorder
  change as `drastic`; powers the `revise` command's scoping), `revision_merge.py`
  (the single shared `merge_by_key` splice primitive every scoped-revision
  agent call uses — see Conventions & decisions below), `agents/`
  (ingest, **genre**, structure, **moodboard**, characters, casting, scenes,
  soundscape, visuals, cinematography, storyboard, screenplay, fidelity,
  **revision** — advisory ripple-effect suggestions + deterministic identity-
  drift detection for the `revise` command).
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
- **Scene-structure alignment (`fidelity.check_scene_alignment`/
  `strip_orphan_scenes`) — deterministic, self-healing, distinct from story
  fidelity above:** story fidelity is a *qualitative* judgment call (does this
  stage still tell the same story?), so it's an LLM grader that only *advises*
  at the gate. Scene alignment is a different question — does this stage's own
  scene-keyed data structurally match scenes.json, the actual upstream INPUT
  every scene-keyed stage (soundscape/visuals/cinematography/screenplay/
  storyboard — `artifact_diff.SCENE_KEYED_ARTIFACTS`) is declared to depend on
  — and that's not a judgment call, it's an objective bug (a scene scenes.json
  has that this stage doesn't, or a stale scene left over from before an
  earlier `revise` edit to scenes.json). So it's deterministic (no LLM, reuses
  `artifact_diff.ARTIFACT_SHAPES`'s scene-keying knowledge) and **self-heals
  automatically inside `run_group`, BEFORE the operator ever sees the review
  gate** — not just advisory like fidelity/genre. For each scene-keyed stage's
  freshly-computed result: `strip_orphan_scenes` unconditionally drops any
  entry whose scene_number scenes.json doesn't have (a `revise_keys`-scoped
  agent call can only add/replace keys, never delete one — see
  `revision_merge.merge_by_key` — so a stale orphan can only be cleared by
  filtering it out directly); any remaining MISSING scene numbers trigger one
  reiteration — the SAME agent function called again scoped to just those
  numbers (`existing=<flawed result>, revise_keys=<missing set>`, the exact
  mechanism `reel/cli.py`'s `revise` command already uses for hand-edits) —
  then a final re-check. `_spec()` gained an optional `realign(result, keys)`
  callable (soundscape/visuals/cinematography/screenplay/storyboard's specs
  each pass one; structure/characters/casting/moodboard/scenes don't — scenes
  IS the source of truth, nothing to align it against). Any gap the single
  reiteration attempt still leaves (no `realign` for that stage, or the LLM's
  scoped rerun still didn't cover it) shows up at the gate via `_format_alignment`,
  the same pattern as `_format_fidelity`/`_format_genre` but computed once by
  `run_group` rather than re-checked every gate loop iteration. **Found and
  fixed a real latent bug while building this**: `screenplay.py`'s per-scene
  loop used `scene_doc.setdefault("scene_number", scene_num)` — meaning a
  scene_number the model returned (even a wrong one) was trusted as-is, since
  `setdefault` only fills in an ABSENT key; changed to force-set it like
  `"number"` already correctly does, matching the same fix `storyboard.py`'s
  `plan_storyboard` already had (each call is sent exactly one scene, so the
  correct scene_number is always known — no reason to trust the model's echo
  over it). Verified via a full stubbed end-to-end `pipeline.run()` (not just
  isolated unit tests): a simulated soundscape omission (scene 2 missing) plus
  a simulated stale orphan (scene 99) were both corrected automatically before
  the gate — exactly 2 LLM calls for soundscape (initial + one reiteration),
  final scene_numbers `[1, 2, 3]` — and the screenplay.py bug above was caught
  BY this same test (a naive stub that always echoed `scene_number: 1`
  surfaced "still missing [2, 3] after reiteration" until the fix landed).
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
- **Storyboard is built deterministically (no LLM) except when `feedback` is
  given:** the screenplay agent gets **casting** (a "locked on-screen look" block
  per character, so action stays true to what's rendered); the storyboard agent's
  `_scene_bundles` fuses the COMPLETE detail of every artifact — cast look +
  voice/mannerism + casting `visual_prompt`/image, full visuals (filter,
  visual_moments, emotional_fn), full soundscape (sound_events, emotional_fn),
  full camera (framing, coverage, transition, per-shot emotional_fn), scene
  purpose, AND the screenplay's own written shots + attributed dialogue. Pipeline
  passes screenplay←`casting`, storyboard←`characters`+`draft`+`genre`.
  `_scene_bundles` iterates `scenes.get("scenes", [])` (the list `scenes.py` —
  "the structure formed while preparing the scenes at the beginning of the
  pipeline" — already established) and looks up every other artifact's per-scene
  entry by `scene_number`, so one bundle per scene, in scenes.json's own order,
  is structurally guaranteed regardless of what any individual upstream stage's
  output looks like. **`_build_scene_board` then constructs the ENTIRE scene
  deterministically from that bundle — no LLM call at all in the default path.**
  Rationale (from a direct "does this need an LLM?" question, followed by a
  field-by-field audit of the previous per-scene LLM prompt): a fusion stage
  isn't supposed to make new creative decisions, only faithfully combine ones
  earlier stages already made — and every output field already has an
  authoritative source in an upstream artifact. `header`/`visual_overview`/
  `audio_overview` are straight copies from the bundle (respecting
  `audio.silence` — no invented ambient bed for a deliberately silent scene).
  Panels come from `_align_shots`, which pairs cinematography's shot list
  (authoritative for panel count/order — the coverage plan) with screenplay's
  shot list (by `shot_number` when it lines up — screenplay.py's own prompt
  already instructs it to follow the camera coverage — else positional
  fallback); `dialogue` is copied verbatim from `screenplay_shots[].dialogue`
  PLUS a `vo: true` entry for `.voiceover` (a separate field an LLM had
  actually been dropping in practice); `transition` on the last panel uses
  `camera.transition_to_next` when present. The ONE field genuinely invented
  rather than sourced is `characters_in_frame` for close-shot panels — a
  documented heuristic (dialogue speaker, for CU/ECU/MCU/OTS/POV shots; full
  scene cast otherwise), since no artifact specifies who's visually in a given
  shot. `duration`/`header.duration_estimate` are likewise a documented
  heuristic (shot-type base + dialogue word count at ~2.5 words/sec), since no
  artifact provides per-shot screen time either. `storyboard_style` prefers
  `visuals.visual_palette`+`cinematography.cinematography_style`, then
  `moodboard.overall_aesthetic`, then genre+tone. **`feedback` is the one
  remaining LLM path** (`_llm_generate_scene`, the prior full per-scene
  generation prompt, `agent_profiles.storyboard: synthesis`) — a directed
  creative note ("make it darker") needs actual judgment the deterministic
  builder can't provide; a fresh run or a scoped `revise_keys` revision with no
  `feedback` is fully deterministic. That LLM path still reconciles the
  model's response against the bundle it was built for (a wrong/omitted
  `scene_number` is corrected — unambiguous, since each call sends exactly one
  scene — and an empty response is recorded in `dropped_scenes`, surfaced at
  the gate via `_summarize_storyboard`, mirroring `scenes.py`'s own
  `dropped_scenes`) rather than trusting its echo blindly.
- **Target total runtime (`--target-duration N`, default 45s — `reel/duration_budget.py`):**
  an engine-independent budget for the rendered movie's total length, threaded
  through scene/shot COUNT planning and each clip's requested render duration.
  Deliberately has NO knowledge of which video backend is configured — Veo is
  only one of several supported image-to-video engines (`i2v.py` also supports
  diffusers-based LTX/Wan/CogVideoX and a remote comfyui/http endpoint), so
  `duration_budget.py` only computes GUIDANCE TEXT (`suggest_scene_target`,
  `suggest_shots_per_scene` — "aim for about N scenes"/"about M shots per
  scene", assuming ~6s/shot for planning purposes only) fed into
  `scenes.py`'s existing `target` param and a new `cinematography.py`
  `shots_guidance` param — scene/shot counts stay each agent's own creative
  judgment call, never forced arithmetic. The actual per-clip render duration
  is a SEPARATE concern, translated by each backend's own adaptor: Veo 3.1's
  `duration_seconds` only accepts **4, 6, or 8** exactly — not a continuous
  range — and MUST be 8 outside 720p resolution or with extend-mode
  continuity (verified live against the official guide at
  ai.google.dev/gemini-api/docs/veo, not guessed) — so `i2v.py`'s
  `_veo_nearest_valid_duration`/`_gen_gemini` (Veo-specific, NOT in
  `duration_budget.py`) round the caller's requested duration to the nearest
  valid value, forced to 8 when required; the diffusers/comfyui-http backends
  (`_frames`) honor the same request directly as a frame count instead, with
  no particular constraint asserted (unverified for either). The requested
  duration itself comes from the storyboard panel's own already-computed
  `duration` estimate (`storyboard._estimate_duration`, deterministic — see
  the entry above) — parsed back to seconds by `duration_budget.
  parse_duration_seconds`, included in `_render_one_panel`'s `_content_hash`
  (so a duration-only change from a revision correctly invalidates a stale
  clip), and passed to `i2v.generate_clip(..., duration_seconds=...)`, newly
  threaded through `_gen_gemini`/`_gen_diffusers`/`_gen_http`'s signatures
  (previously Veo always got the SDK default of 8s, unconditionally, since
  nothing in the codebase ever passed a value). `_summarize_storyboard`
  gained a target-vs-estimated-total readout at the gate (`duration_budget.
  estimated_total_seconds`, summing every scene's own duration estimate) —
  informational, flags "⚠ off target" past a ±15% (min 5s) tolerance, never
  blocks. `--target-duration` follows the same inherit-across-`--resume`
  treatment `--max-scenes`/`--profile` already have (`cli.py`'s
  `_save_run_params`/`_load_run_params`, `output/run_params.json`) — an
  unfinished run resumed bare doesn't silently revert to the 45s default.
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
  **`--max-scenes`/`--profile` are inherited across `--resume`, not silently
  reset** (`cli.py`: `_save_run_params`/`_load_run_params`,
  `output/run_params.json`) — `main()` resolves an omitted `--max-scenes` via
  a distinct sentinel (`_MAX_SCENES_UNSET`, NOT the same as `_max_scenes_arg`'s
  own `None`, which is the meaningful explicit value "all") so it can tell
  "flag not given, inherit" apart from "explicitly asked for all"; an
  omitted `--profile` inherits the same way (no sentinel needed there — `None`
  already unambiguously means "not given", since there's no CLI way to
  explicitly request "no override" as a distinct value). An explicit flag on
  the resume command always overrides the inherited value AND updates the
  stored one, so it sticks for any later resume too. The printed "resume:"
  hint at a pause always echoes the actual effective `--max-scenes`/
  `--profile` used, so copy-pasting it reproduces the same run instead of
  silently reverting to argparse's own defaults (1 scene, no profile
  override) — the bug this fixes. `--genre` needed no equivalent treatment:
  it's already checkpointed to `output/genre.json` and reloaded on `--resume`
  by `pipeline.run()` itself (`genre_loaded = _checkpoint_load(out, "genre")
  if resume else None`), predating this fix. Missing/corrupt `run_params.json`
  (a pre-existing `--out` from before this existed, or one only ever touched
  by standalone `stage`/`revise` commands, which don't go through this path)
  degrades to the normal argparse defaults, never raises.
- **Session identity (`reel/session.py`):** one full story-to-video run (ingest
  through render) is a **session**, identified by a generated id
  (`<timestamp>-<random>`) persisted to `output/session.json`
  (`session_id`/`source`/`started_at`/`status`/`resumes`). `session.start(out,
  fresh=...)` mints a NEW id on a fresh (non-`--resume`) `pipeline.run()` call
  (`fresh=True`, overwriting any prior `session.json` — matches the existing
  fresh-run checkpoint-overwrite semantics), or **reattaches** to whatever
  session is already in `out` otherwise (`--resume`, and every standalone
  `stage`/`render` invocation) — so a multi-day run spanning several
  `--resume`s, or a mix of full-pipeline + standalone-stage invocations against
  the same `--out`, all stay one session rather than each minting a new id.
  `session.finish(out, status)` marks a terminal status
  (`complete`/`paused`/`failed`) — wired into `pipeline.run`'s successful
  return and `cli.main`'s `PipelineStopped`/`KeyboardInterrupt`/generic-
  exception handlers. `gemini.py`'s `_log_call` (the `gemini_api.log` writer)
  and `pipeline._write_scene_prompt_log` (the per-scene Veo prompt log) both
  best-effort-read the active session id via `session.current(out)` and tag
  every line/file with it, so logs stay attributable to the run that produced
  them even if `--out` is later reused for a different story. `session.py` has
  zero dependency on any other `reel` module (stdlib only), so it's imported
  freely from `gemini.py`/`pipeline.py`/`stages.py`/`cli.py` without any risk
  of an import cycle.
- **Revision agent (`python -m reel.cli revise`, `reel/agents/revision.py` +
  `reel/artifact_diff.py` + `reel/revision_merge.py`):** a standalone,
  post-hoc loop over a completed (or paused) run's `output/` checkpoints —
  separate from the live per-run HITL `Gate`, which only exists during a
  fresh `pipeline.run()`. Reattaches to the run's existing session
  (`session.start(out, fresh=False)`) and stays `"running"` across as many
  revision rounds as the operator wants; `session.finish` is only called on
  explicit `quit`/`pause`. Each round: pick any stage (or `"source"` for the
  raw ingested text) → hand-edit its JSON via `gate.edit_in_editor` (extracted
  to a module-level function so both the live gate and `revise` share it,
  `gate.py:_edit_in_editor` is now a 1-line delegate) → `artifact_diff.diff_artifact`
  deterministically finds which scene numbers / character-or-location names /
  panel numbers actually changed (no LLM — `diff_keyed_list` + a small
  `ARTIFACT_SHAPES` table keyed by each artifact's list field + key function,
  e.g. `scenes.json` by `number`, `casting.json` by `name`, `storyboard.json`
  nested by `(scene_number, panel)`). **v1 scope assumption: scene
  count/order is stable across a revision** — an added/removed scene number
  sets `diff.drastic=True`, and `structure`/`moodboard`/`source` (no
  scene/name-keyed structure at all) are always treated as drastic
  (`artifact_diff.WHOLE_FILE_ARTIFACTS`) — a drastic change falls back to a
  full, non-scoped downstream regen (with a confirm + warning), not a refusal.
  For a non-drastic edit: `stages.downstream_of(name)` (transitive closure
  over the `STAGES` registry's `inputs`/`optional` — the only place stage
  deps exist as data) gives the affected stage set; each downstream agent
  (`casting`/`characters`/`scenes`/`soundscape`/`visuals`/`cinematography`/
  `screenplay`/`storyboard` — all gained uniform `existing=`/`revise_keys=`
  kwargs, threaded through `stages.run_stage`) still gets FULL story context
  in its prompt (cross-scene/cast coherence needs it) but the caller only
  *trusts* the response for `revise_keys`; everything else is spliced back in
  byte-identical from `existing` via `revision_merge.merge_by_key` — the one
  shared primitive every agent uses, so untouched entries' text (and thus
  their `pipeline._content_hash`) never drifts from incidental LLM rewording.
  This is what makes casting **identity preservation** automatic: an untouched
  character/location's `visual_prompt` stays byte-identical → its
  `_render_casting_images` hash matches → its rendered PNG is reused with zero
  code changes needed in that function. A genuinely NEW name doesn't need to
  be in `revise_keys` at all (`merge_by_key` appends it automatically).
  `screenplay`/`storyboard` already loop one LLM call per scene, so scoping
  them just filters which scenes enter the loop (screenplay additionally
  seeds `_prior_scenes_block` continuity from `existing` for any
  untouched-but-preceding scene, so a later target still sees real prior
  context, not a blank slate). Casting-name-keyed vs. scene-number-keyed
  artifacts don't share a key type — `cli._translate_revise_keys` handles the
  one meaningful cross-type case (a `scenes` edit reaching `casting` via the
  revised scenes' `location` names) and falls back to a full non-scoped regen
  for any other stage pair with no known translation, rather than skipping it
  silently. **Drastic character/location identity changes are deliberately
  NOT auto-handled** (out of scope for v1, per explicit instruction): a
  deterministic (no LLM) heuristic, `revision.is_drastic_identity_change`
  (`difflib.SequenceMatcher` ratio over the input description that fed
  casting, threshold `config.revision.identity_drift_threshold`, default
  0.55), just warns and preserves the OLD casting entry/image by default. A
  `revision_agent.suggest_ripple_scenes` LLM call (open model, `models.text`,
  `agent_profiles.revision: quality`) is separately, narrowly scoped:
  advisory-only suggestions for OTHER scenes that might depend on what
  changed (a plot detail, a prop) — presented for accept/reject, never
  auto-applied. **Video-specific — panel-level re-render**: `_render_scene_frames`
  (`pipeline.py`) now records explicit `"start_frame"`/`"end_frame"` paths per
  panel in `output/video/manifest.json` (previously only the start/`"seed"`
  was recorded, though the end/tail PNG was always written to disk). New
  `only_scenes: set[int] | None` param targets specific `scene_number`s
  (vs. `max_scenes`'s leading-slice); a companion `existing_manifest` param is
  required to avoid a scoped call overwriting every OTHER scene's manifest
  record. The per-panel render/hash/overlay/tail-extraction logic was
  extracted into a shared `_render_one_panel` (used by both
  `_render_scene_frames`'s normal loop and the new `pipeline.rerender_panels`)
  and the per-scene ffmpeg concat into `_stitch_scene`, so the two call sites
  can't drift. `rerender_panels(scene_number, panel_numbers)` re-renders the
  targeted panel(s) **plus exactly ONE panel immediately after them**
  (reseeded from the new end_frame, to keep that one seam smooth), then
  **stops** — a deliberate, confirmed cost bound against Veo API spend, since
  the existing `_content_hash`/`_stale` chain would otherwise naturally
  cascade a re-render through the *entire rest* of the scene (every
  subsequent panel's seed derives from the previous panel's tail). Stopping
  the cascade correctly requires care: the panel *after* the one-hop panel
  must not be spuriously judged stale on some future run just because the
  one-hop panel's tail bytes genuinely changed underneath it — the fix is
  **not** to touch that panel's actual clip/tail files (the accepted
  one-hop-only visual discontinuity beyond that point is real and stays
  real), but to recompute and rewrite *only its `.hash` sidecar* to the value
  it would have if fully re-chained, so a later `_render_scene_frames`/
  `rerender_panels` call doesn't redundantly re-trigger it — verified via a
  4-panel stubbed scene (re-render panel 2 → panels 2 and 3 change, panel 4's
  clip bytes stay identical, and a follow-up full-scene render pass does NOT
  re-render panel 4). `cli._apply_scene_render_revision` picks
  `rerender_panels` (cheap, one-hop-bounded) specifically for a **storyboard
  panel-only** edit (identified via `artifact_diff.diff_nested`) and falls
  back to scene-level `_render_scene_frames(..., only_scenes=...)` for any
  other upstream stage reaching video (soundscape/visuals/cinematography/
  screenplay/scenes/casting), since those legitimately touch every panel's
  prompt in the scene, not just one. **Offered automatically, not just as a
  standalone command:** right after a full `python -m reel.cli story.txt`
  run completes (video render included), `cli._offer_revise` prompts once —
  type `revise` to drop straight into the same loop (`cli._revise_loop`,
  factored out of the standalone `revise` command so both share it) without
  a second invocation, or press Enter to exit. `pipeline.run()` already
  marks the session `"complete"` on its own successful return; choosing
  `revise` here reopens it (`session.start(fresh=False)` reattaches and
  flips status back to `"running"`) — declining leaves it exactly as
  `pipeline.run()` left it. Skipped entirely when config `hitl.enabled` is
  false (unattended/batch runs) or there's no TTY (`input()` raising
  `EOFError` is treated as "don't block", the same convention `gate.py`
  already uses) — a cron/CI run is never left waiting on this prompt. Inside
  the loop, `quit`/`exit` (session → `"complete"`) and `pause` (session →
  `"paused"`) are the two ways out, plus Ctrl-C (→ `"paused"`). The menu
  lists `source` once for the raw ingested text (not a redundant second
  `ingest` entry — `ingest` is the stage that *produces* `source.json`,
  `Stage.produces="source"`, so `_revise_one` routes `stage_name in
  ("source", "ingest")` to the same `_revise_source` handler, and the
  interactive menu skips listing `ingest` at all to avoid showing two
  entries for the same underlying artifact).
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
- **`--max-scenes` now ONLY restricts actual media-rendering stages** —
  casting-image generation and `scene_render`'s video generation. Every
  design/planning stage (screenplay, storyboard, soundscape, visuals,
  cinematography) always processes every scene in the story by default,
  regardless of `--max-scenes`. This was already true for soundscape/visuals/
  cinematography/storyboard (verified by inspection — none of the four take a
  `max_scenes` parameter at all); `screenplay` was the one inconsistency
  (`draft_screenplay`'s `max_scenes` defaulted to `3`, capping drafting to
  match the video-render cap) — fixed by changing its default to `None` (all
  scenes) and having `pipeline.run()` stop passing its render-scoped
  `max_scenes` into the screenplay call. Moodboard's `tiles` field remains
  intentionally capped to `max_scenes` — that's correctly scoped, since tiles
  get rendered to actual images (a real media-generation step), unlike the
  moodboard's film-wide aesthetic fields which are never scene-scoped at all.
  Standalone `stage screenplay --max-scenes N` (via `stages.py`'s `run_stage`,
  shared CLI default of 1 across all stage types) was deliberately left
  capped-by-default — a different use case (explicit single-stage testing)
  from the full pipeline run this fix targeted.
- **Status:** The pipeline runs **genre → structure/characters → moodboard →
  scenes/casting → soundscape/visuals/cinematography → screenplay → storyboard →
  render**, with a human-in-the-loop gate per stage plus per-stage **fidelity** and
  **genre** scoring. **Live-verified end-to-end on this host**, including a full
  video render + stitch to `output/video/movie.mp4`, using `--resume` across
  several sessions on the bundled sample story.
- **Veo prompts are now assembled via an explicit five-part dict, always in
  this fixed order** — `[Cinematography] + [Subject] + [Action] + [Context] +
  [Style & Ambiance]`, per Google's official Veo 3.1 prompting guide
  (cloud.google.com, confirmed via live fetch — a different, newer guide than
  the ai.google.dev vocabulary `veo_guide.py` syncs). Built from structured
  data (casting.json, scene `visual_overview`, panel camera fields), not the
  storyboard agent's free-text `image_prompt` (now a fallback only). See
  `pipeline._five_part_veo_prompt` and the 2026-07-08 session log entry for
  the full chain of fixes this came from (action-verb false positives,
  duplicate/contradictory focus terms, per-frame prompt + API-call logging).
  Depth-of-field (deep/shallow focus) is intentionally unmanaged — left to
  whatever the source content says, per explicit instruction. Verified
  against the live sample-story storyboard (8/8 panels, zero `verify_prompt`
  issues); not yet live-rendered end-to-end with this new assembly.
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
- **Veo prompts follow the official five-part formula** (per
  cloud.google.com's Veo 3.1 prompting guide — a different, newer source than
  the ai.google.dev API reference `veo_guide.py` syncs vocabulary from):
  `[Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]`,
  in that fixed order, ALWAYS. Built via an explicit dict in
  `pipeline._five_part_veo_prompt` (helpers `_panel_cinematography`/
  `_panel_subject`/`_panel_context`/`_panel_style_ambiance`) from **structured
  data** — casting.json's `character.physical_form` for Subject, the scene's
  cast location entry for Context, `visual_overview` for Style & Ambiance,
  the panel's own shot_type/camera_angle/camera_movement/lens for
  Cinematography — rather than the storyboard agent's free-text
  `image_prompt` (which has no guaranteed internal order, so reordering it
  after the fact isn't reliable; reconstructing from structured fields is).
  `image_prompt` is still generated by the storyboard agent and still used as
  a fallback when a caller has no `casting_lookup` (e.g. a bare `gen-video`
  prompt). Depth-of-field (deep/shallow focus) is intentionally NOT asserted
  anywhere in this assembly — left to whatever the source content says, if
  anything; only framing hints ("portrait" for close-ups, "macro lens" for
  inserts, from `_VEO_FOCUS`/`_VEO_FOCUS_FOUNTAIN`, kept identical between
  `pipeline.py` and `fountain.py`) are still asserted. Presence (not order) of
  Style/Camera/Focus-ish keywords is still checked before every Gemini API
  call via `veo_guide.verify_prompt`, issues/warnings logged, never blocking.
  Audio cues (ambient/SFX labels, voice-over/off-screen clarity, background-music
  directive, no-subtitles, cross-clip voice consistency) are **constructed**
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
- **Shelved (2026-07-09): a formal `tests/` end-to-end test suite** (stdlib
  `unittest`, stubbing every paid API — Gemini image/video — plus Ollama for
  hermeticity) was drafted (full `pipeline.run()` drive-through + a
  scene-alignment self-heal scenario) but explicitly shelved before being
  verified or committed; the two draft files were removed per instruction, no
  trace left. Revisit if/when a real test suite becomes a priority — this
  project's established practice until then remains throwaway stubbed
  scripts per session (see the session log entries above for worked
  examples), not a committed suite.
- **Next up:** confirm the still-running `storyboard` live test (bundle
  location/cast data, panel consistency with the rendered location image);
  Increment 5 — wire location + character references into Veo's
  `reference_images` for a scene's opening frame; live smoke-test the SDK
  video-call fixes on a real run; moodboard tile auto-render (opt-in); richer
  ingest (PDF/EPUB/.fdx); draft all scenes (not just first N); edit / sound mix
  / final cut phase.

## Session log
- 2026-07-09 (later 5) — **Added a target-total-runtime parameter (`--target-duration`,
  default 45s) threaded through scene/shot planning AND each rendered
  clip's actual requested duration.** User asked for this to influence
  "scenes/shots and all things that follow." First traced what actually
  determines a real rendered video's length today — found `i2v.generate_clip`
  had NO `duration_seconds` parameter at all, so every Veo call fell through
  to `gemini.generate_video`'s bare SDK default of 8s, unconditionally; the
  real lever was purely panel/shot COUNT. Rather than guess at Veo's valid
  duration values, verified them live via WebFetch against the official
  ai.google.dev Veo 3.1 guide: exactly **4, 6, or 8** seconds, not a
  continuous range, and MUST be 8 outside 720p resolution or with extend-mode
  continuity — this project's established practice of confirming external API
  constraints rather than guessing (see the extend-mode session log entry
  from 2026-07-08 for the same pattern). **Mid-implementation, user corrected
  the architecture twice**: (1) "this scene definition may be independent of
  VEO which is one of the supported video generation engines" — caught that
  the first draft of `duration_budget.py` baked Veo's specific 4/6/8
  constraint into what should be engine-agnostic scene/shot-count planning
  guidance; restructured so `duration_budget.py` has zero knowledge of any
  video backend (pure "how many scenes/shots fit a target runtime" math), and
  moved the Veo-specific rounding (`_veo_nearest_valid_duration`) into
  `i2v.py` itself, scoped to only `_gen_gemini`. (2) "the specific video
  generator may align to its own limitation... translated by its own
  adaptor" (clarified as "or agent" — kept this codebase's existing
  "backend" terminology rather than overloading "agent", which already means
  something specific here — the LLM-based `reel/agents/*.py` modules) —
  extended the same per-backend-adaptor treatment to `_gen_diffusers`/
  `_gen_http` (previously would have silently ignored a duration request
  entirely): both now honor a requested duration as a frame count via
  `_frames`, with no particular valid range asserted for either (unverified,
  unlike Veo's confirmed constraint) — honest rather than presumptuous about
  capabilities not actually confirmed. Final architecture: `duration_budget.
  suggest_scene_target`/`suggest_shots_per_scene` (engine-independent
  planning guidance, ~6s/shot assumption) feed `scenes.py`'s existing
  `target` param and a new `cinematography.py` `shots_guidance` param;
  `_render_one_panel` parses each storyboard panel's OWN already-computed
  duration estimate (`storyboard._estimate_duration`, deterministic per the
  entry below) back to seconds and passes it to `i2v.generate_clip`, which
  each backend's own adaptor translates however is appropriate for that
  engine. Also added to `_content_hash` (a duration-only change from a
  revision must invalidate a stale clip) and a target-vs-estimated readout
  at the storyboard gate (`_summarize_storyboard`, flags "⚠ off target"
  past ±15%/min 5s tolerance, informational only). `--target-duration`
  follows the same inherit-across-`--resume` treatment `--max-scenes`/
  `--profile` already got in an earlier session (`_save_run_params`/
  `_load_run_params`). Verified extensively offline: `_veo_nearest_valid_duration`
  rounding + force-8 cases; `_frames`' generic frame-count adaptor;
  `duration_budget` confirmed to have zero Veo-specific symbols; a full
  stubbed end-to-end `pipeline.run()` confirming the target actually reaches
  the scenes/cinematography prompts AND that each panel's own estimated
  duration reaches `i2v.generate_clip`; `_gen_gemini`'s real rounding/
  force-8 logic exercised directly (5s→4s, 7s→6s, non-720p→forced 8s,
  no-request→unchanged default of 8s); the existing `rerender_panels`
  one-hop-cascade regression test re-run clean (needed only a scratchpad
  stub-signature update, not a real code fix). *(Uncommitted at time of
  writing.)*
- 2026-07-09 (later 4) — **Fidelity agent gained a deterministic, self-healing
  scene-structure alignment check, wired to automatically reiterate any
  scene-keyed stage that drifts from scenes.json — before the operator ever
  sees the gate — and this same work caught a real pre-existing bug in
  screenplay.py.** User asked to "make sure the fidelity agent make sure that
  the alignment to scenes are maintained in every stage and reiterate as
  needed... with all the inputs coming in in every stage." Distinguished this
  from what `fidelity.check_stage` already does (a qualitative LLM judgment —
  does this stage still tell the same STORY — only ever advisory at the gate)
  — scene alignment is a different, objective question: does a stage's own
  scene-keyed data structurally match scenes.json, which is literally one of
  its declared `stages.py` inputs. Being objective (not a judgment call), it's
  deterministic (no LLM) and self-heals automatically rather than just
  advising. New `fidelity.check_scene_alignment`/`strip_orphan_scenes`, reused
  the scene/name-keying knowledge already centralized in `artifact_diff.py`
  (added `SCENE_KEYED_ARTIFACTS`/`NAME_KEYED_ARTIFACTS` there as the one
  shared definition — `cli.py`'s `_SCENE_KEYED_STAGES`/`_NAME_KEYED_STAGES`
  now alias them instead of duplicating). Wired into `pipeline.run()`'s
  `run_group`: `_spec()` gained an optional `realign(result, keys)` callable
  (added to soundscape/visuals/cinematography/screenplay/storyboard's specs —
  each just calls the same agent with `existing=`/`revise_keys=`, the exact
  mechanism the `revise` CLI command already uses); right after a stage
  computes, orphan scenes (stale leftovers `revise_keys`-scoped calls can
  never delete, only add/replace — see `revision_merge.merge_by_key`) are
  stripped unconditionally, then any still-missing scenes trigger ONE
  reiteration attempt before the gate is shown at all. `_gated` gained an
  `alignment_rep` param (a snapshot, unlike fidelity/genre which re-check
  every gate loop iteration) and a new `_format_alignment` readout — only
  ever prints when the one reiteration attempt still left a gap. **Building
  the verification for this surfaced a real, independent bug**: a full
  stubbed end-to-end `pipeline.run()` test (not just isolated unit tests —
  deliberately chosen so the actual `run_group` wiring got exercised, not a
  reimplementation of it) correctly self-healed a simulated soundscape
  omission and stripped a simulated stale orphan scene exactly as designed,
  but a crude screenplay stub (always echoing `scene_number: 1`) kept failing
  alignment even after reiteration — traced to `screenplay.py`'s
  `scene_doc.setdefault("scene_number", scene_num)`, which only fills in an
  ABSENT key, so a wrong or stale scene_number from the model was trusted
  as-is; `"number"` right next to it was already correctly force-set, just
  not `"scene_number"` (the field `artifact_diff`/fidelity's alignment check
  actually keys off) — same class of bug `storyboard.py`'s `plan_storyboard`
  already had fixed for the same reason. Fixed identically (force-set, don't
  setdefault — each call is sent exactly one scene, so the correct
  scene_number is always known, no reason to trust the model's echo).
  Re-verified the full end-to-end test after the fix: soundscape self-healed
  in exactly 2 LLM calls (initial + 1 reiteration), screenplay now aligns on
  the first attempt with no reiteration needed at all, storyboard shows no
  alignment warning (already deterministic-by-default from the entry above,
  so it's aligned by construction). Existing screenplay.py scoped-revision
  regression test re-run clean after the fix. *(Uncommitted at time of
  writing, on top of the three uncommitted entries below from the same
  session.)*
- 2026-07-09 (later 3) — **Storyboard construction made fully deterministic
  (no LLM call) in the default path.** Direct follow-up question after the
  merge-gap audit above: "does storyboard aggregation require an LLM ... this
  could be done based on the input artifacts locally scene by scene ... would
  this give better results." Worked through it field by field against what
  `_scene_bundles` already assembles: `header`/`visual_overview`/
  `audio_overview` were already near-1:1 copies from the bundle (exactly what
  the just-added "MERGE SOURCE" prompt block had been asking the LLM to do
  faithfully rather than reinvent); most panel fields (`shot_type`/
  `camera_angle`/`camera_movement`/`lens`/`dialogue`/`sound`/`transition`) had
  a direct source too. Two fields looked LLM-only at first — `composition`
  and shot alignment — but turned out not to be: cinematography.py's own
  per-shot schema already includes a `framing` field ("specific compositional
  note"), making storyboard's `composition` field redundant with data that
  already existed; and screenplay.py's prompt already instructs it to derive
  its own shots from the camera coverage, so `camera.shots[]` and
  `screenplay_shots[]` should already share the same order in the common
  case, needing only a fallback pairing (by `shot_number`, else positional)
  for when they don't — not real LLM judgment. This went further than what
  was proposed and agreed just beforehand ("keep a lightweight LLM call for
  shot alignment + composition") — flagged the deviation transparently rather
  than silently doing more than asked, then proceeded given it's a strict
  quality improvement in the same already-approved direction. Implemented:
  `_align_shots` (shot_number match, positional fallback), `_build_panel`/
  `_build_scene_board` (deterministic construction of every field),
  `_panel_characters_in_frame` (the one genuinely INVENTED field — no
  artifact says who's visually in a shot — a documented heuristic: dialogue
  speaker for close-shot-family panels, full scene cast otherwise),
  `_estimate_duration`/`_format_total_duration` (the other invented field —
  no artifact gives per-shot screen time — shot-type base duration extended
  to cover dialogue at ~2.5 words/sec), `_deterministic_storyboard_style`
  (prefers visuals'/cinematography's own top-level style fields, then
  moodboard, then genre+tone). **`feedback` is the one path that still calls
  the LLM** (`_llm_generate_scene`, the prior full generation prompt) — a
  directed creative note needs actual judgment the deterministic builder
  can't provide; `plan_storyboard` branches on whether `feedback` was given.
  The scene_number-correction/`dropped_scenes` tracking added in the entry
  above is now specific to that LLM-feedback path (structurally impossible in
  the deterministic path — there's no model response to misbehave). Verified
  with a full realistic scene test (2 characters, dialogue + voiceover, a
  wide shot and a close-up) confirming every design decision landed correctly
  in one pass: the WS panel defaulted to the full scene cast,
  `characters_in_frame` on the CU panel correctly narrowed to just the
  speaker, the voiceover line appeared alongside regular dialogue with
  `vo: true`, the last panel's transition used cinematography's
  `transition_to_next` while the earlier panel got the "CUT TO" default,
  sound correctly combined ambient bed + screenplay's own SFX field, and
  `storyboard_style` combined visuals' and cinematography's own top-level
  fields; plus explicit tests confirming zero `llm.generate` calls in the
  no-feedback path, the LLM fallback still fires correctly when `feedback` IS
  given (with scene_number correction and dropped-scene tracking both intact),
  and scoped `revise_keys` revision with no feedback rebuilds only the
  targeted scene deterministically while leaving other scenes byte-identical.
  `pipeline._summarize_storyboard` confirmed to render the new output shape
  without changes. Net effect: storyboard — previously one `synthesis`-tier
  (the slowest/most expensive profile) LLM call per scene — now costs zero
  LLM calls for a fresh run or most revisions, is immune to the whole class
  of drift/omission bugs the merge-gap audit above was busy patching, and
  only pays for an LLM call when there's a genuine creative note to apply.
  *(Uncommitted at time of writing, on top of the two uncommitted entries
  below from the same session.)*
- 2026-07-09 (later 2) — **Storyboard merge hardened: scene-list structure now
  enforced deterministically (not just architecturally correct), and a
  field-by-field prompt audit closed real merge gaps, including one that
  could silently drop voice-over lines.** User asked to "make sure" the
  storyboard merges all completed stages scene-by-scene while adhering to
  the scene structure `scenes.py` established — tracing `_scene_bundles`
  confirmed the ARCHITECTURE was already correct (it iterates
  `scenes.get("scenes", [])` and builds one bundle per scene via
  `scene_number` lookups into every other artifact, so scenes.json's own
  order/count is the natural backbone), but found a real gap: nothing
  verified the model's per-scene response actually respected that structure
  — a wrong/omitted `scene_number` or an empty response for one call had no
  correction or visibility, unlike `scenes.py`'s own `dropped_scenes`
  handling. Fixed in `plan_storyboard`: each call is known to correspond to
  exactly one scene (its bundle), so a returned `scene_number` mismatch is
  corrected to the one actually sent (no ambiguity to resolve — deterministic
  bookkeeping, not a guess) and an empty response is recorded in a new
  `dropped_scenes` list rather than silently vanishing from the board;
  `pipeline._summarize_storyboard` surfaces it at the gate, mirroring
  `_summarize_scenes`'s existing pattern. Verified via a stubbed test
  covering all three failure modes in one run (wrong scene_number corrected,
  an empty-response scene dropped-and-recorded, order preserved) plus the
  existing scoped-revision regression test (unaffected). **User then asked a
  follow-up mid-fix**: whether the PROMPT ITSELF has gaps in how it uses the
  merged bundle data, "it may as well employ scene by scene formation if
  needed" (confirmed: it already does, this follow-up was about prompt
  CONTENT, not architecture). Did a field-by-field audit — every field
  `_scene_bundles` puts into a bundle, cross-checked against whether the
  PROMPT's rules or schema actually instruct the model to use it — and found
  seven real gaps where bundle data was sent but never actually grounded to
  an output field: `art.visual_filter` had no output field at all (added one
  to `visual_overview`'s schema); `art.key_props`/`art.visual_moments`
  (beat-keyed) were never referenced; `audio.silence` was never referenced
  (risk: a deliberately silent scene still gets an invented ambient bed);
  `audio.sound_events` was never mapped to `audio_overview.key_sounds`;
  `camera.transition_to_next` (already decided by cinematography) was never
  cited as authoritative for a panel's own `transition`; and — the most
  consequential, since it's a genuine data-loss risk, not just an
  under-specification — `screenplay_shots[].voiceover` is a field SEPARATE
  from `.dialogue` that the existing "dialogue is locked, copy verbatim"
  rule never mentioned at all, meaning a screenplay shot's voice-over line
  had no instruction path into the storyboard whatsoever. Fixed by adding an
  explicit "MERGE SOURCE" block to the prompt mapping every one of these
  bundle fields to its exact output field (plus a `visual_filter` schema
  field addition and a strengthened dialogue rule explicitly calling out
  voiceover as separate from dialogue). Verified via `.format()` smoke test
  (no stray placeholders from the new block's literal JSON-brace examples)
  and re-ran the existing scoped-revision regression test clean. *(Both
  fixes in this entry uncommitted at time of writing.)*
- 2026-07-09 (later) — **`--max-scenes`/`--profile` now survive `--resume`
  instead of silently resetting to their defaults.** User asked a direct
  diagnostic question about the printed resume hint (`python -m reel.cli
  samples/sample_story.txt --out output --resume`, missing `--max-scenes=all`
  from the original run) — traced it to a real gap: `main()` never persisted
  `args.max_scenes`/`args.profile` anywhere; each invocation was fully
  independent, so the printed hint (and `--resume` alone in general) would
  silently fall back to argparse's defaults (1 scene, no profile override)
  regardless of what the original run actually used — a correctness bug, not
  just a UX one, since any remaining unfinished casting-image/video rendering
  would only cover scene 1 instead of the full story. Confirmed `--genre` was
  the ONE exception already handled correctly (checkpointed to
  `output/genre.json`, reloaded via `_checkpoint_load` when `resume=True`).
  Fixed by adding the same treatment for the other two: `cli.py` gained
  `_save_run_params`/`_load_run_params` (`output/run_params.json`, holding
  the EFFECTIVE — already-inheritance-resolved — `max_scenes`/`profile`/
  `genre` a run used, re-written every invocation so it stays current across
  any number of resumes) and a `_MAX_SCENES_UNSET` sentinel distinct from
  `_max_scenes_arg`'s own `None` (the meaningful, explicit value "all") so
  `main()` can tell "flag omitted, inherit" apart from "explicitly asked for
  every scene" — `--profile` needed no equivalent sentinel since `None`
  already unambiguously means "not given" for it (no CLI way exists to
  explicitly request "no override" as a distinct value). An explicit flag on
  the resume command always overrides the inherited value and becomes the
  new stored default for any LATER resume too. The printed "resume:" hint
  now always echoes the actual effective values used, so copy-pasting it
  reproduces the same run instead of silently downgrading it. Verified via
  stubbed tests (`cli.run` mocked to capture what it was actually called
  with, no live Ollama/Gemini): fresh run with `--max-scenes all` persists
  `null`; a bare `--resume` correctly inherits `all`; an explicit
  `--max-scenes 3` on a later resume overrides AND updates the stored value;
  a SUBSEQUENT bare resume after that now inherits `3`, not the original
  `all`; `--profile` inheritance works the same way; and a `--resume` against
  a directory with no `run_params.json` at all (simulating a pre-existing
  `--out` from before this fix, or one only ever touched by standalone
  `stage`/`revise` commands) falls back to the normal defaults without
  crashing. Docs updated (README's Pause & resume section, this file).
  *(Uncommitted at time of writing.)*
- 2026-07-09 — **Fixed a real character-name mismatch between `scenes` and
  `characters` (found by the user inspecting live output, not by test
  coverage): the `scenes` agent could independently re-derive a different
  label for the same person than the `characters` agent already settled on,
  silently breaking every later name-keyed lookup.** Concretely: the sample
  story only ever describes one character in prose ("a beautiful young
  woman"), never a proper name — `characters.py` named her "Young Woman" in
  `characters.json`, but `scenes.py` (run independently, with no shared
  naming anchor) called her "Woman" in every scene's `characters` list.
  `pipeline.run()`'s casting-image active-names filter (`pipeline.py:1635-
  1640`) builds its allow-list from `scenes[...].characters` and does a
  literal string match against `casting.json` — since `"Young Woman" !=
  "Woman"`, her image was silently skipped in every run, independent of
  `--max-scenes` (the name mismatch was present in every scene, not just
  ones beyond the cap — raising `--max-scenes` alone would not have fixed
  it). This is the exact same class of problem `location` already had a
  fix for (scenes.py locks location naming to one consistent string,
  casting matches against it) — character names never got the same
  treatment, since `characters` and `scenes` run concurrently in the
  pipeline with no cross-reference. Fixed in `reel/agents/scenes.py` with
  a NEW optional `characters` parameter on `segment_scenes` (pipeline.py's
  "2/10 structure ‖ characters" stage already completes before "3/10
  scenes" runs, so passing it through is a pure plumbing change, no
  reordering needed) — two layers, deliberately paired rather than relying
  on either alone (this codebase has hit prompt-instruction-alone drift
  before, e.g. the storyboard focus-term bias from an earlier session):
  (1) `_canonical_names_block` feeds characters.json's settled names into
  the prompt as a CANONICAL CHARACTER NAMES list with an explicit
  do-not-shorten/rephrase instruction; (2) `_reconcile_character_names` is
  a deterministic (no LLM) post-processing safety net — if a scene names
  someone with a string that isn't an exact canonical name but whose words
  are a strict subset of exactly ONE canonical name's words (e.g. "Woman" ⊂
  "Young Woman"), it's rewritten to the canonical form; genuinely ambiguous
  (matches more than one canonical name) or unrelated names are left
  untouched rather than guessed. `stages.py`'s "scenes" `Stage` gained
  `characters` as an `optional` input (also means `stages.downstream_of
  ("characters")` now correctly includes "scenes" transitively — verified);
  `pipeline.py`'s two `segment_scenes(...)` call sites (initial + feedback
  rerun) now pass `characters=characters`. Verified via stubbed offline
  tests (canonical-names-block-present-in-prompt, reconciliation fixing the
  exact "Woman"→"Young Woman" case, ambiguous-match/no-match/no-characters-
  available all correctly left untouched, full `run_stage("scenes")`
  round-trip) — **then applied live against this repo's own `output/`** at
  the user's request: found (and had the user stop first) a stale live
  `reel.cli` process still holding that `output/` dir from a prior session
  before touching anything, backed up the pre-fix `scenes.json` to
  `scenes.json.bak-before-name-fix`, re-ran `stage scenes` (local Ollama,
  free — regenerated scenes.json, confirmed "Young Woman" now used
  correctly in scenes 4-5) and `stage casting_images` (Gemini — confirmed
  `output/casting/young_woman.png` now exists and `casting.json`'s
  `image_path` is set). *(Code fix uncommitted at time of writing — the
  live output/ regeneration is gitignored working-directory state, not
  something to commit.)*
- 2026-07-08 (later 4) — **Offer the revision loop automatically right after a
  full pipeline run completes, plus an explicit 'exit' option.** User asked
  for two additions to the just-built revision agent: (1) once the video
  render finishes, offer to go straight into revise mode "apart from the
  newly added command line option" (i.e. *in addition to*, not instead of,
  the standalone `revise` command), and (2) a clear way to exit. Refactored
  the standalone `revise` command's inner loop out into `cli._revise_loop`
  (shared) so a new `cli._offer_revise(out)` — called from `main()` right
  after a successful `run(...)` return, before the final `return 0` — can
  drop into the exact same loop without a second process invocation. Design
  point worth noting: `pipeline.run()` already marks the session
  `"complete"` on its own successful return (from the earlier session-id
  work), so choosing to revise at the offer prompt has to explicitly REOPEN
  it (`session.start(out, fresh=False)` — reattach, flip status back to
  `"running"`) rather than just calling `_revise_loop` directly; declining
  correctly leaves the session exactly as `pipeline.run()` already left it,
  with no extra session.json write. The offer itself is gated on config
  `hitl.enabled` (skipped for unattended/batch runs — matches the same
  bool that already gates the live per-stage gate) and wrapped in an
  `EOFError` catch (non-interactive/no-TTY runs, e.g. cron, never block on
  it — the same convention `gate.py`'s docstring already documents for its
  own `input()` calls). For the "exit" request specifically: `_revise_loop`
  already had `quit`/`pause`; added `exit`/`e` as an explicit synonym for
  `quit` (both → session `"complete"`) since the user used that exact word,
  and updated the loop's own prompt text to advertise it. **Found and fixed
  a real latent bug while writing the test for this** (not something the
  user asked about, but surfaced by exercising the menu): the stage-picker
  menu listed both `source` (the special raw-text entry) AND `ingest` (a
  normal `STAGES` entry) as separate choices, but `ingest`'s artifact IS
  `source.json` (`Stage("ingest", ..., produces="source")`) — selecting
  `ingest` from the menu would have gone through the GENERIC per-stage edit
  path instead of `_revise_source`'s special handling (which recomputes
  `chunks`/`word_count`/`char_count` after a hand-edit, since those are
  derived from `text` and go stale otherwise). Fixed two ways: `_revise_one`
  now routes `stage_name in ("source", "ingest")` to `_revise_source`
  identically, AND the menu skips listing `ingest` at all (redundant with
  `source`, which is already correctly handled) rather than showing two
  entries for one artifact. Verified via stubbed tests (no live Ollama/Gemini
  calls): `_offer_revise` skipped under `hitl.enabled=false`; EOFError caught
  gracefully; blank Enter exits without entering the loop; typing `revise`
  enters the loop AND reopens the session to `"running"`; `_revise_loop`'s
  `exit`/`pause`/Ctrl-C paths all set the correct terminal session status;
  `_revise_one("ingest", ...)` correctly routes to `_revise_source`, recomputes
  chunks, and triggers the full-regen downstream cascade (confirmed via a
  monkeypatched `stages.run_stage` recorder — real Ollama calls deliberately
  avoided; an earlier version of this same test without the mock was caught
  hanging against a live Ollama call mid-session and killed via `TaskStop`
  before it could run unbounded). *(Uncommitted at time of writing — builds on
  the still-uncommitted revision-agent work from the entry above.)*
- 2026-07-08 (later 3) — **Added a revision agent (`python -m reel.cli revise`):
  go back to any completed stage (including hand-editing the raw ingested
  story text), preserve character/location visual identity by default,
  selectively re-run only what's actually affected downstream, and re-render
  specific video panels with a bounded one-hop cascade.** A large, explicitly
  planned feature (used `EnterPlanMode` + two background Explore agents +
  one Plan agent given the size — new modules, signature changes across 8
  agent files, a `pipeline.py` render-path refactor, and a new interactive
  CLI command). Confirmed with the user up front: edits are hand-made via
  `$EDITOR` (reusing `gate.py`'s existing JSON view/edit mechanism, not a
  new freeform-feedback UX); downstream regeneration uses each stage's
  already-configured model profile (no special "revision model"); panel
  re-render cascades **exactly one hop forward** then stops (bounded Veo
  cost) — three explicit design decisions from the user before any code was
  written. New: `reel/artifact_diff.py` (deterministic keyed-list diff, no
  LLM — `ARTIFACT_SHAPES` table + `diff_keyed_list`/`diff_artifact`/
  `diff_nested`/`diff_source_text`; a scene-count/reorder change is
  `drastic`, since v1 assumes scene numbering stays stable across a
  revision — a documented, deliberate scope limit, not an oversight — and
  falls back to a full non-scoped regen with a warning+confirm rather than a
  silent partial revision or an outright refusal); `reel/revision_merge.py`
  (`merge_by_key` — the one shared splice primitive every scoped agent call
  uses: full-story LLM context in, but only `revise_keys`' output is
  trusted, everything else spliced back byte-identical from `existing`);
  `reel/agents/revision.py` (`suggest_ripple_scenes` — an LLM call, open
  model only, advisory-only "might this other scene also need a look"
  suggestions, never auto-applied; `is_drastic_identity_change` — a
  **deterministic** `difflib` heuristic, not an LLM call, for the same
  "cheap reproducible fingerprint" reason `pipeline._content_hash` is
  deterministic — flags when an edited character/location description has
  drifted enough that the OLD casting image should be preserved by default
  rather than silently replaced). `gate.py`'s `_edit_in_editor` extracted to
  a module-level `edit_in_editor` (zero `self` usage, confirmed by reading
  it — trivial, safe extraction) so `revise` reuses the exact same
  proven $EDITOR mechanics as the live HITL gate, verified via a scripted
  fake-`$EDITOR` shell script (one that edits the temp file, one that
  doesn't). Eight agents (`casting`/`scenes`/`soundscape`/`visuals`/
  `cinematography`/`screenplay`/`storyboard`/`characters`) gained uniform
  `existing=`/`revise_keys=` kwargs — each stub-tested individually
  (monkeypatched `llm.generate` returning a canned full response, asserting
  the caller only trusts the targeted keys and preserves the rest
  byte-identical, confirming e.g. a casting revision that only asked for
  "Alice" doesn't let an LLM's incidental rewording of "Bob" flip Bob's
  `_content_hash` and trigger an unwanted image re-render). `screenplay`/
  `storyboard` needed different handling from the other four since they
  already loop one LLM call per scene (confirmed via reading both files) —
  scoping them filters which scenes even enter the loop (cheaper than
  calling-then-discarding), and `screenplay` specifically needed its
  `_prior_scenes_block` continuity context seeded from `existing` for any
  untouched scene before a target, verified by asserting a scoped rerun's
  prompt actually contains an unchanged prior scene's marker text.
  `stages.downstream_of` (new — BFS over the `STAGES` registry's `inputs`/
  `optional` reverse-adjacency, the only place stage dependencies exist as
  data) computes the affected stage set from any edited stage; verified
  against the real registry (e.g. `downstream_of("storyboard")` ==
  `{scene_render, fidelity}` exactly). `pipeline.py`'s `_render_scene_frames`
  gained explicit `"start_frame"`/`"end_frame"` manifest fields (the tail
  PNG was always written to disk but never recorded — a one-line gap) and an
  `only_scenes`/`existing_manifest` pair (targets specific scene numbers
  without a scoped call silently wiping every OTHER scene's manifest
  record — caught via a 3-scene stub test before it shipped). The per-panel
  render/hash/overlay/tail-extraction logic was extracted into a shared
  `_render_one_panel` (used by both the normal loop and the new
  `rerender_panels`) and per-scene stitching into `_stitch_scene`, so a bug
  fix to either can't drift between two call sites. **`rerender_panels`
  was the highest-risk piece and caught two real bugs during its own
  stub-test build** (a 4-panel fake scene, monkeypatched `i2v.*`, byte-level
  clip comparison before/after): (1) initially threaded `prev_clip_path`
  from "what this call itself just rendered" rather than "the immediately
  preceding panel's CURRENT clip regardless of whether it was touched this
  call" — produced a `_content_hash` that disagreed with what a later full
  `_render_scene_frames` pass would compute for the same panel, so a
  follow-up run spuriously re-rendered panels that should have stayed put;
  fixed with a `_resolve_prev_clip_path` helper mirroring the existing
  `_resolve_start_frame` one. (2) The one-hop cascade-stop mechanism: a
  first instinct to "delete the `.hash` sidecar to force a re-render" is
  actually backwards — confirmed by reading `_stale()` directly, a *missing*
  hash file is treated as a pre-tracking baseline (NOT stale), the opposite
  of forcing; the correct, verified mechanism is to leave the
  one-hop-boundary panel's own clip/tail files completely untouched (the
  accepted visual discontinuity there is real and should stay real) but
  **recompute and rewrite its `.hash` sidecar** to the value it would have
  if fully re-chained — forward-looking bookkeeping that prevents future
  redundant re-renders without forging what was actually rendered. Verified
  end-to-end: re-render panel 2 of a 4-panel scene → panels 2 and 3 (one
  hop) get new content, panel 4 stays byte-identical, AND a follow-up full
  `_render_scene_frames` call does not re-render panel 4 (the actual
  correctness property, not just "the hash file changed"). New
  `python -m reel.cli revise [--out DIR]` interactive loop
  (`reel/cli.py`): reattaches to the run's existing session (`session.start
  (fresh=False)` — stays `"running"` across every round, only
  `session.finish`d on explicit quit/pause, per the third confirmed design
  decision), menu of stages + `source`, edit → `artifact_diff` → (drastic?
  confirm full regen : identity-drift warnings for casting + LLM ripple
  suggestions for scene-keyed edits, accept/reject) → confirm → save +
  selectively `run_stage` each downstream stage. Cross-key-type scoping
  (`_translate_revise_keys`) handles the two real cases: same key type
  propagates as-is; a `scenes` edit reaching `casting` translates via the
  revised scenes' `location` names (the only way a scenes edit could affect
  casting — it can't touch characters at all); any other combination falls
  back to a full non-scoped regen for that one stage rather than being
  silently skipped. A separate `_names_to_scene_numbers` handles the inverse
  translation (a `casting`/`characters` edit reaching `scene_render`) by
  scanning storyboard panels' `characters_in_frame` (falling back to
  scenes.json's `characters` field) for the scenes a revised name actually
  appears in — added specifically so a locked-identity change (which feeds
  every panel's Veo Subject) re-renders the scenes it actually affects
  instead of none at all. `_apply_scene_render_revision` dispatches to the
  cheap one-hop `rerender_panels` specifically for a storyboard **panel**-only
  edit (via `diff_nested`) and falls back to scene-level
  `_render_scene_frames(..., only_scenes=...)` for anything else reaching
  video, since an upstream soundscape/visuals/cinematography/screenplay/
  scenes/casting change legitimately touches every panel's prompt in the
  scene, not just one. Verified via three orchestration tests with
  `stages.run_stage`/`cli._apply_scene_render_revision` monkeypatched as
  call-recorders against real on-disk checkpoints (no LLM/API stubs needed
  at this layer): a casting-only edit correctly reached
  `{casting_images, fidelity, screenplay, storyboard}` with
  screenplay/storyboard falling back to full regen (no casting→scene-keyed
  translation exists) while `scene_render` correctly scoped to exactly the
  two scenes Alice appears in via `_names_to_scene_numbers`; a drastic
  scene-count-change correctly triggered a full `revise_keys=None` regen for
  every downstream stage; a storyboard single-panel edit correctly resolved
  to `panel_targets={1: [2]}` and dispatched to `scene_render` alone. A final
  fully-wired end-to-end test (real `_apply_scene_render_revision`, stubbed
  `i2v`/`llm`) drove `cli._revise_one` through to `rerender_panels` and
  confirmed the same one-hop-cascade byte-level result as the isolated
  pipeline test. `config/models.yaml` gained `agent_profiles.revision:
  quality` and a `revision: {identity_drift_threshold: 0.55}` block (an
  explicitly unvalidated starting heuristic, tunable without a code change).
  Deferred, documented rather than silently missing: source-text edits get
  no fine-grained diff (word-level diffing raw prose into "which scene's
  source_line moved" was judged genuinely hard, disproportionate for v1);
  cinematography's nested `shots` aren't independently merge-spliced (no
  consumer needs that granularity); storyboard *text* revision is
  scene-granular only (panel-granularity is exclusively the visual
  `rerender_panels` path). All work verified via stubbed offline tests only
  (no live Ollama/Gemini calls) per this project's established practice —
  a live smoke test against the bundled sample story is the natural next
  session's first step. *(Uncommitted at time of writing.)*
- 2026-07-08 (later 2) — **Added a session-identity concept (`reel/session.py`)
  spanning the whole story-to-video workflow.** User asked to treat "the whole
  workflow of providing story to producing the video" as one session, then
  clarified via AskUserQuestion they wanted an actual session id/concept added
  to the pipeline (not just doc framing). Until now every artifact under
  `output/` was scoped only by the `--out` directory itself — fine for a single
  run, but `gemini_api.log` and the per-scene Veo prompt logs had no way to
  tell which lines came from which invocation if the same `--out` was reused
  (resumed days apart, or mixed with standalone `stage`/`render` calls).
  New `session.py`: `start(out, source=, fresh=)` mints a new id
  (`<UTC timestamp>-<random hex>`) on a fresh `pipeline.run()` (overwriting any
  prior `session.json`, matching existing fresh-run checkpoint-overwrite
  semantics) or reattaches to the existing session otherwise; `finish(out,
  status)` marks a terminal status; `current(out)` is a best-effort read used
  purely for log tagging. Zero dependency on other `reel` modules by design, so
  it can be imported from `gemini.py` without any import-cycle risk. Wired in:
  `pipeline.run()` calls `start(..., fresh=not resume)` right after creating
  `out/`, logs the id, stores it in `project.json`, and calls `finish(out,
  "complete")` just before returning; `stages.run_stage()` and `cli.py`'s
  standalone `render` command both call `start(..., fresh=False)` so they
  reattach to whatever session a prior full-pipeline run (or another standalone
  stage) already established in that `--out`; `cli.main()` calls `finish(...,
  "paused")` on `PipelineStopped`/`KeyboardInterrupt` and `finish(...,
  "failed")` on any other exception (re-raised afterward, so tracebacks are
  unaffected). `gemini._log_call` now reads the active session id via
  `session.current()` and prepends `session=<id>` to every
  `gemini_api.log` line; `pipeline._write_scene_prompt_log` adds a `Session:
  <id>` header line to every `scene_NN_veo_prompts.txt`. Verified via stubbed
  offline tests (no real API calls): fresh-mints-new-id vs.
  reattach-keeps-same-id vs. a second fresh run minting yet another new id;
  gemini log-line tagging; scene-prompt-log header tagging;
  `cli.main`'s `PipelineStopped` path correctly writing `status: "paused"` to
  `session.json` via a monkeypatched `run()`. `README.md` gained a "Session
  tracking" section under Pause & resume. *(Committed and pushed.)*
- 2026-07-08 (later) — **Veo extend-mode made as robust as seed mode;
  `--max-scenes` consistency fixed so it only restricts actual rendering.**
  Two independent investigations, both triggered by direct questions rather
  than a bug report. (1) User asked whether `continuity_mode: extend` (Veo's
  native video-to-video scene extension, carries ambient/music audio across a
  cut, added a prior session but never enabled by default) was safe to use —
  traced it fully wired end-to-end (`gemini.extend_video` → `i2v._gen_gemini`
  → `pipeline._render_scene_frames`'s `prev_clip_path` threading, confirmed
  correct even across the resume-skip branch) but found real robustness gaps
  versus `generate_video()`: no retry on transient Veo errors (8/13/14) — a
  one-shot attempt meant any routine transient hiccup permanently downgraded
  that frame to seed continuity instead of retrying the extend call itself —
  and no `.error` field inspection at all, so even a caller wanting to retry
  couldn't tell a transient failure from a hard one. Also live-verified via
  WebFetch against the current official ai.google.dev Veo guide that
  `config/models.yaml`'s comment was stale: it claimed only non-"fast"
  veo-3.1-*-preview models supported extend, but both `-generate-preview` and
  `-fast-generate-preview` do (only `-lite-` doesn't) — `gemini.py`'s own
  extend_video docstring already had this right, only the config comment had
  lagged. Fixed: refactored extend_video's one-shot call into
  `_extend_video_once`, wrapped with the same retry-with-backoff +
  `.veo_code` extraction + per-attempt API-log line `generate_video()` has.
  First implemented the 720p constraint (extend is fixed at 720p regardless
  of the general video config) as a silent clamp inside extend_video() —
  **corrected per explicit direction**: coercing the configured OUTPUT
  RESOLUTION for a clip is worse than choosing a different CONTINUITY
  MECHANISM for it. Moved the check to `i2v._gen_gemini()`, which now skips
  attempting extend entirely (zero wasted API calls) and goes straight to
  seed continuity when resolution isn't 720p, rather than clamping or trying
  a call guaranteed to fail/misbehave for it. Verified via stubbed offline
  tests: skip-vs-attempt branching by resolution, transient-retry-then-
  success, non-transient failure raising without wasted retries, `.error`
  parsing against a realistic SDK operation object. *(Committed as 7f7a13b,
  pushed.)*

  (2) User asked whether `--max-scenes N` caps every stage through N —
  traced it stage by stage and found it genuinely doesn't, but the actual
  shape of the inconsistency mattered: `soundscape`/`visuals`/
  `cinematography`/`storyboard` correctly ignore it already (none of the four
  even accept a `max_scenes` parameter — confirmed by reading each function
  signature directly) and always process the full scene list, while
  `casting`-image rendering and `scene_render` correctly restrict themselves
  to it. **User clarified the intended principle**: only actual RENDERING
  stages (media generation — image/video API calls) should restrict
  themselves to `max_scenes`; every design/planning stage should stay aligned
  with the full scene list and generate for all scenes by default — meaning
  soundscape/visuals/cinematography/storyboard's existing behavior was
  already correct, and the one stage actually violating this principle was
  `screenplay` (`draft_screenplay`'s `max_scenes` defaulted to `3`, and
  `pipeline.run()` fed it the same render-scoped cap used for casting-images/
  video). Fixed: `draft_screenplay`'s default changed `3` -> `None` (all
  scenes when called without an explicit cap — the parameter itself is kept,
  so `stage screenplay --max-scenes N` can still explicitly request a smaller
  draft for quick standalone testing); `pipeline.run()`'s screenplay call
  site no longer passes its `max_scenes` through at all, so a full pipeline
  run always drafts every scene's screenplay regardless of how many scenes
  get rendered to video. Confirmed moodboard's existing `tiles` cap is a
  different, correctly-scoped mechanism (not a violation of the same
  principle) — tiles get rendered to actual images, unlike the moodboard's
  film-wide aesthetic fields, which were never scene-scoped to begin with.
  Deliberately left standalone `stage screenplay`'s CLI default (`max_scenes:
  1`, shared across all stage types via `stages.run_stage`) untouched — a
  different use case (explicit single-stage testing) from the full-pipeline
  behavior this fix targeted. Verified via a stubbed offline test (5 fake
  scenes, no `max_scenes` arg -> all 5 drafted). `README.md`'s Scene
  rendering section and the main CLI's `--max-scenes` help text both had the
  same stale "drafts AND renders" claim — corrected in both places.
- 2026-07-08 — **Veo prompt assembly rebuilt around the official five-part
  formula; depth-of-field handling removed; focus-hint dicts reconciled and
  de-duplicated; per-frame prompt logging added.** Several related fixes,
  building on each other: (1) found and fixed an "Action: no clear action
  verb detected" false-positive in `veo_guide.verify_prompt` — its verb
  whitelist was a narrow ~30-word tuple matched via naive substring search
  (no word boundaries), which let "close-up" accidentally satisfy the check
  via the substring "close"; replaced with a ~90-root regex, word-boundary
  matched, inflection-aware. (2) Added `output/logs/scene_NN_veo_prompts.txt`
  — `pipeline._write_scene_prompt_log`, wired into `_render_scene_frames` so
  every render (pipeline or standalone `render`/`stage scene_render`) writes
  a plain-text record of the exact prompt behind every frame. (3) Added a
  persistent `output/logs/gemini_api.log` — `gemini.set_log_dir`/`_log_call`,
  logging every real Image/Video/Video_extend API call (model, backend
  sdk/urllib, outcome, retry attempts) — previously only console `print`s,
  never persisted. Also fixed a related bug found live: `_generate_video_sdk`
  never inspected the SDK operation's `.error` field (same shape the urllib
  path already parses for a retryable code), so any transient Veo error
  (8/13/14) surfaced as a generic "SDK returned no video" and always fell
  through to a full urllib fallback conversation instead of retrying in
  place; now extracts `.veo_code` and retries transient codes within the SDK
  path itself, matching urllib's existing backoff. (4) User flagged a real
  self-contradiction found via the alt-prompt-format comparison exercise: a
  close-up panel's prompt carried both "Deep focus" (the storyboard agent's
  own prompt-example boilerplate, copied regardless of actual shot type) AND
  "shallow focus, portrait" (the shot-appropriate hint pipeline.py injected
  on top) — first fixed by having `_panel_video_prompt` strip a conflicting
  pre-existing focus term before injecting its own; **user then asked to
  remove ALL deep/shallow-focus handling entirely** rather than manage the
  conflict — depth-of-field is no longer asserted anywhere in `pipeline.py`/
  `fountain.py`; whatever the source content says about focus (or doesn't)
  stands untouched. (5) Auditing that removal surfaced two real gaps: (a)
  `fountain.py`'s `_VEO_FOCUS_FOUNTAIN` (the standalone `render` CLI's
  equivalent dict, tracked as a pair with `pipeline.py`'s `_VEO_FOCUS` per
  `veo_guide.py`'s own code-location list) still had every old deep/shallow
  value — fixed to match, so both render paths behave identically; (b)
  pipeline.py's dict was missing the `"EXTREME-CLOSE-UP"` (fully hyphenated)
  key variant fountain.py already had — added. Both dicts now byte-identical.
  Checking the live storyboard also showed the model *itself* almost always
  writes "Deep focus" regardless of shot type (its own prompt example biases
  it, the root cause from step 4) — with the downstream fix removed there was
  no longer anything correcting this, so close-ups never actually got
  "shallow focus". Fixed at the generation source instead: added a second,
  contrasting example panel (a CU) to `storyboard.py`'s JSON schema example
  (previously only ever showed one WS example), and strengthened the rule
  text to explicitly forbid copying a focus term from a different panel/
  example. (6) **User then gave a specific five-part prompt formula**
  (`[Cinematography]+[Subject]+[Action]+[Context]+[Style & Ambiance]`) and
  asked for `output/logs/scene_01_veo_prompts.txt` rephrased into it as a
  comparison file — produced `_alt.txt` (flowing prose), `_alt2.txt`
  (bracketed sections), `_alt3.txt` (labeled breakdown), and `_alt4.txt`
  (a six-part variant — Subject/Motion/Environment/Aesthetics/Camera/Audio —
  from a second screenshot the user provided). While preparing `_alt4`,
  spotted and fixed a second real defect surfaced by the exercise: the
  original five-part rephrase for frame 1 still carried the deep/shallow
  contradiction from step 4 (the alt-file work started before step 4's
  removal was applied) — fixed via the same conflict-stripping approach,
  confirmed against the live prompt data, both alt and canonical logs
  refreshed to match. (7) **User then confirmed via a live WebFetch of
  cloud.google.com's official Veo 3.1 prompting guide** that the five-part
  formula from step 6 is in fact THE authoritative, universal formula (exact
  wording match, no separate formula for image-to-video vs text-to-video) —
  a different, newer guide than the ai.google.dev API reference
  `veo_guide.py` syncs vocabulary from (that sync only extracts keyword
  categories — style/shot/camera/focus/audio terms — not structural formula
  order, so it can't auto-detect this kind of guide). **User asked to make
  this formula ALWAYS govern real Veo prompt construction, via explicit
  dictionary-style assembly** — rebuilt `_panel_video_prompt`'s visual half
  entirely: new `_five_part_veo_prompt` + `_panel_cinematography`/
  `_panel_subject`/`_panel_context`/`_panel_style_ambiance` build a five-key
  dict from STRUCTURED data (casting.json's `character.physical_form` for
  Subject, a cast location entry's `visual_prompt` for Context, the scene's
  `visual_overview` for Style & Ambiance, the panel's own shot_type/
  camera_angle/camera_movement/lens for Cinematography) rather than the
  storyboard agent's free-text `image_prompt` — deliberate, since that text
  has no guaranteed internal order and reordering an opaque LLM-written blob
  after the fact isn't reliable, only reconstructing from structured fields
  is. `_render_scene_frames` now builds and threads a real `casting_lookup`/
  `location_desc`/`visual_overview` into every call; `image_prompt` remains
  as a fallback for callers without that context (e.g. standalone
  `gen-video`). Also added `_VEO_SHOT_LABEL` (abbreviation → natural language,
  e.g. `WS`→"wide shot", `ECU`→"extreme close-up") for cleaner Cinematography
  phrasing — found and fixed a duplication this introduced (`ECU`'s
  `_VEO_FOCUS` hint used to include "extreme close-up" text, which now
  doubled up with the new label), fixed in both `pipeline.py` and
  `fountain.py` by trimming those hint values down to just the framing term
  ("portrait"/"macro lens"), since shot-type wording is now always supplied
  by the label itself. Verified end-to-end against the live storyboard: all
  8 panels pass `verify_prompt` with zero issues, five-part order confirmed
  correct and duplication-free across every shot type present. Docs synced:
  `CLAUDE.md`'s stale "Subject→Action→Style→Camera→Focus/Ambiance" bullet
  replaced with the current formula + architecture description (session-log
  history left untouched — those entries describe what was true *then*);
  `README.md` gained a formula summary in the Scene rendering section;
  `storyboard.py`'s module docstring (NOT its `PROMPT` string — caught and
  reverted an early mistake that would have leaked implementation detail
  into the actual LLM-facing prompt) gained a note that `image_prompt` is a
  fallback format only now; `veo_guide.py`'s `_CODE_LOCATIONS` updated to
  distinguish the two different guides in play. Nothing committed yet at
  time of writing — several distinct fixes accumulated across the session,
  pending a single reviewed commit.
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
