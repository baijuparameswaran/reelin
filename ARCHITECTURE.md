# Architecture

> How `reel` is built: stack, layout, hardware constraints, and the
> established conventions/decisions behind each subsystem. Stable reference
> material — changes here are architectural, not day-to-day progress (see
> [PROGRESS.md](PROGRESS.md) for that) or per-agent prompt detail (see
> [AGENTS.md](AGENTS.md)).

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
- `tests/test_prompt_rules.py` — stdlib `unittest`, offline (no LLM/API
  calls, <1s): validates every agent PROMPT template actually follows this
  project's prompting conventions (sandwiching, tie-breakers, schema
  fields) and that the deterministic functions backing some of those rules
  (`scenes._validate`/`_attach_source_excerpts`/`_reconcile_character_names`,
  `casting._location_entries`/`_prop_entries`, `storyboard._build_scene_board`,
  `duration_budget.suggest_shots_per_scene`) do what the prompts describe,
  plus a static check (via `inspect.getsource`) that the steer-vs-neutral
  provider-policy split below actually holds in the code. Run via `make test`.
- Entry points via `Makefile`: `setup`, `demo`, `run`, `models`, `update`,
  `update-all`, `install-cron`, `test`. Run pipeline as `python -m reel.cli`.

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
- **`revise` inherits the original run's attributes** — profile, max_scenes,
  target_duration, and genre/moodboard creative-direction steering —
  instead of silently reverting to bare defaults. `revise` is a SEPARATE
  process invocation from the `pipeline.run()` that produced the checkpoints
  it's editing, so none of these carried over automatically before this fix:
  `llm.set_direction`'s process-wide directive started unset in the new
  process (losing genre/moodboard steering on every regenerated stage even
  though genre.json/moodboard.json exist on disk); `stages.run_stage`'s
  calls never passed `profile=` (silently reverting each stage to its own
  config default instead of e.g. a `--profile fast` override); `max_scenes`
  wasn't threaded through at all (re-capping a completed `--max-scenes all`
  run back down to `run_stage`'s own default of 1 the moment anything
  downstream got regenerated); and `--target-duration`'s scene/shot-count
  guidance had no pathway into `revise`'s calls whatsoever. Fixed via:
  `pipeline.compose_direction(genre_spec, moodboard_spec)` — the genre+
  moodboard guidance composition `run()`'s local `apply_direction()` used
  inline, extracted to module level so `cli._restore_direction(out)` (new,
  called once at the top of `_revise_loop`) can reload genre.json/
  moodboard.json from `out/` and re-apply the exact same steering before any
  stage regenerates. `cli._load_run_params(out)` (already existed, for
  `--resume` inheritance) is reused here too — `_revise_loop` loads
  `profile`/`max_scenes`/`target_duration` once and threads them through
  every `_revise_one`/`_revise_source` call, which pass `profile=`
  explicitly into every `stages.run_stage(...)` call site. `max_scenes`
  needed one deliberate exception, `cli._effective_max_scenes(stage_name,
  max_scenes)`: `screenplay` always gets `None` (uncapped) regardless of the
  inherited value, matching the established "`--max-scenes` only restricts
  actual media-rendering stages, screenplay always drafts everything" policy
  — passing the inherited value uniformly would have reintroduced the exact
  cap that policy deliberately removed. `target_duration` needed new
  plumbing: `stages.py`'s `_scenes`/`_cinematography` wrappers gained
  `target`/`shots_guidance` params (both default to falsy = "no override,
  use this stage's own default", so every pre-existing call site is
  unaffected), and `cli._duration_kwargs(stage_name, out, target_duration)`
  computes the right one from `reel.duration_budget` — `cinematography`'s
  needs the CURRENT scene count, reloaded fresh from disk since `scenes` may
  have just been regenerated earlier in the same revision round.
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
- **Prompt-pitfall audit (lost-in-the-middle + priority conflicts) — every
  multi-rule agent prompt now "sandwiches."** The long-context agents
  (`scenes.py`, `casting.py`, `soundscape.py`, `visuals.py`,
  `cinematography.py`, `screenplay.py`, `storyboard.py` — the ones with
  8-17 rule bullets ahead of a large data block, per Liu et al. 2023's
  finding that LLMs attend to a prompt's start/end far better than its
  middle) now each end with a short "Before you respond, re-check against
  [the data] above" block restating their highest-stakes rules (source
  fidelity / verbatim quoting / dialogue-copied-verbatim / scene_number
  correctness / isolation clauses / gender consistency), in addition to
  the rules already stated up front — not a replacement, a second
  reinforcement placed where recency actually helps. Also fixed three real
  **priority conflicts** (two rules that could pull opposite ways with no
  stated tie-breaker): `cinematography.py`'s "motifs should develop/recur
  across scenes" vs. "vary angle/lens for a shared location's establishing
  shot" (now: location-distinctness wins for that specific shot type,
  motif development happens in the scene's OTHER shots);
  `screenplay.py`'s camera-coverage-derived shots vs. source-material
  fidelity (now: SOURCE OVER COVERAGE — adapt or drop a shot rather than
  inventing content to fill a suggested camera set-up); `casting.py`'s
  "invent the actor" vs. "only source-grounded attributes" (clarified as
  two different scopes — invent WHO plays the role, ground WHICH
  attributes they have — not an actual conflict, just ambiguous wording).
  `storyboard.py` also got a clarifying note that "self-contained
  image_prompt" and "don't re-describe the location fully in every panel"
  aren't in tension either — self-containment covers character look/
  camera grammar/action per panel; the location gets a brief anchor
  phrase after the first panel, not a restatement. The four short single-
  call prompts (`characters.py`, `structure.py`, `genre.py`,
  `moodboard.py`) and the grader prompts (`fidelity.py`, `revision.py`)
  were reviewed and left as-is — already schema/rules-then-data-last with
  short data blocks (2.5-12k chars) and no multi-rule lists, so the
  lost-in-the-middle risk that motivated this audit doesn't apply to them.
  Every changed prompt verified via `py_compile` + a direct `.format()`
  smoke test confirming the reminder block renders and lands at the true
  end of the string.
- Version control: git, branch `main`.
