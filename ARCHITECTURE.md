# Architecture

> How `reel` is built: stack, layout, hardware constraints, and the
> established conventions/decisions behind each subsystem. Stable reference
> material — changes here are architectural, not day-to-day progress (see
> [PROGRESS.md](PROGRESS.md) for that) or per-agent prompt detail (see
> [AGENTS.md](AGENTS.md)).
>
> Compacted 2026-07-16 and again 2026-07-24: condensed from a much longer,
> narrative form (deep bug-fix-by-bug-fix justification, verification
> steps, running test counts) down to what each convention IS and why, per
> direct instruction to keep the auto-loaded context tidy. That history
> still lives, in condensed form, in PROGRESS.md's session log.

## Stack & layout
- **Runtime:** Python 3.12 (`.venv/`), local LLMs via **Ollama**. Core 3rd-party
  dep is PyYAML; the Ollama client is stdlib `urllib` (in `reel/llm.py`). Image
  rendering adds **optional** deps (`diffusers`/`torch`/etc., see
  `requirements-image.txt`) — the text pipeline runs without them.
- `reel/` — package:
  - `models.py` — provider policy + text front door for graders; `text()`
    always routes to a LOCAL profile (`local_profile`) with `steer=False`, so
    graders stay both neutral and independent of whatever authored the
    artifact.
  - `gemini.py` also carries `generate_text` — the hosted frontier-tier text
    call behind `provider: gemini`. Reuses this module's existing key,
    headers, `_post` retry/backoff, and `gemini_api.log`, and treats a
    blocked prompt or a `MAX_TOKENS`/early finishReason as an error rather
    than returning empty or truncated content.
  - `stages.py` — per-stage registry + `run_stage` for independent
    invocation.
  - `llm.py` — open-model client: model-agnostic Ollama + profile/fallback
    + `with_feedback` + the global creative-direction steering hook
    `set_direction`.
  - `pipeline.py` — orchestration + per-stage gates.
  - `gate.py` — human-in-the-loop review gate.
  - `imagegen.py` — pluggable text-to-image (Gemini default, diffusers,
    auto1111).
  - `gemini.py` — Google Gemini REST helpers (image gen + Veo video, stdlib
    urllib, API key from env).
  - `i2v.py` — pluggable image-to-video (default Gemini Veo; also
    diffusers LTX/Wan or a remote endpoint); Veo pre-flight prompt
    verifier via `veo_guide.verify_prompt`.
  - `veo_guide.py` — Veo prompting-guide snapshot + staleness check +
    `verify_prompt` (checks a Subject/Action/Style/Camera/Focus-ish keyword
    is present, order-agnostic — distinct from `veo_prompt.py`'s five-PART
    assembly ORDER below); manual refresh via `python -m reel.cli veo-sync`.
  - `veo_prompt.py` — the actual Veo prompt builder: the five-part
    `[Cinematography]+[Subject]+[Action]+[Context]+[Style & Ambiance]`
    formula, Subject-anchoring for an attached seed/reference image,
    action/dialogue-grounded character relevance filtering, and audio-cue
    assembly. Pure, no render-loop/filesystem state — shared by
    `pipeline.py`'s main render path AND `fountain.py`'s standalone render
    path so the two entry points can't silently drift apart.
  - `cli.py` — entry point. `manifest.py` — model list for the updater.
  - `fountain.py` — Fountain parser + screenplay→storyboard/shot builder
    for the standalone `render` command; `to_storyboard` builds a board
    shape-compatible with `storyboard.py`'s own deterministic build, so it
    feeds the SAME shared render formula the main pipeline uses.
  - `spend.py` — estimated $ spend from `gemini_api.log` (per-model/kind
    pricing snapshot); `python -m reel.cli spend`, printed after every run.
  - `session.py` — session identity (one story-to-video run's id,
    persisted to `output/session.json`); zero dependency on other `reel`
    modules, imported freely without import-cycle risk.
  - `artifact_diff.py` — deterministic keyed-list diff (no LLM); compares
    two versions of a stage's JSON artifact to find which scene numbers /
    character-or-location names / panel numbers actually changed, and
    flags a scene-count/reorder change as `drastic`.
  - `revision_merge.py` — the single shared `merge_by_key`/`merge_fields`
    splice primitive every scoped-revision agent call uses.
  - `panel_grouping.py` — pure, deterministic same-cast/duration-bounded
    panel grouping for multi-segment timestamped Veo prompts (see
    "Multi-segment timestamped Veo prompts" below); the sole owner of the
    `char_set_changed` "shot boundary" predicate, imported back into
    `pipeline.py` as `_char_set_changed`.
  - `agents/` — ingest, genre, structure, moodboard, characters, casting,
    scenes, soundscape, visuals, cinematography, storyboard, screenplay,
    fidelity, critique, revision.
- `config/models.yaml` — model profiles, per-agent profile map, `hitl` gate
  knobs, `genre`/`moodboard`/`fidelity`/`critique` blocks, `image`/`video`
  blocks (backend/model), runtime knobs. Validated at every CLI invocation
  (best-effort, non-blocking) by `llm.validate_config`.
- **Gemini API key** (`python -m reel.secrets`, chmod-600 file under
  `~/.config/reel/` — never an env var, never anything in the project dir):
  one key now serves three things — image, video, and the optional hosted
  `frontier` text profile. Without it, image/video stages no-op gracefully
  with a hint and a hosted text profile falls back to its local
  `fallback_profile`; the local text pipeline is unaffected.
- `scripts/` — `update-models.sh` (cadence), `install-cron.sh`,
  `model-updates.log`.
- `samples/` — bundled test story. `output/` — generated artifacts
  (gitignored).
- **Tests cannot spend money (`tests/__init__.py`, enforced not assumed):**
  importing the test package blocks outbound sockets, so a test that forgets
  to mock its transport fails with `NetworkBlockedInTests` naming the address
  it tried, instead of quietly billing a Gemini/Veo call. Blocked at the
  SOCKET layer rather than by patching `urlopen`/`gemini._post`, because
  those are per-module — an SDK doing its own HTTP (`google-genai`) slips
  past a per-module guard but not this one. Loopback is blocked too, so no
  test can depend on a live Ollama daemon. `tests/test_no_api_cost.py`
  meta-tests the guard itself (it would otherwise fail silently if deleted)
  and asserts the suite behaves identically with and without a Gemini key.
- `tests/test_prompt_rules.py` — stdlib `unittest`, offline (no LLM/API
  calls, <1s): validates every agent PROMPT template follows this
  project's prompting conventions (sandwiching, DO-NOT lists,
  tie-breakers, schema fields) and that the deterministic functions
  backing some of those rules do what the prompts describe, plus a static
  check that the steer-vs-neutral provider-policy split actually holds in
  the code. Run via `make test`.
- Entry points via `Makefile`: `setup`, `demo`, `run`, `models`, `update`,
  `update-all`, `install-cron`, `test`. Run pipeline as `python -m reel.cli`.

## Hardware reality (binding constraint)
Host `dev-host`, WSL2/Ubuntu 24.04. **NVIDIA GeForce RTX 2070 Super, 8 GB
VRAM** (confirmed via `nvidia-smi`). WSL RAM ~12 GB (`MemTotal` ≈ 12 GB; 16 GB
laptop) via `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=12GB`). 4 GB swap.
872 GB disk.
- Installed models and VRAM fit: `qwen3:4b` (2.5 GB, full GPU), `qwen3:8b`
  (5.2 GB, full GPU), `gemma3:12b` (8.1 GB, partial offload ~85% GPU).
- **GPU requires the official Ollama installer, not the snap.** The snap
  package uses strict confinement that blocks `/dev/nvidia*` access — models
  fall back to 100% CPU. Install via:
  `curl -fsSL https://ollama.com/install.sh | sh`
  Verify with `ollama ps` — look for non-zero "Size VRAM" after loading a model.
- **With GPU:** first-token latency ~2–5 s (vs ~30–60 s on CPU).
  `request_timeout_seconds` is **120 s**. **Intel NPU is not accessible from
  WSL2** — GPU is the only accelerator usable by Ollama here.
- **23 GB RAM** enables GPU+CPU split for models larger than VRAM: qwen3:14b
  (~8.9 GB, ~90% GPU) and qwen3:30b (~19 GB, ~42% GPU + ~11 GB CPU RAM).
- Profile model assignments: `fast`=qwen3:4b (100% GPU, ~80 tok/s),
  `quality`=qwen3:8b (100% GPU, ~40 tok/s), `synthesis`=qwen3:14b (~90% GPU,
  ~25 tok/s, num_ctx 16384), `quality_high`=qwen3:30b (42% GPU + CPU,
  ~8 tok/s, best quality escalation target). Reasoning traces (`think`) are
  **disabled** for all profiles by default — set `runtime.think: true` (or
  per-profile) to re-enable (~2× slower but higher quality).
- `max_parallel_agents: 2` — GPU holds two 4B or one 8B+4B simultaneously.

## Conventions & decisions
- **Model-agnostic by design:** agents pick a *profile* (`fast`/`quality`/
  `synthesis`/`quality_high`), never a model name. Preferred = Qwen3 4B/8B;
  auto-fallback to other installed models so the pipeline always runs.
- **Provider policy (single source of truth: `reel/models.py`):** Gemini is
  used ONLY for image + video generation, and only if `GEMINIAPIKEY` is set;
  every text/LLM stage always uses the local open models (Ollama).
  Image/video config backends are `auto` → resolve to `gemini` when a key
  exists, else the `open_backend`. `reel.models` exposes only `text()` for
  grader agents; `reel.llm` is the open-text engine behind it. No Gemini
  text path exists by design. Text is a two-way split: **creative** stages
  default to the local open models and may opt into a hosted frontier profile
  (below); **grader/checker** stages are always local and always neutral.
- **Hosted frontier text tier (`provider: gemini`,
  `gemini.generate_text`, profile `frontier`):** a deliberate, narrow
  exception to the otherwise absolute "no Gemini text path" rule — taken
  because it reuses the key image/video already use, so it needs no second
  credential, no SDK, and no server-side setup. The provider lives on the
  PROFILE, not the agent, so "agents pick a profile, never a model name"
  still holds: opting a stage in is a one-line `agent_profiles` change and no
  agent module learns a provider or model string. Nothing uses it by default.
  It exists for the one place local context length was a hard ceiling rather
  than a tradeoff: `quality_high` at `num_ctx: 8192`, minus the scenes
  prompt's ~2,900 tokens of rules and ~3,000 of `MAX_CHARS` source text,
  leaves room for only ~14 scenes of JSON before the array truncates —
  capping how completely the film can ever cover the story. A 1M-in/65K-out
  model removes that ceiling, `MAX_CHARS` truncation, and the ~13-min-per-
  stage wall at ~8 tok/s together. Degrades to the profile's
  `fallback_profile` with a printed note when no key is set, or on an unknown
  provider name, so the pipeline still runs fully offline; a failure from a
  provider that IS reachable propagates instead (a real error, not a
  fall-back-to-local situation). Steering is composed BEFORE the provider
  branch, so a hosted creative stage is steered like any other. Text calls
  log to the same `gemini_api.log` as image/video, so `spend.py` sees them
  with no second format to parse. **Graders stay local**
  (`models.local_profile`, applied inside `models.text`) — the provider-level
  half of the independence rule `steer=False` already enforces, and load-
  bearing now that a creative stage can run on Gemini: the grader must not be
  the same model that authored the artifact. It also keeps grader cost at
  zero, since graders fire on every stage.
- **Source-text budget is per-PROFILE, not a global constant
  (`llm.max_chars`):** `MAX_CHARS` (12,000 chars ~ 3,000 tokens) is sized for
  a local profile at `num_ctx: 8192`. Agents call `llm.max_chars(profile)`
  instead of reading the constant, because the truncation happens in the
  agent BEFORE the prompt is built — so switching a stage to a 1M-context
  model would otherwise still cut the story at its first ~2,000 words.
  Resolution: `options.max_source_chars` on the profile wins; a LOCAL profile
  derives its own budget from its `num_ctx` (`local_max_chars`, below); a
  hosted profile whose provider isn't usable takes its `fallback_profile`'s
  budget, so the truncation decision can't disagree with where `llm.generate`
  actually routes; otherwise the provider budget (`gemini`: 600,000 chars).
  Never raises — unknown/malformed yields `MAX_CHARS`, the safe floor.
  Applies to the three whole-source agents: `structure`, `characters`,
  `scenes`.
- **Local caps are hardware-derived per model, not one flat number
  (`llm.local_max_chars`):** `num_ctx` is this project's hardware knob —
  chosen per profile against 8 GB VRAM plus the CPU-RAM KV cache — so the
  source budget derives from it. It's a TOTAL (prompt + response) budget, so
  `_PROMPT_RESERVE_TOKENS` (3,000 — sized on `scenes`, the largest of the
  three whole-source prompts at ~2,900) and `_OUTPUT_RESERVE_TOKENS` (2,000,
  matching what was measured left over at the old flat cap) come off the top
  before the rest is spent at `_CHARS_PER_TOKEN` (4). Today that yields
  12,768 chars at `num_ctx: 8192` (fast/quality/quality_high — within 10% of
  the old hand-tuned 12,000, so the tiers this host actually runs on barely
  move) and 45,536 at 16,384 (`synthesis`, previously capped as if it were
  8K). Floors at `_MIN_SOURCE_CHARS` for a tiny `num_ctx`; a profile
  declaring no `num_ctx` derives back to the 8,192 default rather than
  collapsing to the floor.
- **Frontier model choice is measured, not assumed:** benched on the real
  scenes prompt against the bundled samples — `gemini-3.6-flash` (8-10
  scenes, 26-36s, 4/4 parsed) chosen over `gemini-3.1-pro-preview` (9-13
  scenes, 42-70s, 3/4 parsed) because the scene-count difference sits inside
  each model's own run-to-run variance while flash is GA, ~1.7x faster, and
  produced no unparseable responses. `gemini-2.5-pro` is listed by the models
  endpoint but 404s on `generateContent`. See PROGRESS.md for the raw runs.
- **Story-fidelity at every stage (`reel/agents/fidelity.py`, open model):**
  each stage's output is scored against the original story text
  (`fidelity_score` 0-100, drift/omissions/contradictions, verdict) before
  the review gate, shown at the gate readout. `fidelity.score_pipeline`
  aggregates per-stage scores into one pipeline score (`overall =
  round(0.5·mean + 0.5·min)`, so the weakest stage caps consistency) →
  `output/fidelity.json`. Toggle via config `fidelity.per_stage`;
  best-effort, never blocks.
- **Self-critique-and-refine at every creative stage
  (`reel/agents/critique.py`, open model, wired into `pipeline._gated`):** a
  third check, distinct from fidelity (source consistency) and genre (genre
  fit) — judges craft quality of a stage's own output against its own
  governing SYSTEM/PROMPT template. Runs automatically before the operator
  sees the gate; a `needs_improvement` verdict triggers a re-run via the same
  `feedback` mechanism a human's gate feedback uses. `critique.iterations`
  (config, **default 1**) sets how many critique→refine rounds are allowed —
  at the default this is exactly one pass and the refined result is not
  re-critiqued, identical to the original behaviour; raising it re-critiques
  each refinement and stops early the moment the critic is satisfied, at N
  times the model calls. `critique.enabled` (default true) is still the off
  switch — a bogus `iterations: 0` floors to 1 rather than silently disabling.
  The pre-critique raw result is preserved as `output/<stage>.0.json`.
  Best-effort, never blocks. Scoped to `pipeline.run()` only, not the
  standalone `revise` gate.
- **Empty-result safety net (`stages.is_stage_result_empty`/
  `pipeline._ensure_nonempty_result`) — not toggleable, always on:** a
  generation with nothing meaningful (a JSON-parse failure, or the stage's
  own defining content key missing/empty — e.g. `scenes` for the scenes
  stage, per `stages._STAGE_CONTENT_KEYS`) is rerun ONCE via the same
  `rerun_fn` every other re-run path uses (an explicit "try again" note,
  not silence); still empty after that retry raises `StageEmptyResultError`,
  failing the run rather than letting empty data reach downstream stages.
  Checked at every point `_gated` produces a fresh result — initial
  compute, self-critique refine, and every gate-loop feedback rerun, not
  just the first attempt. `stages.run_stage` (the standalone `stage NAME`
  path, which doesn't go through `_gated`) carries its own copy. Scoped to
  the creative content-generating stages only — not casting_images/
  moodboard_tiles/scene_render/fidelity, whose "empty" means something else
  (no API key, an intentional no-op by design).
- **Scene-structure alignment (`fidelity.check_scene_alignment`/
  `strip_orphan_scenes`) — deterministic, self-healing, distinct from story
  fidelity:** does a scene-keyed stage's own data structurally match
  scenes.json (the actual upstream input it depends on)? Unlike
  fidelity/genre (qualitative, advisory-only), this is an objective
  structural check — no LLM — that self-heals automatically inside
  `run_group` before the gate: strips any orphan scene, reiterates any
  missing one (scoped `existing=`/`revise_keys=`), and only surfaces a
  remaining gap at the gate if the single reiteration attempt didn't fully
  close it.
- **Creative direction = genre + moodboard steer every stage
  (`reel/agents/genre.py`, `reel/agents/moodboard.py`, open models):** genre
  is resolved once up front (`--genre` > config > auto-inferred); moodboard
  runs right after structure (film-wide visual-tone bible + render-ready
  `tiles`, capped to `max_scenes`). The pipeline composes
  `genre.guidance()+moodboard.guidance()` into `llm.set_direction()`
  (prepended to every steered generation's system message); graders call
  `models.text` with `steer=False` and judge neutrally. Genre also enforces
  per-stage alignment (`genre.enforce_stage`, scored beside fidelity) and
  aggregates via `genre.score_pipeline`. Moodboard tiles are rendered to
  actual images via the image backend; the moodboard spec itself is
  text-only (open models).
- **Storyboard is built deterministically (no LLM) except when `feedback`
  is given:** `_build_scene_board` constructs the entire scene from the
  fused upstream bundle (cast look, visuals, soundscape, camera coverage,
  screenplay's own shots/dialogue) with zero LLM calls — a fusion stage
  shouldn't make new creative decisions, only combine ones earlier stages
  already made. Panel alignment comes from pairing cinematography's shot
  list with screenplay's (`_align_shots`); `characters_in_frame` for close
  shots is the one genuinely invented field (a documented heuristic:
  dialogue speaker, else full scene cast). `feedback` is the one path that
  still calls the LLM (`_llm_generate_scene`), since a directed creative
  note needs judgment the deterministic builder can't provide.
- **Target total runtime (`--target-duration N`, default 45s,
  `reel/duration_budget.py`):** an engine-independent budget threaded
  through SHOT-count planning guidance (soft, secondary to cinematography's
  own coverage rules) and each clip's requested render duration. It
  deliberately does NOT set scene count any more: `suggest_scene_target` used
  to interpolate a literal range ("roughly 2-4 scenes" at the 45s default)
  into the end of the scenes prompt's opening sentence, the highest-salience
  position and immediately after two anti-inflation warnings, where a
  concrete number reliably beat the coverage rules forty lines below. That
  made a runtime default — not the story — decide how much of the story got
  filmed, and contradicted this project's own rule that a budget bounds
  RENDERING (`--max-scenes`) while every design stage processes everything.
  `segment_scenes` now uses its own coverage-first, numberless default
  (parsed from the storyboard panel's own duration estimate). Each video
  backend translates the requested duration its own way — Veo only accepts
  4/6/8s exactly (`i2v._veo_nearest_valid_duration` rounds/forces as
  needed); diffusers/http backends honor it as a frame count.
  `--target-duration` is inherited across `--resume` like
  `--max-scenes`/`--profile`.
- **Standalone video render (`python -m reel.cli render [--fresh]`):**
  builds a camera-directed render plan from `screenplay.fountain`+
  `cinematography.json` (every scene/shot, no caps) via
  `fountain.to_storyboard`, then renders through the SAME
  `pipeline._render_scene_frames` the main pipeline uses — no LLM stage
  runs. `gemini.generate_video` retries transient HTTP/Veo-operation errors
  with backoff.
- **Veo prompt construction is one shared module (`reel/veo_prompt.py`),
  not duplicated per entry point:** the five-part formula, Subject-anchoring
  for an attached reference image, action/dialogue-grounded character
  relevance filtering, and audio-cue assembly all live here, imported by
  both `pipeline.py`'s main render path and `fountain.py`'s standalone
  path. Fountain-parsed speaker cues (ALL CAPS) are resolved
  case-insensitively against casting.json's Title-Case names
  (`fountain._canonical_name`) so a dialogue speaker is never wrongly
  marked off-screen.
- **The Veo [Action] element carries a deliberate "director's freedom"
  layer (`veo_prompt.panel_action`):** each panel's `emotional_note`
  ("the emotion this panel must evoke," authored by the cinematography
  agent, falling back to visuals'/soundscape's scene-wide read, per
  `storyboard._build_scene_board`) is folded into the Action clause as a
  short performance-direction phrase ("Alice reaches for the doorknob,
  conveying quiet dread"), deduped against text already present in the
  action. The point where *rendering* adds interpretive performance
  direction on top of an already-established beat — never WHAT happens
  (scenes.py/screenplay.py stay the sole source-bound authority for that),
  only HOW it's performed/felt.
- **The same interpretive boundary now also applies at the SCENES stage
  (`scenes.py` rule 11, DIRECTOR'S INTERPRETIVE EXPANSION):** every scene
  carries `emotional_beat` (what the moment must make the audience feel) and
  `expression` (how that reads on a face/body), and an emotional or
  expressional moment the source only *implies* — the held look, the
  decision not to speak, the beat where grief lands — is treated as filmable
  material that may even earn its own scene. This is the one deliberate
  loosening of rules 1-2 ("SOURCE TEXT IS THE ONLY AUTHORITY"), and it is
  scoped exactly as narrowly as the render-time layer above: an inferred
  EMOTION is grounded in the source, an inferred PLOT POINT is not (no
  invented events, characters, locations, props, spoken lines, or outcomes).
  Two structural consequences: rule 4 keeps `summary` unembellished and
  source-checkable so the interpretive layer stays quarantined in its own
  named fields (fidelity grading still has a clean record to judge), and
  rule 7 was tightened so "two beats from one passage" can't degrade into
  two scenes restating one moment. Both fields are propagated to every stage
  that can act on them — soundscape/visuals/cinematography (each grounding
  them to its own `emotional_function`), screenplay
  (`_director_block`), and storyboard (a `bundle.director` block, used as
  the `emotional_note` floor beneath the DP's more specific per-shot read) —
  rather than stopping in scenes.json, which is the dead-field failure mode
  `visuals.key_props` and panel `emotional_note` both previously had.
- **Config schema validation (`llm.validate_config`) and estimated $ spend
  tracking (`reel/spend.py`) are both best-effort, non-blocking checks
  layered on existing state:** `validate_config` is a lightweight,
  dependency-free shape check for `config/models.yaml` (unknown keys, wrong
  types on consequential fields), printed once at the top of every
  `cli.main()` invocation — never blocking, just makes a typo loud.
  `spend.py` aggregates real-money cost estimates by re-reading
  `gemini_api.log`'s already-logged per-call params against a pricing
  snapshot (documented as an estimate, not a reconciled bill); printed
  after every pipeline/`render` run and available on-demand via
  `python -m reel.cli spend`.
- **Per-stage abstraction + independent invocation (`reel/stages.py`):**
  every stage is declared once as a `Stage` (name, input artifacts, agent,
  `produces`) in a registry, letting any single stage run independently
  (`run_stage("scenes", ...)` / `python -m reel.cli stage scenes`) by
  resolving each dependency from a prior `output/<input>.json` checkpoint.
  `python -m reel.cli stages` lists stages + inputs.
- **`runtime.max_parallel_agents`** controls how many agent calls run
  concurrently — 1 on CPU-only/single-model hosts (sequential), raised on
  GPU/multi-model hosts to actually parallelize independent branches
  (structure ‖ characters; soundscape ‖ visuals ‖ cinematography).
- **Human-in-the-loop (`hitl` in models.yaml):** every LLM stage gates for
  review — approve, or type feedback to re-run that stage.
  `enabled: false` for batch/unattended; `timeout_seconds` auto-approves on
  idle. Parallel branches compute together, then gate sequentially; each
  stage also writes its own `output/<stage>.json`.
- **Streaming + timeouts:** `llm.generate` streams tokens, so
  `runtime.request_timeout_seconds` is an *inactivity* window, not a
  total-time cap.
- **Pause / resume:** type `stop` (or Ctrl-C) at any gate to pause;
  `--resume` reloads completed stages and continues from the first
  unfinished one. `--max-scenes`/`--profile`/`--target-duration`/
  `--no-render` are all inherited across `--resume` (persisted to
  `output/run_params.json`) rather than silently reset — an explicit flag
  on the resume command always overrides and updates the stored value.
- **`--no-render` skips casting-image + video rendering for a normal full
  pipeline run** (mirrors `revise`'s own render toggle) — every
  design/planning stage still runs normally; only the two API-cost stages
  are skipped. Threaded via a `dry_run` param all the way down the render
  call chain — a dry-run call still builds and logs its exact request
  params (`gemini_api.log`, `outcome=skipped(no-render)`) before skipping
  the actual network call.
- **Session identity (`reel/session.py`):** one full story-to-video run is
  a session, identified by a generated id persisted to
  `output/session.json`. A fresh (non-`--resume`) run mints a new id;
  every other invocation (`--resume`, standalone `stage`/`render`,
  `revise`) reattaches to whatever session is already in `out`.
  `gemini_api.log` lines and per-scene Veo prompt logs are tagged with the
  active session id.
- **`gemini_api.log` records the actual request parameters for every real
  Gemini/Veo call** (prompt, seed/reference paths, aspect_ratio,
  resolution, duration, person_generation), not just that a call happened
  — a `params={...}` JSON blob appended to each line, parseable
  independently of the human-readable summary fields.
- **Revision agent (`python -m reel.cli revise`, `reel/agents/revision.py`
  + `reel/artifact_diff.py` + `reel/revision_merge.py`):** a standalone,
  post-hoc loop over a completed/paused run's checkpoints. Each round:
  hand-edit a stage's JSON (or the raw source text) via `$EDITOR` →
  `artifact_diff.diff_artifact` deterministically finds which scene
  numbers/names/panels actually changed (no LLM) → confirm → selectively
  regenerate only the affected downstream stages
  (`stages.downstream_of`), splicing untouched entries back in
  byte-identical via `revision_merge.merge_by_key`/`merge_fields`
  (field-level, not just entry-level). A drastic change (implied scene
  add/remove, or a whole-file artifact like structure/moodboard) falls back
  to a full regen with a confirm+warning. A hand-edit directly to
  scenes.json CAN add or delete a scene (the one scene-keyed artifact that
  allows it); deletion strips the scene from every downstream artifact +
  the video manifest via `fidelity.strip_orphan_scenes`. Casting identity
  is preserved automatically: an untouched character/location's
  `visual_prompt` stays byte-identical, so its rendered portrait is reused
  rather than regenerated. A source-TEXT edit is scoped to the scenes it
  actually affects via a deterministic diff pre-filter + an LLM
  confirmation call (`agent_profiles.revision: quality_high`, the largest
  local tier), falling back to a full regen only when no scenes.json
  exists yet or the change is judged drastic.
  `revision.suggest_ripple_scenes` (advisory-only, never auto-applied) and
  `revision.is_drastic_identity_change` (deterministic `difflib` heuristic,
  warns before silently re-casting a drifted character/location) round out
  the agent. Panel-level video re-render (`pipeline.rerender_panels`)
  cascades exactly one hop forward from a targeted panel then stops, a
  deliberate Veo-cost bound — a `.hash` sidecar two hops out is recomputed
  (not the actual clip) so a later pass doesn't redundantly re-render it.
- **`revise`'s downstream cascade gets the SAME per-stage review gate a
  fresh `pipeline.run()` uses** (`cli._gate_stage_result` reuses
  `pipeline._gated`, not a second implementation) — fidelity/genre scoring,
  feedback-driven re-run (re-scoped to the same round's
  `existing`/`revise_keys`), view-to-edit, and auto-approve timeout all
  behave identically to a fresh pipeline stage's gate. `gate=None` (the
  default at every call site) is a complete no-op, preserving `revise`'s
  pre-existing auto-apply behavior for any caller that doesn't pass a real
  gate.
- **`revise` skips casting + all image/video rendering by default, with
  image rendering and video rendering independently enable-able**
  (`IMAGE_RENDER_STAGES`/`VIDEO_RENDER_STAGES`, since they spend different
  budgets for different reasons) — via `--render-images`/`--render-video`
  flags, `render images/video on/off` in the interactive loop, or directly
  hand-editing `casting` (which always applies regardless of the flags —
  only its own downstream stages respect them).
- **`revise` inherits the original run's attributes** — profile,
  max_scenes (see below for the one exception), target_duration, and
  genre/moodboard creative-direction steering — instead of reverting to
  bare defaults, since `revise` is a separate process invocation with no
  automatic carryover. `cli._restore_direction` reloads genre.json/
  moodboard.json and re-applies `llm.set_direction` before any stage
  regenerates; `screenplay` always gets an uncapped `max_scenes` regardless
  of the inherited value, matching the "screenplay always drafts
  everything" policy.
- **`revise` no longer inherits `max_scenes` — it always operates as
  `--max-scenes all`**, since scoping is controlled by `revise_keys`, not
  `max_scenes`, and an inherited prototype-scale cap could otherwise hide a
  diff-identified scene from the render stages for no reason. Paired with
  upfront "evaluation" and per-stage regeneration printouts so the operator
  sees both what was identified and what each stage is doing about it.
- **A source-TEXT edit is scoped to the scenes it actually affects**, not
  always a full regen: `artifact_diff.candidate_changed_scenes`
  (deterministic pre-filter) + `unified_source_diff` (compact
  paragraph-level diff) feed `revision.identify_source_text_changes` (LLM),
  which confirms/refines the candidate set and decides `drastic` (implies a
  scene should be added/removed) vs. a scoped `changed_scene_numbers` list.
  Fails safe to `drastic` on any malformed/inconsistent response; every
  returned scene number is sanitized against scenes that actually exist.
  Once scoped, the same `existing=`/`revise_keys=` machinery a direct
  scenes.json hand-edit uses takes over — no second, parallel
  implementation.
- **Casting data model:** each character entry has an `actor` block (the
  performer's own intrinsic features) and a `character` block (that actor
  aged/costumed into the role). Image generation renders the character
  only — exactly one image per character (`output/casting/<name>.png`,
  Gemini `image.model`, config-driven — currently `gemini-3.1-flash-image`)
  from `character.visual_prompt` — the
  identity seed for Veo; no actor render, no stock-photo lookup. Locations
  are cast the same way (`kind: "location"`, no `actor` layer, showing the
  space's own architecture instead of a person) — `casting` runs after
  `scenes` for this reason. Recurring props (2+ scenes anywhere in the
  story) are cast the same way too (`kind: "prop"`), with a very
  descriptive, isolation-strict `visual_prompt` since Veo has no
  cross-generation memory.
- **Scene rendering = image-to-video:** after storyboard + screenplay, each
  storyboard panel is rendered as a video clip via `i2v` (default backend
  Gemini Veo). The first panel of each scene seeds from the in-frame
  character's casting portrait (identity anchor); later panels chain from
  the previous clip's last frame for continuity (scene boundary = cut);
  idempotent by content hash of prompt+seed. Seed selection is boundary-
  aware (`pipeline._boundary_aware_seed`, shared by both the singleton- and
  merged-group render paths): a character "shot boundary" panel (cast
  changing) always seeds fresh from `_frame_char_anchor` rather than the
  previous panel's tail frame (which would still show the outgoing cast) —
  computed from the same `is_boundary` value (`_char_set_changed`) the
  adjacent `effective_prev_clip`/extend-eligibility check already uses, so
  the two can't disagree. A boundary panel with 2+ resolvable character
  portraits uses Veo's `reference_images` instead (up to 3, Veo's max) —
  mutually exclusive with seed/extend continuity for that call (Veo API
  constraint: a reference-image call can't also carry a start frame),
  falling back to the normal boundary-aware seed path on any failure.
  `continuity_mode: extend` (native video-to-video continuity, carries
  ambient/score audio across the cut) is only attempted when the cast
  hasn't changed panel to panel, and only at 720p. Every Veo prompt is
  verified against the guide before submission
  (`veo_guide.verify_prompt`, warnings only, never blocking); audio cues
  are constructed by dedicated `veo_guide` helper functions rather than
  hand-rolled inline. `rerender_panels` (the `revise` targeted-panel path)
  applies the same boundary-aware seed logic via its own
  `_resolve_start_frame(pnum, is_boundary)` — `is_boundary` computed once
  in `_render` and shared with `prev_clip_path` there too, so the two
  can't disagree, mirroring `_boundary_aware_seed`'s pattern.
- **Multi-segment timestamped Veo prompts (`reel/panel_grouping.py`,
  config `video.multi_segment_prompting`, default on):** consecutive
  panels WITHIN one scene sharing the same in-frame cast
  (`panel_grouping.group_panels`, same `char_set_changed` boundary
  predicate `_resolve_panel_references` uses) and fitting Veo's 8s max
  duration render as ONE Veo call via Google's timestamp-segment technique
  (`[00:00-00:02] ... [00:02-00:04] ...`, `veo_prompt.
  multi_panel_video_prompt`) instead of one call per panel — Subject/
  Context/Style stated once while Cinematography/Action vary per segment
  (`pipeline._render_panel_group`). Never crosses a `scene_number`
  boundary (cross-scene merging is a deferred follow-on, even though the
  data model allows consecutive scenes to share a location). A merged
  group's own call always uses the plain image-seed path (never `extend`,
  whose duration can't be pinned to the group's timestamps) and is forced
  off when `video.overlays.enabled` is true (no per-segment
  time-windowing) or the backend isn't Gemini/Veo; falls back to per-panel
  rendering on any exception. The manifest's per-panel `frame_record`s for
  a group all share one physical `clip` path and carry a `group_panels`
  sibling list — `_stitch_scene`/`_clips_in_order` dedupe by resolved path
  (`_dedupe_clip_paths`). **Known v1 limitation:** `rerender_panels`
  refuses (rather than attempting recursive group-aware cascading) if a
  targeted panel or its one-hop cascade target belongs to a merged group;
  escape hatch is re-rendering the whole scene via
  `_render_scene_frames(..., only_scenes={N})`.
- On this host prefer `--profile fast` (one model, no reload churn between
  agents).
- Update cadence lives in `scripts/update-models.sh` (pull + version-check
  + smoke test + log), wired weekly/monthly via `make install-cron`,
  runnable on-demand via `make update`.
- **Prompt-pitfall audit + DO-NOT/structure enforcement:** the long-context,
  multi-rule agent prompts (scenes/casting/soundscape/visuals/
  cinematography/screenplay/storyboard) each end with a "Before you
  respond, re-check against [data] above" reminder block restating their
  highest-stakes rules a second time (mitigates the lost-in-the-middle
  effect for prompts with a large data block ahead of them). Every agent
  prompt (creative and grader alike) states an explicit "DO NOT:"
  failure-mode list plus a "respond with ONLY a single JSON object
  matching EXACTLY this shape" structure-strictness instruction right
  before its JSON schema. Three real priority conflicts were found and
  fixed with explicit tie-breakers: cinematography's
  motif-development-vs-location-distinctness, screenplay's SOURCE OVER
  COVERAGE, and casting's invent-the-actor-vs-story-fidelity
  clarification.
- Version control: git, branch `main`.
