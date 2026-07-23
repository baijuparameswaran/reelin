# Architecture

> How `reel` is built: stack, layout, hardware constraints, and the
> established conventions/decisions behind each subsystem. Stable reference
> material — changes here are architectural, not day-to-day progress (see
> [PROGRESS.md](PROGRESS.md) for that) or per-agent prompt detail (see
> [AGENTS.md](AGENTS.md)).
>
> Compacted 2026-07-16: condensed from a much longer, narrative form (deep
> bug-fix-by-bug-fix justification, verification steps, running test
> counts) down to what each convention IS and why, per direct instruction to
> keep the auto-loaded context tidy. That history still lives, in
> condensed form, in PROGRESS.md's session log.

## Stack & layout
- **Runtime:** Python 3.12 (`.venv/`), local LLMs via **Ollama**. Core 3rd-party
  dep is PyYAML; the Ollama client is stdlib `urllib` (in `reel/llm.py`). Image
  rendering adds **optional** deps (`diffusers`/`torch`/etc., see
  `requirements-image.txt`) — the text pipeline runs without them.
- `reel/` — package:
  - `models.py` — provider policy + open-text front door; graders import
    this, `text()` is the only public function, always routes to Ollama
    with `steer=False` so graders stay neutral.
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
- **Gemini API key** (for image/video): read from env `GEMINIAPIKEY` (or
  `GEMINI_API_KEY`/`GOOGLE_API_KEY`). Without it, image/video stages no-op
  gracefully with a hint; the text pipeline is unaffected.
- `scripts/` — `update-models.sh` (cadence), `install-cron.sh`,
  `model-updates.log`.
- `samples/` — bundled test story. `output/` — generated artifacts
  (gitignored).
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
  text path exists by design.
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
  governing SYSTEM/PROMPT template. Runs once automatically before the
  operator sees the gate; a `needs_improvement` verdict triggers exactly one
  re-run via the same `feedback` mechanism a human's gate feedback uses
  (never recursive). The pre-critique raw result is preserved as
  `output/<stage>.0.json`. Toggle via config `critique.enabled` (default
  true); best-effort, never blocks. Scoped to `pipeline.run()` only, not the
  standalone `revise` gate.
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
  through scene/shot-count planning guidance (soft, secondary to each
  agent's own coverage rules) and each clip's requested render duration
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
  layer (`veo_prompt.panel_action`):** each storyboard panel's
  `emotional_note` — per-shot creative direction ("the emotion this panel
  must evoke") authored by the cinematography agent (falling back to
  visuals'/soundscape's scene-wide read), consolidated by
  `storyboard._build_scene_board` — is folded into the Action clause as a
  short performance-direction phrase ("Alice reaches for the doorknob,
  conveying quiet dread"), deduped against text already present in the
  action. Before this existed, `emotional_note` was authored by three
  creative agents and shown to the operator at the review gate, but
  silently dropped before ever reaching the actual Veo prompt. This is the
  one deliberate point where the rendering layer adds interpretive
  performance direction on top of an already-established beat — it never
  changes WHAT happens (scenes.py/screenplay.py stay the sole source-bound
  authority for that), only HOW it's performed/felt, the same way a
  director notes a line reading without touching the script.
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
  Gemini `gemini-2.5-flash-image`) from `character.visual_prompt` — the
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
  idempotent by content hash of prompt+seed. A "shot boundary" panel (scene
  start, or the in-frame cast changing) with 2+ resolvable character
  portraits uses Veo's `reference_images` instead of a single seed, so
  every character in frame gets an identity lock (up to 3, Veo's max)
  rather than just one — mutually exclusive with seed/extend continuity
  for that one call, falling back to the normal seed path on any failure.
  `continuity_mode: extend` (native video-to-video continuity, carries
  ambient/score audio across the cut) is only attempted when the cast
  hasn't changed panel to panel, and only at 720p (a genuine Veo
  constraint). Every Veo prompt is verified against the guide before
  submission (`veo_guide.verify_prompt`, warnings only, never blocking);
  audio cues are constructed by dedicated `veo_guide` helper functions
  rather than hand-rolled inline.
- **Multi-segment timestamped Veo prompts (`reel/panel_grouping.py`,
  config `video.multi_segment_prompting`, default on):** consecutive
  storyboard panels WITHIN one scene that share the same in-frame cast
  (`panel_grouping.group_panels`, using the same `char_set_changed`
  boundary predicate `_resolve_panel_references` already uses) and whose
  summed duration fits Veo's 8s max are rendered as ONE Veo call via
  Google's documented timestamp-segment technique
  (`[00:00-00:02] ... [00:02-00:04] ...`, built by
  `veo_prompt.multi_panel_video_prompt`) instead of one call per panel —
  fewer API calls, and Subject/Context/Style are stated once while
  Cinematography/Action vary shot to shot (`pipeline._render_panel_group`).
  Deliberately scoped to WITHIN one scene only — never crosses a
  `scene_number` boundary, even though the data model allows consecutive
  scenes to share a location (see `panel_grouping.py`'s module docstring);
  cross-scene merging is a deferred follow-on, not implemented. A merged
  group's own call always uses the plain image-seed path (never
  `continuity_mode: extend`, whose fixed ~7s/call has no duration parameter
  to pin against the group's timestamps) and is forced off entirely when
  `video.overlays.enabled` is true (overlay burn-in has no per-segment
  time-windowing) or the backend isn't Gemini/Veo. Falls back to per-panel
  rendering automatically on any exception. The manifest's per-panel
  `frame_record`s for a merged group all share one physical `clip` path and
  carry a `group_panels` list of sibling panel numbers — `_stitch_scene`/
  `_clips_in_order` dedupe by resolved path (`_dedupe_clip_paths`) so a
  shared clip isn't concatenated multiple times. **Known v1 limitation:**
  `rerender_panels` (the `revise` CLI's targeted panel re-render) refuses
  with a clear error if a targeted panel — or its one-hop cascade target —
  belongs to a merged group, rather than attempting recursive group-aware
  cascading; the escape hatch is re-rendering the whole scene via
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
