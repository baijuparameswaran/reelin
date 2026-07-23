# Progress

> What's true right now, and the append-only history of how it got there.
> Keep **Current state** current every session; append new entries to the
> TOP of **Session log** (newest first) — never rewrite older entries.
> Architecture reference lives in [ARCHITECTURE.md](ARCHITECTURE.md); the
> per-agent prompt reference is [AGENTS.md](AGENTS.md).
>
> Compacted 2026-07-16: the session log below was condensed from its
> original blow-by-blow form (verification steps, running test counts,
> superseded intermediate attempts) down to decision + rationale per dated
> entry, and stale/resolved "Current state" items were removed. History is
> preserved at the git revision prior to that compaction if deeper detail
> is ever needed.

## Current state
- **`--max-scenes` only restricts actual media-rendering stages** —
  casting-image generation and `scene_render`'s video generation. Every
  design/planning stage (screenplay, storyboard, soundscape, visuals,
  cinematography) always processes every scene regardless. Moodboard's
  `tiles` remain capped to it (a real media-generation step, unlike the
  moodboard's film-wide aesthetic fields).
- **Status:** the pipeline runs genre → structure/characters → moodboard →
  scenes/casting → soundscape/visuals/cinematography → screenplay →
  storyboard → render end to end, with a per-stage human-in-the-loop gate
  plus fidelity/genre scoring. Live-verified including a full video render +
  stitch to `output/video/movie.mp4` on the bundled sample story.
- **Veo prompts are assembled via the official five-part formula**
  (`[Cinematography]+[Subject]+[Action]+[Context]+[Style & Ambiance]`, per
  Google's Veo 3.1 guide) from structured data (casting.json, scene
  `visual_overview`, panel camera fields) rather than the storyboard agent's
  free-text `image_prompt`. Shared by both the main pipeline and the
  standalone `fountain.py` render path via `reel/veo_prompt.py`, so the two
  entry points can't drift apart. Depth-of-field is intentionally left
  unmanaged (whatever the source content implies, if anything).
- **Gemini image + video are live-verified** (key managed via
  `python -m reel.secrets`) — one portrait per character/location/prop via
  `gemini-2.5-flash-image`; scene clips via Veo (default
  `veo-3.1-fast-generate-preview`), stitched into `output/video/movie.mp4`.
  Multi-character Veo `reference_images` at cast boundaries and native
  `extend`-mode continuity are both implemented and config-toggleable.
- **Render steps (casting images, moodboard tiles, scene frames) are
  idempotent by a content hash of prompt+seed, not just file existence** —
  a HITL-feedback-revised prompt correctly re-renders instead of being
  skipped because the file already exists.
- **`tests/test_prompt_rules.py` is a real, committed, fully offline suite**
  (`make test`, no LLM/API calls, <1s) validating every agent prompt's
  conventions (sandwiching, DO-NOT lists, schema adherence) and the
  deterministic functions backing some of those rules — 336 tests as of the
  latest session (across the whole `tests/` suite, not just this one file).
- **Still shelved:** a formal end-to-end `tests/` suite stubbing every paid
  API for a full `pipeline.run()` drive-through — drafted once, then
  explicitly removed before being committed. This project's practice for
  that scope remains throwaway per-session stubs, not a committed suite.
- **Next up:** moodboard tile auto-render (opt-in); richer ingest
  (PDF/EPUB/.fdx); an edit / sound-mix / final-cut phase.
- **Known deferred issue:** at a character-boundary panel where
  reference-images end up NOT actually sent that call (disabled, fewer than
  2 resolvable portraits, or the API call fails), the single-image seed
  fallback can reuse the PREVIOUS panel's tail frame (the old cast) instead
  of a fresh anchor for the character(s) entering the new panel —
  `pipeline._render_scene_frames`'s seed selection isn't boundary-aware the
  way the adjacent extend-mode continuity check already is. Fix (if picked
  up): make `seed` boundary-aware via the same `_char_set_changed` check,
  falling back to `_frame_char_anchor` at a boundary panel.

## Session log
- 2026-07-23 (later) — Wired storyboard panels' `emotional_note` (per-shot
  creative direction — "the emotion this panel must evoke in the audience" —
  authored by the cinematography agent, falling back to visuals'/
  soundscape's scene-wide read) into the actual rendered Veo prompt for the
  first time, via new `veo_prompt.panel_action` folding it into the [Action]
  element as a short performance-direction clause ("conveying quiet dread"),
  used by both `five_part_veo_prompt` and `multi_panel_video_prompt`. Closes
  a real gap: this field was authored by three creative agents and shown at
  the review gate, but silently dropped before ever reaching the actual
  render. Deliberately scoped as the pipeline's one "director's freedom"
  injection point — interpretive performance direction only, never new
  source-bound facts (those stay scenes.py/screenplay.py's exclusive
  territory). New `tests/test_panel_action_emotional_note.py` (13 tests);
  349 tests total, still passing.
- 2026-07-23 — Added multi-segment timestamped Veo prompts
  (`reel/panel_grouping.py`, new; `veo_prompt.multi_panel_video_prompt`;
  `pipeline._render_panel_group`; config `video.multi_segment_prompting`,
  default on): consecutive storyboard panels WITHIN one scene sharing the
  same in-frame cast and fitting within Veo's 8s max duration are now
  rendered as ONE Veo call via Google's documented
  `[00:00-00:02] ... [00:02-00:04] ...` timestamp-segment technique, instead
  of one call per panel — fewer API calls, Subject/Context/Style stated
  once while Cinematography/Action vary per segment. A planning pass before
  implementation surfaced a load-bearing fact the naive design missed: only
  the plain image-seed Veo call actually honors a caller-chosen
  `duration_seconds` (reference-images hardcodes 8s, extend-mode has no
  duration parameter at all) — a merged group's own call is therefore
  always forced onto the plain-seed path, and is disabled entirely when
  `video.overlays.enabled` is true (no per-segment time-windowing exists for
  burned-in captions) or the backend isn't Gemini/Veo. Deliberately scoped
  to WITHIN one scene only (never crosses a `scene_number` boundary) after
  direct pushback that an early draft's "location is constant" reasoning
  was really an unstated assumption rather than a scoping decision;
  cross-scene merging (consecutive scenes sharing a location, which the
  data model already permits) was explicitly deferred as TBD, not
  implemented. `_char_set_changed` was relocated verbatim to
  `panel_grouping.char_set_changed` as the single source of truth (imported
  back into `pipeline.py` under its old name). `rerender_panels`'s targeted
  panel re-render now refuses (rather than silently corrupting a merged
  clip's shared-clip invariant) if a targeted panel or its one-hop cascade
  target belongs to a merged group — recursive group-aware cascading was
  scoped out of v1 as the highest-risk piece, with re-rendering the whole
  scene as the working escape hatch. New `tests/test_panel_grouping.py` (18
  tests, offline); 336 tests total, still passing.
- 2026-07-16 — Added explicit "best in the field" persona framing to every
  agent's `SYSTEM` prompt (e.g. "one of the most sought-after film casting
  directors working today"), and added a "DO NOT:" failure-mode list plus a
  "respond with ONLY a single JSON object matching EXACTLY this shape"
  structure-strictness instruction to every agent `PROMPT` (creative and
  grader alike) — reinforced a second time in the closing reminder block
  for the seven sandwiched prompts. Updated `tests/test_prompt_rules.py`'s
  sandwich-tail assertions to match; 318 tests, still passing, offline.
  Also compacted PROGRESS.md and ARCHITECTURE.md (this pass) per direct
  instruction to keep the auto-loaded context tidy — condensed the
  session log and convention bullets, removed stale/resolved items.
- 2026-07-15 (later) — Removed the "MINIMIZE SCENE COUNT" bias from
  scene/shot segmentation, replacing it with a director's-eye completeness
  principle, per direct instruction that it was under-serving the story.
  `scenes.py` rule 9 rewritten to "CAPTURE THE STORY FULLY"; `cinematography.py`'s
  shot-count rule and `duration_budget.py`'s guidance text reworded to match
  (a scene-count target is now a soft, secondary guide, never a reason to
  merge distinct beats). Scoped-revision structure-alignment rules (keep
  count stable within a revision's `revise_keys`) were deliberately left
  untouched — a separate, unrelated mechanism.
- 2026-07-15 — Brought `fountain.py`'s standalone `render` path to prompt-
  construction parity with the main pipeline, and completed the prior
  architecture review's three "structural" recommendations. Extracted the
  five-part Veo prompt formula into `reel/veo_prompt.py` (now shared by
  `pipeline.py` and `fountain.py`, no longer duplicated); reworked
  `fountain.to_storyboard` to build a storyboard-shape-compatible board
  (location/visual_overview/audio_overview/multi-character
  `characters_in_frame`) instead of a sparser approximation — this also
  surfaced and fixed a real case-sensitivity bug where ALL-CAPS Fountain
  speaker cues didn't match Title-Case casting.json names. Added
  `llm.validate_config` (lightweight config/models.yaml shape check,
  non-blocking) and `reel/spend.py` (estimated $ spend from
  `gemini_api.log`, `python -m reel.cli spend`).
- 2026-07-14 (later 2) — `storyboard._panel_characters_in_frame` now
  narrows `characters_in_frame` to whoever a panel's own action/dialogue
  text actually names (word-boundary/dialogue-speaker matching), instead of
  defaulting to the full scene cast for any wide/establishing shot — the
  source-level fix complementing the render-time narrowing added below.
  Also fixes Subject-text and dialogue-attribution accuracy, not just Veo
  reference-image selection.
- 2026-07-14 (later) — Veo reference/seed images for a panel are now scoped
  to that panel's own action/dialogue-relevant characters
  (`pipeline._panel_relevant_characters`), not the raw (often
  over-inclusive) `characters_in_frame` list — avoids wasting the
  3-reference budget or seeding from a character absent from that specific
  panel. Falls back to the full list only for a genuine group/establishing
  shot naming no one individually.
- 2026-07-14 — Veo Subject text now references an attached seed/reference
  image ("NAME (as shown in the reference image)") instead of re-describing
  a character in full, whenever that character's actual portrait is being
  sent in the same call (`pipeline._anchored_character_names`, reverse-maps
  sent image paths against casting.json). An unanchored character (no
  resolvable portrait) still gets the full locked description.
- 2026-07-12 (later) — `--no-render` now still logs every Gemini/Veo call's
  actual request parameters (`dry_run` param threaded through the full
  render call chain, `outcome=skipped(no-render)`) — only the network/SDK
  call itself is skipped, not the prompt-building or logging.
- 2026-07-12 — `gemini_api.log` now records the full request parameters
  (`params={...}` JSON) for every real Gemini/Veo call, not just that a
  call happened — added to all four real API-call functions.
- 2026-07-11 (later 9) — `revise`'s render-skip-by-default is now two
  INDEPENDENT toggles, `render_images`/`render_video` (was one combined
  flag) — `--render-images`/`--render-video` CLI flags, or
  `render images on/off`/`render video on/off` in the interactive loop.
- 2026-07-11 (later 8) — `revise`'s downstream cascade now gets the SAME
  per-stage HITL review gate a fresh `pipeline.run()` uses (reuses
  `pipeline._gated`, not a second implementation) — fidelity/genre scoring,
  feedback-driven re-run, and view-to-edit are all now available during a
  revision round, not just an upfront confirm.
- 2026-07-11 (later 7) — Diagnosed a live case of scenes.json growing 1→3
  scenes during `revise` (a source-text edit's drastic fallback correctly
  added scenes for a genuinely new story beat) and closed a real prompt
  gap: the drastic full-regen path gave `scenes.segment_scenes` no revision
  context at all, so an edited passage's own paragraph breaks could bias it
  toward over-fragmenting despite rule 9. Added `_revision_reminder_note`
  (states the prior scene count, reinforces rule 9 + a "paragraph break
  isn't automatically a scene break" caveat) for this specific
  unscoped-drastic-regen case.
- 2026-07-11 (later 6) — Scoped-revision prompts (scenes/cinematography/
  screenplay/storyboard) now explicitly tell the model the CURRENT
  scene/shot/panel count during a scoped revision round, framing any count
  change as a deliberate exception — the model-visible half of enforcement;
  the entry below already did the code/merge-level half.
- 2026-07-11 (later 5) — `revision_merge.merge_by_key` no longer lets a
  scoped revision's model response silently ADD an unrequested new key — an
  addition is only accepted if the caller's `keys_to_replace` explicitly
  authorized it, closing the last structural-drift gap after the two
  entries below.
- 2026-07-11 (later 4) — `revise` now GUARANTEES every scene-keyed artifact
  stays aligned with scenes.json after every round, UNCONDITIONALLY
  (`cli._align_scene_keyed_stages`, called at the end of every revision
  round regardless of which stage was edited) — supersedes the narrower
  first-pass fix below, per direct pushback that the fix needed to not
  depend on the edited stage's own downstream cascade.
- 2026-07-11 (later 3) — First-pass fix for the same alignment bug
  (superseded by the entry above): only backfilled a scene-keyed stage when
  it happened to be in the edited stage's own downstream closure.
- 2026-07-11 (later 2) — Scoped revisions now merge FIELD BY FIELD within a
  targeted entry (`revision_merge.merge_fields`), not just entry-by-entry —
  a field the model reworded without real content change keeps its old
  value, so an incidental rewording of an untouched field no longer leaks
  through.
- 2026-07-11 (later) — Added `--no-render` to the main
  `python -m reel.cli SOURCE.txt` full-pipeline command (mirrors `revise`'s
  existing render-skip toggle) — inherited across `--resume` the same way
  `--max-scenes`/`--profile` already are.
- 2026-07-11 — `revise` now always operates as `--max-scenes all`
  regardless of the original run's cap (scoping is via `revise_keys`, not
  `max_scenes`, so the cap was a redundant, sometimes-harmful restriction);
  added upfront "evaluation" and per-stage "regenerating/full regen/nothing
  to regenerate" printouts so the operator sees both what was identified
  and what each stage is doing about it.
- 2026-07-10 (later 12) — `continuity_mode: extend` now only fires when a
  panel's in-frame characters match the immediately preceding panel's
  (`pipeline._char_set_changed`, shared with the reference-image boundary
  logic) — prevents extending audio/visual continuity from the wrong cast
  across a character-boundary panel.
- 2026-07-10 (later 11) — Implemented multi-character Veo `reference_images`
  at scene/cast boundaries (Increment 5 of the 2026-07-04 locations-casting
  work, previously paused): every character in frame at a boundary panel
  with 2+ resolvable casting portraits gets its own identity reference (up
  to Veo's 3-image cap), instead of only the first character getting a
  single-image seed. Falls back to the normal seed path on any failure. New
  config `video.multi_character_references` (default on).
- 2026-07-10 (later 10) — `revise` can now DELETE and ADD scenes directly in
  scenes.json (previously only modify) — deletion strips the scene from
  every downstream scene-keyed artifact + the video manifest (via the same
  deterministic `strip_orphan_scenes` primitive a fresh run already
  self-heals with); addition re-sorts the merged list into narrative order.
  Scoped to scenes.json only — other scene-keyed artifacts still can't
  add/remove scenes directly.
- 2026-07-10 (later 9) — A source-TEXT edit in `revise` is now SCOPED to the
  scenes it actually affects (deterministic diff pre-filter + paragraph-level
  unified diff + an LLM confirmation call on `agent_profiles.revision:
  quality_high`), instead of always falling back to a full downstream
  regen. Fails safe to `drastic` (full regen) on any malformed/ambiguous/
  scene-adding response.
- 2026-07-10 (later 8) — `revise` now skips casting regeneration and all
  image/video rendering by DEFAULT (the only stages that cost real API
  spend), opt-in via `--render`/`render on`/`render off`.
- 2026-07-10 (later 7) — Fixed a real pre-existing bug: `soundscape.json`
  never had a `score_direction` field even though `storyboard.py` always
  read `bundle.audio.score_direction` — added the schema field. (A larger
  architectural change explored this session — character-first pipeline
  reordering, character-introduction scene boundaries, a "subtle score"
  toggle — was built, reviewed, then explicitly discarded per instruction;
  only this genuine bug fix was kept.)
- 2026-07-10 (later 6) — `revise` now inherits the original run's
  profile/max_scenes/target_duration/genre+moodboard steering instead of
  reverting to bare defaults — `revise` is a separate process invocation,
  so none of this carried over automatically before this fix (most
  consequential gap: creative-direction steering was silently absent for
  every revised stage).
- 2026-07-10 (later 5) — Fixed a real reported bug: a JSON-schema EXAMPLE in
  casting.py's prompt ("Marcel's pocket watch") leaked into actual model
  output, since "Marcel" happened to collide with a real character in the
  same story. Replaced with unambiguous bracket-style placeholders across
  casting.py/scenes.py; added a permanent regression test asserting no
  invented-name examples appear in any rendered prompt.
- 2026-07-10 (later 4) — `make demo`/`make run` now depend on `make test`,
  so a failing prompt-rule test blocks a pipeline invocation before it
  spends time/API quota.
- 2026-07-10 (later 3) — Added `tests/test_prompt_rules.py` (committed,
  offline, no LLM/API calls, `make test`) — validates every agent prompt
  actually follows this project's prompting conventions and that
  deterministic enforcement functions do what they claim, replacing ad-hoc
  per-session smoke tests with a permanent suite.
- 2026-07-10 (later 2) — Recurring PROPS now get a locked, very descriptive
  `visual_prompt` (same identity-consistency treatment as characters/
  locations) so the same object renders consistently across scenes —
  `casting._prop_entries` casts any prop appearing in 2+ scenes anywhere in
  the story.
- 2026-07-10 (later) — Added scene-level `props` (scenes.py rule 10,
  source-grounded) and fixed a real gap: `visuals.py`'s `key_props` never
  actually reached a rendered Veo frame — wired through `storyboard.py`'s
  `visual_overview.key_props` and `pipeline._panel_context`'s Context
  section.
- 2026-07-10 — Scene segmentation was changed to actively MINIMIZE scene
  count (a cost-control bias, **fully reversed in the 2026-07-15 entry
  above**) and each scene gained a deterministic `source_excerpt`+
  `word_count` (the actual contiguous portion of source text it covers,
  beyond the short `source_line` anchor) — this part remains current.
- 2026-07-09 (later 7) — Removed the hardcoded "at least 2 shots per scene"
  floor (both the cinematography.py prompt rule and
  `duration_budget.suggest_shots_per_scene`'s math) — a scene now only
  needs a minimum of 1.
- 2026-07-09 (later 6) — Prompt-pitfall audit: added "Before you respond,
  re-check against [data] above" reminder blocks (sandwiching) to the seven
  long-context, multi-rule prompts, and fixed three real priority conflicts
  with explicit tie-breakers (cinematography's motif-vs-location-
  distinctness, screenplay's SOURCE OVER COVERAGE, casting's invent-actor-
  vs-story-fidelity clarification).
- 2026-07-09 (later 5) — Added `--target-duration` (default 45s):
  engine-independent scene/shot-count planning guidance
  (`reel/duration_budget.py`) plus each rendered clip's actual requested
  duration, translated per-backend (Veo's `_veo_nearest_valid_duration`
  rounds to its valid 4/6/8s set; diffusers/http backends honor it as a
  frame count).
- 2026-07-09 (later 4) — Fidelity agent gained a deterministic (no LLM),
  self-healing scene-structure alignment check
  (`fidelity.check_scene_alignment`/`strip_orphan_scenes`) — distinct from
  the qualitative story-fidelity check — wired into `pipeline.run_group` to
  auto-correct a scene-keyed stage that drifts from scenes.json before the
  operator ever sees the gate. Caught and fixed a real bug along the way:
  `screenplay.py` used `setdefault` instead of force-setting
  `scene_number`, so a wrong model echo was trusted as-is.
- 2026-07-09 (later 3) — Storyboard construction made fully deterministic
  (no LLM call) in the default path — `_build_scene_board` constructs
  every field from upstream artifacts directly; the LLM path
  (`_llm_generate_scene`) now only runs when `feedback` (a directed
  creative note) is given.
- 2026-07-09 (later 2) — Hardened the storyboard merge: enforces
  scenes.json's structure deterministically (corrects/records a wrong or
  dropped per-scene response) and closed seven real prompt gaps where
  bundle data (visual_filter, key_props, silence, sound_events,
  transition_to_next, and critically `voiceover` as separate from
  `dialogue`) was sent but never actually grounded to an output field.
- 2026-07-09 (later) — `--max-scenes`/`--profile`/`--target-duration` now
  survive `--resume` instead of silently resetting to argparse defaults
  (`cli._save_run_params`/`_load_run_params`, `output/run_params.json`) — a
  real correctness bug, not just UX, since an unfinished `--max-scenes all`
  run resumed bare would have only rendered scene 1.
- 2026-07-09 — Fixed a real character-name mismatch between `scenes` and
  `characters` (found via live output inspection): the two agents ran
  independently and could name the same person differently ("Woman" vs.
  "Young Woman"), silently breaking every name-keyed lookup including
  casting-image rendering. Fixed with a canonical-names prompt block plus a
  deterministic reconciliation fallback (`scenes._reconcile_character_names`).
- 2026-07-08 (later 4) — The revision loop is now offered automatically
  right after a full pipeline run completes (`cli._offer_revise`), not just
  via the standalone `revise` command; added an explicit `exit` synonym for
  `quit`. Found and fixed a real bug: the stage-picker menu listed `ingest`
  and `source` as if they were two different artifacts.
- 2026-07-08 (later 3) — Added the revision agent
  (`python -m reel.cli revise`): hand-edit any completed stage's JSON (or
  the raw source text) via `$EDITOR`, then selectively re-run only what's
  actually affected downstream (`reel/artifact_diff.py` deterministic diff,
  `reel/revision_merge.py`'s `merge_by_key` splice primitive,
  `reel/agents/revision.py`'s advisory ripple suggestions + deterministic
  identity-drift heuristic). Panel-level video re-render cascades exactly
  one hop forward then stops, a deliberate cost bound.
- 2026-07-08 (later 2) — Added a session-identity concept
  (`reel/session.py`) spanning the whole story-to-video workflow — one run
  gets a persistent id (`output/session.json`), reattached across
  `--resume`/standalone-stage/`revise` invocations, tagging
  `gemini_api.log` and per-scene prompt logs so they stay attributable to
  the run that produced them.
- 2026-07-08 (later) — Made Veo `continuity_mode: extend` as robust as the
  seed path (retry-with-backoff, `.error` inspection) and fixed it to skip
  attempting extend (not clamp resolution) when the configured resolution
  isn't 720p. Fixed a real `--max-scenes` inconsistency: `screenplay`'s
  default cap (3) was the one design/planning stage still wrongly
  restricted by the render-scoped cap — changed to `None` (draft every
  scene by default).
- 2026-07-08 — Rebuilt Veo prompt assembly around the official five-part
  formula (confirmed via a live fetch of Google's official guide) built
  from structured data rather than the storyboard agent's free-text
  `image_prompt`; removed all depth-of-field (deep/shallow focus) handling
  per instruction, leaving it to source content; added persistent
  `gemini_api.log` and per-scene Veo prompt logs.
- 2026-07-04 — Locations are now cast alongside characters (`casting.py`'s
  `kind: "location"` entries, no `actor` layer, inverse isolation rule) —
  `casting` now runs after `scenes`, not concurrently, since it needs the
  scene→location mapping. Wiring multi-character/location references into
  Veo's `reference_images` (Increment 5) was deliberately paused this
  session — later completed 2026-07-10 (later 11).
- 2026-07-03 — Actually installed the `google-genai` SDK (previously
  silently absent — every "verified" Gemini/Veo call had gone through the
  raw REST fallback) and fixed the real API-shape bugs it surfaced (plural
  `generate_videos`/`GenerateVideosConfig`, keyword-only
  `Video.from_file`/`files.download`, `person_generation` fixed by
  generation mode not model tier, `enhance_prompt` removed as
  non-existent). Moved Veo audio-cue construction into `veo_guide.py`
  helper functions. Bumped `agent_profiles.scenes` to `quality` after a
  live 4b-vs-8b comparison showed 4b dropping/mis-numbering scenes.
- 2026-07-02 — Render steps (casting images, moodboard tiles, scene frames)
  are now idempotent by a content hash of prompt+seed, not just file
  existence, so a HITL-feedback-revised prompt correctly re-renders instead
  of silently staying stale. Added Veo audio-continuity best practices
  (ambient/score separation, "no background music" default assertion,
  dry-acoustics anchoring, dialogue cues folding in each character's voice
  description) and `gemini.extend_video()` (native video-to-video
  continuity, experimental).
- 2026-06-30 — Aligned all Veo prompts to the official guide's five-element
  order and three audio cue types across both render paths (pipeline.py and
  fountain.py); added `veo_guide.verify_prompt()` (checks the five required
  elements before every Gemini call); removed dead code (`stock.py`, img2img
  chain, unused `models.py` wrappers).
- 2026-06-27 — Disabled reasoning traces (`runtime.think: false`, ~2x
  faster) and added GPU-aware model selection (`reel/llm.py`'s
  `gpu_vram_mb`/`can_run_model`, `manifest.py --runnable-only`/`--hardware`).
- 2026-06-25 — Upgraded `thinking`/`quality_high` profiles to
  qwen3:14b/qwen3:30b (GPU+CPU split, confirmed viable with 23GB system
  RAM) after confirming the Intel NPU is inaccessible from WSL2. Enabled
  full GPU offload after identifying the snap Ollama package's confinement
  was blocking `/dev/nvidia*` (fixed via the official installer) —
  `num_gpu: -1`, `max_parallel_agents` 1→2, timeout 600s→120s.
- 2026-06-22 — Added the genre and moodboard cross-cutting agents (steer
  every stage via `llm.set_direction`, genre also grades per-stage
  alignment); enriched storyboard/screenplay to fuse full upstream detail;
  added the standalone `render` command and Gemini video-call
  retry/backoff.
- 2026-06-21 — A dense day of foundational work: switched image/video
  generation to the Gemini API (image gen scoped to character portraits
  only, dropping the old stock-photo/img2img chain); added the unified
  `reel/models.py` model-abstraction + provider policy (Gemini only for
  image/video, all text on open models); added `reel/stages.py` (per-stage
  registry + standalone `stage NAME` invocation); added per-stage +
  pipeline-aggregate story-fidelity scoring, shown at the review gate;
  capped scenes (not shots) by `--max-scenes`; added `reel/fountain.py`
  (Fountain parser) and structured screenplay shots/dialogue/voiceover.
- 2026-06-20 — Added pluggable text-to-image rendering (`imagegen.py`) and
  image-to-video scene rendering (`i2v.py`, continuity-chained clips);
  restructured casting into separate `actor`/`character` blocks; added
  Openverse stock-photo lookup + img2img identity chain (later removed in
  the 2026-06-21/06-30 sessions).
- 2026-06-17 — Extended the pipeline well past the thin slice: added
  casting, soundscape, visuals, cinematography, and storyboard agents;
  added the human-in-the-loop review gate (`gate.py`); made independent
  stages run concurrently; fixed a hard-coded socket timeout by streaming
  LLM generation and checkpointing each approved stage; added pause/resume
  (`stop` at a gate, `--resume` to continue).
- 2026-06-16/17 — Built iteration 1: the full screenplay-material agent
  pipeline (ingest→structure/characters→scenes→screenplay), a
  model-agnostic Ollama client with profile fallback, and the model-update
  cadence script.
- 2026-06-16 — Initialized the repo and continuity scaffolding (git,
  CLAUDE.md).
