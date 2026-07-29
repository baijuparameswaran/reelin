# Progress

> What's true right now, and the append-only history of how it got there.
> Keep **Current state** current every session; append new entries to the
> TOP of **Session log** (newest first) — never rewrite older entries.
> Architecture reference lives in [ARCHITECTURE.md](ARCHITECTURE.md); the
> per-agent prompt reference is [AGENTS.md](AGENTS.md).
>
> Compacted 2026-07-16 and again 2026-07-24: the session log below was
> condensed from its original blow-by-blow form (verification steps,
> running test counts, superseded intermediate attempts) down to decision +
> rationale per dated entry, and stale/resolved "Current state" items were
> removed — including two obsolete function-path references in AGENTS.md
> (functions relocated to `veo_prompt.py` during the 2026-07-15 extraction
> but still cited under their old `pipeline._` names) and a stale default
> model name (`gemini-2.5-flash-image` → `gemini-3.1-flash-image`, per
> `config/models.yaml`) found during this pass. History is preserved at the
> git revision prior to each compaction if deeper detail is ever needed.

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
  `image.model` (currently `gemini-3.1-flash-image`); scene clips via Veo
  (default `veo-3.1-fast-generate-preview`), stitched into
  `output/video/movie.mp4`. Multi-character Veo `reference_images` at cast
  boundaries, native `extend`-mode continuity, and consecutive same-cast
  panels merged into one multi-segment timestamped call are all
  implemented and config-toggleable.
- **Render steps (casting images, moodboard tiles, scene frames) are
  idempotent by a content hash of prompt+seed, not just file existence** —
  a HITL-feedback-revised prompt correctly re-renders instead of being
  skipped because the file already exists.
- **`tests/test_prompt_rules.py` is a real, committed, fully offline suite**
  (`make test`, no LLM/API calls, <1s) validating every agent prompt's
  conventions (sandwiching, DO-NOT lists, schema adherence) and the
  deterministic functions backing some of those rules — 477 tests as of the
  latest session (across the whole `tests/` suite, not just this one file).
- **Still shelved:** a formal end-to-end `tests/` suite stubbing every paid
  API for a full `pipeline.run()` drive-through — drafted once, then
  explicitly removed before being committed. This project's practice for
  that scope remains throwaway per-session stubs, not a committed suite.
- **TBD — `duration_budget.suggest_shots_per_scene` is a conserved total.**
  The same anchor problem `suggest_scene_target` was removed for, one level
  down: `est_total_shots` is pinned at `target_seconds / 6` and then DIVIDED
  by scene count, so more scenes mechanically means fewer shots each (~8
  shots for the whole film at the 45s default, regardless). The scene-count
  fix therefore halved this stage's per-scene budget as a side effect —
  scenes is now told to cover the story fully while cinematography is told to
  spread a fixed shot pool across whatever scenes produced, so the two stages
  pull against each other. Fix shape: a per-scene FLOOR derived from coverage
  needs rather than a quotient of a fixed pool, with total runtime falling
  out of that. Deliberately NOT bundled with the scene-count removal so that
  change could be measured on its own — and worth A/B-ing the same way rather
  than assuming, because unlike scene count, shot count has a real per-clip
  cost: every shot becomes a panel and every panel a Veo call.
- **Next up:** the shots-per-scene TBD above; moodboard tile auto-render
  (opt-in); richer ingest (PDF/EPUB/.fdx); an edit / sound-mix / final-cut
  phase.

## Session log
- 2026-07-28 — `make setup` now pulls the Ollama models too, not just the venv
  (`SKIP_MODELS=1` opts out; `make setup-models` still runs that half alone).
  It delegates to `scripts/update-models.sh --no-test` — the same script
  `make update` uses — rather than keeping the weaker parallel pull loop that
  was in the Makefile, so setup inherits the daemon start, Ollama version gate,
  hardware-fit filtering, and per-model failure tolerance for free. A missing
  `ollama` binary prints the official-installer one-liner (with the snap
  caveat) and skips instead of failing, since installing it needs sudo.
  Fixed a real bug this surfaced: `manifest.models()` walked every profile
  including the HOSTED `frontier` tier, so both `make setup-models` and
  `scripts/update-models.sh` had been feeding `gemini-3.6-flash` — a Gemini
  model name, not an Ollama tag — straight to `ollama pull`. `--runnable-only`
  didn't catch it either: `can_run_model` returns True for an unrecognized tag
  by design ("never silently drop a model we have no data on"). Now filtered on
  `provider != "ollama"`; nothing is lost, since a hosted profile's
  `fallback_profile` is itself a profile in the same loop.
  `tests/test_manifest.py` (6 tests, 5 confirmed to fail without the fix,
  including one guarding the real config/models.yaml so a future hosted profile
  can't leak in); 477 total.
- 2026-07-28 (later) — `make setup` now also CONFIGURES which model each
  profile uses, from detected hardware, before pulling anything
  (`reel/hardware_config.py`, `make hardware-config`, `python -m
  reel.hardware_config` to preview). `llm.resolve_model` already adapts at run
  time, but only among models already PULLED — too late to help the step that
  decides what to pull, which is why `config/models.yaml`'s 8 GB-VRAM tuning
  (`quality_high: qwen3:30b`, a 19 GB download) reached every host verbatim.
  Deltas go to a GITIGNORED `config/models.local.yaml`, deep-merged field by
  field over the tracked file by `llm.config` — models.yaml is hand-tuned and
  heavily commented against one machine, so rewriting it in place would
  destroy that commentary and leave every clone dirty. Two deliberate limits:
  it only picks from a profile's own `model`+`fallbacks` ladder (never invents
  a tag), and it only writes when the tracked choice does NOT fit — on this
  host it writes nothing, so the deliberate choices a naive size check gets
  wrong (30b MoE over the same-size 32b dense, ~6x faster) stand. Scope is the
  `model` field only, per instruction: `fallbacks` are already fit-filtered at
  run time by `resolve_model`, and `num_ctx` carries per-model VRAM reasoning
  that `local_max_chars` derives the source budget from.
  Two real bugs found by running it rather than by reading it: (1) my first
  version judged against the MERGED config, so an override already on disk
  became the baseline and a host that later grew could never be restored — now
  judged against new `llm.base_config()`, with `get_profile(name, cfg=None)`
  taking the pre-overlay view; (2) inserting the overlay code above `config()`
  silently stole its `@lru_cache(maxsize=1)` decorator, which the test suite
  caught as an unhashable-dict TypeError. Verified end to end against the real
  config: simulating a no-GPU/8 GB host cuts the pull list from 4 models
  (~35 GB) to 2 (~7.6 GB) with `options`/`fallbacks` preserved, and re-running
  on this host removes the overlay again.
  `tests/test_hardware_config.py` (20 tests, incl. the upgrade-path
  regression); 497 total.
- 2026-07-25 (later 8) — Removed `duration_budget.suggest_scene_target`, the
  last and largest cap on scene count. It turned `--target-duration` into a
  literal range ("roughly 2-4 scenes" at the 45s default) interpolated into
  the END of the scenes prompt's opening sentence — the most salient position
  in the prompt, directly after two anti-inflation warnings — so a concrete
  number was the last thing the model read before the story, and it beat the
  CAPTURE THE STORY FULLY rules forty lines below. That let a runtime default
  decide how much of the story got filmed, and contradicted the project's own
  convention that a budget bounds RENDERING (`--max-scenes`) while every
  design stage processes everything. Runtime still drives shots-per-scene and
  each clip's requested duration; it no longer sets scene count.
  `segment_scenes`' own coverage-first default is now numberless, and the
  prompt's opening was restructured so that framing LEADS and the
  anti-inflation caveat trails as "the only limits on scene count are real
  ones". `cli._duration_kwargs` no longer returns a `target` for `scenes`;
  two tests in `test_revise_inheritance.py` asserted the old contract and
  were rewritten. MEASURED, same story (sample_story.2.txt, 2,612 chars) and
  same model (gemini-3.6-flash), one run per arm: the old numeric anchor
  produced 4 scenes, the coverage-first wording 9 — and the 4 sits squarely
  inside the "roughly 2-4 scenes" the anchor asked for, while the 9 matches
  the 8-10 range this model gave on this story in the earlier model bench,
  which had no scene-count target at all. So the anchor was being followed
  almost literally rather than treated as the "soft, secondary" hint its own
  wording claimed. Both arms had 0 dropped scenes, so the extra scenes are
  source-anchored, not invented. `suggest_shots_per_scene` has the same
  conserved-total flaw one level down and was deliberately left alone so this
  change could be measured in isolation — recorded as an explicit TBD in
  Current state above and in that function's own docstring, with the fix
  shape and the reason it needs its own A/B (shot count, unlike scene count,
  costs a Veo call per unit). 471 tests.
- 2026-07-25 (later 4) — Reworked the hosted frontier tier from Anthropic to
  GEMINI, per direct instruction ("claude may need to setup at server side ..
  roll back all changes specific to claude, instead use gemini as backend for
  frontier and use the current key"). Rolled back `reel/anthropic_text.py`,
  `requirements-frontier.txt`, and the provider-parameterized `secrets.py`
  (back to Gemini-only — with one provider there's no second key to manage);
  kept the profile-level `provider`/`fallback_profile` plumbing, the
  `models.local_profile` grader guard, and the dispatch in `llm.generate`,
  since none of that was Anthropic-specific. New `gemini.generate_text`
  reuses the module's existing key, headers, `_post` retry/backoff, and
  `gemini_api.log`, so text calls need no new credential, no SDK, and no
  second log format for `spend.py`. This makes Gemini-for-text a deliberate,
  narrow exception to the previously absolute "no Gemini text path exists by
  design" rule.
  Model chosen EMPIRICALLY rather than assumed: enumerated what the key can
  actually reach (41 text-capable models; `gemini-2.5-pro` is listed but
  404s on `generateContent`), then benched the real scenes prompt on the
  bundled samples — `gemini-3.6-flash` 8-10 scenes / 26-36s / 4-of-4 parsed,
  `gemini-3.1-pro-preview` 9-13 scenes / 42-70s / 3-of-4 parsed (one response
  was two concatenated JSON objects), `gemini-3.5-flash` 9 scenes / 25s.
  Picked 3.6-flash: the scene-count edge for pro sits inside each model's own
  run-to-run variance, while flash is GA (not `-preview`), ~1.7x faster, and
  never failed to parse. Every run validated all `source_line`s (0 dropped)
  on both models. Caveat: the bundled samples are small (0.6-2.6k chars), so
  this is directional, not definitive — re-bench on real material before
  trusting it for a novel. `tests/test_frontier_profile.py` rewritten for the
  Gemini backend (32 tests); 439 total. Follow-ons unchanged: `spend.py`
  has no price entry for TEXT calls yet, `scenes` isn't opted into `frontier`
  by default, and no agent passes a `responseSchema` (the plumbing exists and
  eliminated the one parse failure in the bench).
- 2026-07-25 (later 3) — Self-critique refine is now bounded by a config
  knob, `critique.iterations`, **defaulting to 1**. Arrived there in two
  steps: first made the refine an unconditional 3-pass loop, then reverted it
  on instruction ("Remove change related to Multiple Critique iterations ..
  leave it as done only once") and reinstated it as configurable with
  ("set default ITERATIONS to 1"). At the default the behaviour is identical
  to the original single-pass code — the pre-existing `test_critique.py`
  tests, including `test_only_fires_once_not_recursively_on_the_refined_result`,
  are UNCHANGED and still pass, which is the real proof rather than a
  re-written assertion. Above 1, each refinement is re-critiqued and the loop
  stops early once the critic is satisfied. `enabled` remains the off switch;
  `iterations: 0` floors to 1 rather than silently disabling. 5 tests added
  for the knob without touching the originals. Then made both knobs
  resolvable PER STAGE (`critique.stages.<name>`, new
  `pipeline.critique_settings`, merged field-by-field so an entry setting only
  `enabled` keeps the global `iterations`), and shipped **`scenes` disabled**:
  it's already checked by story fidelity, genre alignment, and the
  deterministic scene-alignment self-heal, it's the most expensive stage to
  re-run, and a craft critique of a scene list pushes toward re-segmenting —
  churn against rule 9, the rule the stage is tuned for. `validate_config`
  checks `critique.stages` keys against the real stage registry (lazy import
  to dodge the stages -> agents -> llm cycle), since a typo'd stage name is
  otherwise completely silent — the override just never matches while the
  operator believes it did. 9 more tests.
- 2026-07-25 (later 5) — Opted `scenes` into the `frontier` profile
  (`agent_profiles.scenes: frontier`), making it the first and only stage on
  a hosted model; every other stage and every grader stays local. Replaced
  `test_no_agent_is_opted_in_by_default` — which asserted the opposite and
  was correct until now — with one pinning the intended state
  (`hosted == {"scenes"}`), plus a guard that no grader profile is ever
  hosted. Two known bottlenecks remain AHEAD of the model, so this doesn't
  yet cash in the 1M context: `llm.MAX_CHARS` still truncates the source at
  12,000 chars, and `duration_budget.suggest_scene_target` still injects a
  literal "roughly 2-4 scenes" from the 45s `--target-duration` default into
  the prompt's opening sentence.
- 2026-07-25 (later 6) — Made the source-text budget per-PROFILE
  (`llm.max_chars`, replacing direct `MAX_CHARS` reads in `structure`,
  `characters`, and `scenes`), which is what actually cashes in the frontier
  tier's context: the truncation happens in the AGENT before the prompt is
  built, so opting `scenes` onto a 1M-context model had still been cutting
  the story at ~2,000 words. Local profiles keep 12,000 exactly — deliberately
  not scaled from `num_ctx`, since that would change `synthesis`-backed stages
  as a side effect of a hosted-tier change. The load-bearing case is a hosted
  profile with no key: it takes its `fallback_profile`'s budget, so the
  truncation decision agrees with where `llm.generate` actually routes rather
  than sending 600K chars at an 8K-context model. `options.max_source_chars`
  overrides per profile. Then, per follow-up instruction, made the LOCAL side
  hardware-derived per model too (`llm.local_max_chars`) instead of one flat
  12,000 for every tier: `num_ctx` is the hardware knob (chosen per profile
  against 8 GB VRAM + CPU-RAM KV cache), so the budget derives from it minus
  a 3,000-token prompt reserve (sized on `scenes`, the largest whole-source
  prompt at ~2,900) and a 2,000-token output reserve. Yields 12,768 at
  num_ctx 8192 — within 10% of the old hand-tuned value, so the tiers this
  host runs on barely move — and 45,536 at 16,384 for `synthesis`, which had
  been capped as if it were an 8K profile. Two of my own earlier tests
  asserted `== MAX_CHARS` where they meant "the local budget"; the derivation
  surfaced that imprecision and both were corrected to compare against the
  resolved value. 14 tests including an end-to-end one through the real
  scenes agent; 462 total. Still open: `suggest_scene_target` injects a
  literal "roughly 2-4 scenes" from the 45s `--target-duration` default, which
  now caps scene count more than any model or context limit does.
- 2026-07-25 (later 7) — Made "the tests never call a paid API" a MECHANICAL
  guarantee rather than a convention (`tests/__init__.py`): importing the
  test package blocks outbound sockets, so a forgotten `mock.patch` fails
  with `NetworkBlockedInTests` naming the address instead of silently
  billing. Blocked at the socket layer deliberately — patching
  `urlopen`/`gemini._post` is per-module, and the `google-genai` SDK does its
  own HTTP, so it would slip past a per-module guard (verified: the guard
  catches the SDK-backed `generate_image` path too). Loopback is blocked as
  well, so no test can depend on whether an Ollama daemon happens to be
  running. All 462 existing tests passed unchanged under the guard, which
  confirms the suite was already correctly mocked; a deliberate
  forgot-to-mock probe was then confirmed to fail. Also verified the suite is
  credential-independent by re-running it with the Gemini key file moved
  aside — identical results. `tests/test_no_api_cost.py` (9 tests) meta-tests
  the guard, since a deleted guard would otherwise be invisible: every other
  test still passes and the protection is just gone. 471 total.
- 2026-07-25 (later 2) — Diagnosed why scene count was capped, which drove
  everything above. It was never the prompt wording but three things in
  order: `duration_budget.suggest_scene_target` injecting a literal "roughly
  2-4 scenes" (from the 45s `--target-duration` default) into the scenes
  prompt's highest-salience opening sentence; `quality_high`'s `num_ctx:
  8192` leaving only ~2,300 output tokens ≈ ~14 scenes; and `MAX_CHARS =
  12_000` truncating the story itself. Measured `qwen3:30b`'s real limits
  along the way — native context is 262,144, not 8,192, and at 96 KiB/token
  KV a bump to 16,384 costs ~750 MB and fits; the 30b MoE is also *more*
  context-efficient per token than the 14b dense (4 vs 8 KV heads),
  correcting an earlier suggestion in this session to move `scenes` to
  `synthesis` for context room. First built the frontier tier on Anthropic
  (`claude-sonnet-5`); reworked to Gemini in the entry above after the
  server-side-setup concern, but the profile-level `provider` plumbing,
  `models.local_profile` grader guard, and `llm.generate` dispatch all
  originated here and survived the pivot unchanged.
- 2026-07-25 (later) — Added a DIRECTOR'S INTERPRETIVE EXPANSION layer to the
  scenes stage (`scenes.py` rule 11 + new `emotional_beat`/`expression`
  fields), per direct instruction to let an excellent director's viewpoint
  expand each line of the story into the elaborate emotional and expressional
  moments it carries, including ones the source never spells out. This is a
  deliberate, narrowly-scoped loosening of rules 1-2 ("SOURCE TEXT IS THE
  ONLY AUTHORITY"), drawn on exactly the same line
  `veo_prompt.panel_action`'s render-time "director's freedom" layer already
  used: an inferred EMOTION is grounded in the source and may even earn its
  own scene, an inferred PLOT POINT is not (no invented events, characters,
  locations, props, spoken lines, or outcomes). Two rules were tightened to
  hold that line rather than let it erode — rule 4 keeps `summary`
  unembellished so the interpretive layer stays quarantined in its own named
  fields (fidelity grading keeps a clean record to judge against), and rule 7
  now blocks "two beats from one passage" from degrading into two scenes
  restating one moment. Deliberately propagated all the way to the render
  rather than stopping in scenes.json — the dead-field failure mode
  `visuals.key_props` (2026-07-10) and panel `emotional_note` (2026-07-23)
  both previously had: added to soundscape/visuals/cinematography's scene
  projections *and* grounded to each one's `emotional_function` output field
  via a shared HONOR THE SCENE'S EMOTIONAL DIRECTION rule, plus
  `screenplay._director_block` and a `storyboard` `bundle.director` block
  feeding the panel `emotional_note` floor. Also fixed a real latent bug the
  change makes likely for the first time: `_attach_source_excerpts` handed
  the earlier of two scenes sharing a `source_line` offset an empty excerpt
  and `word_count` 0 (now they share the passage's span), since rule 11 makes
  two scenes drawn from one passage legitimate.
  `tests/test_director_emotional_layer.py` (24 tests, the excerpt-collision
  ones confirmed to fail without the fix); 403 tests total.
- 2026-07-25 — Closed the "Known deferred issue" left open on 2026-07-24:
  `rerender_panels`'s `_resolve_start_frame` had the same non-boundary-aware
  seed gap `_render_scene_frames` had before that fix — it always reused the
  previous panel's recorded `end_frame`/tail PNG even at a character "shot
  boundary" (someone entering OR leaving), unlike its own `prev_clip_path`
  computation right next to it, which was already boundary-aware. Now takes
  an `is_boundary` param (computed once in `_render`, shared with
  `prev_clip_path` so the two can't disagree, same pattern
  `_boundary_aware_seed` established) and short-circuits to
  `_frame_char_anchor` at a boundary panel. Matters most for a
  single-character boundary panel, where `_resolve_panel_references` never
  kicks in (needs 2+ resolvable portraits), so the seed path was the ONLY
  identity anchor available. `tests/test_rerender_panels_boundary_seed.py`
  (3 tests, confirmed to fail without the fix); 379 tests total.
- 2026-07-24 — Fixed the seed-selection half of the issue above: seed
  choice in `_render_scene_frames` (both render paths) wasn't
  boundary-aware, so a character "shot boundary" panel could keep reusing
  the previous panel's tail frame (still showing the outgoing cast)
  instead of a fresh anchor. New `pipeline._boundary_aware_seed`, sharing
  the same `is_boundary` value `effective_prev_clip` already computed so
  the two can't disagree. `tests/test_boundary_aware_seed.py` (6 tests);
  376 tests total.
- 2026-07-23 (later 2) — Added an empty-result safety net: any creative
  stage coming back with nothing meaningful (parse failure, or its
  defining content key missing/empty, per new `stages._STAGE_CONTENT_KEYS`/
  `is_stage_result_empty`) is rerun ONCE via the existing `rerun_fn`
  mechanism; still empty fails the run via new `StageEmptyResultError`
  instead of letting empty data reach downstream stages. Checked at every
  point `_gated` produces a fresh result (initial compute, critique
  refine, gate-loop rerun) per direct instruction that this must fire even
  mid-iteration, not just on the first attempt — new
  `pipeline._ensure_nonempty_result`. `stages.run_stage` (which doesn't go
  through `_gated`) carries its own copy; `cli.py` catches the new
  exception for a clean failure message. Two pre-existing test fixtures
  (`test_critique.py`, `test_revise_gating.py`) used a real stage name with
  placeholder/empty payloads for testing an unrelated mechanism and needed
  updating now that the check is stage-schema-aware.
  `tests/test_empty_result_retry.py` (21 tests); 370 tests total.
- 2026-07-23 (later) — Wired storyboard panels' `emotional_note` ("the
  emotion this panel must evoke," authored by cinematography, falling back
  to visuals'/soundscape's scene-wide read) into the actual rendered Veo
  prompt for the first time, via new `veo_prompt.panel_action` folding it
  into [Action] as a performance-direction clause ("conveying quiet
  dread"). Previously authored by three creative agents and shown at the
  gate, but silently dropped before reaching the render. Deliberately the
  pipeline's one "director's freedom" injection point — interpretive
  performance only, never new source-bound facts.
  `tests/test_panel_action_emotional_note.py` (13 tests); 349 tests total.
- 2026-07-23 — Added multi-segment timestamped Veo prompts
  (`reel/panel_grouping.py`, new; `veo_prompt.multi_panel_video_prompt`;
  `pipeline._render_panel_group`; config `video.multi_segment_prompting`,
  default on): consecutive panels WITHIN one scene sharing the same
  in-frame cast, fitting Veo's 8s max, render as ONE call via Google's
  timestamp-segment technique instead of one call per panel — Subject/
  Context/Style stated once, Cinematography/Action vary per segment. Only
  the plain image-seed Veo call actually honors a caller-chosen duration
  (reference-images hardcodes 8s, extend has none at all), so a merged
  group's own call is always forced onto that path, and disabled entirely
  under `video.overlays.enabled` or a non-Gemini backend. Deliberately
  scoped within one scene only (never crosses a `scene_number` boundary)
  after direct pushback that an early draft's "location is constant"
  reasoning was an unstated assumption, not a scoping decision — cross-
  scene merging deferred as TBD. `_char_set_changed` relocated verbatim to
  `panel_grouping.char_set_changed` as the single source of truth.
  `rerender_panels` now refuses (rather than silently corrupting a merged
  clip) if a targeted panel or its cascade target belongs to a merged
  group — recursive group-aware cascading scoped out of v1, whole-scene
  re-render as the escape hatch. `tests/test_panel_grouping.py` (18 tests);
  336 tests total.
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
  round regardless of which stage was edited) — per direct pushback that a
  narrower first-pass fix (only backfilling a stage in the edited stage's
  own downstream closure) needed to not depend on that closure at all.
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
