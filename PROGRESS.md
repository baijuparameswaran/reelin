# Progress

> What's true right now, and the append-only history of how it got there.
> Keep **Current state** current every session; append new entries to the
> TOP of **Session log** (newest first) — never rewrite older entries.
> Architecture reference lives in [ARCHITECTURE.md](ARCHITECTURE.md); the
> per-agent prompt reference is [AGENTS.md](AGENTS.md).

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
- **Increment 5 (multi-character Veo `reference_images`) is now implemented**
  — see the 2026-07-10 (later 11) session log entry for the full mechanism
  (`pipeline._resolve_panel_references`, `gemini.generate_video_with_references`,
  config `video.multi_character_references`). Scoped to CHARACTERS only
  (per explicit request), at a "shot boundary" (scene start, or the
  in-frame cast changing panel to panel), only when 2+ characters have a
  resolvable casting portrait — a single-character boundary still uses the
  proven `image=` seed path, which grounds a lone subject more literally
  than a loose identity reference does. **Still open, not implemented:**
  folding the scene's LOCATION portrait into the same `reference_images`
  call alongside characters (memory `veo_character_consistency` shows this
  works together in a live ad-hoc test, but wasn't asked for here and would
  eat into the 3-reference budget); multi-reference for Gemini STILL-image
  generation (`imagegen.py`'s `_gen_gemini`/`generate_image` still don't
  accept `refs`, even though `gemini.generate_image()` itself already
  does — this specific gap is unrelated to and unaffected by the video-side
  work just landed); Kling 3.0 as a possible alternate provider for
  cross-scene subject-locked generation, not yet vetted against its real
  API. Full notes in memory (`veo_character_consistency` — see auto-memory
  for this project, updated alongside this session's implementation).
- **Recommended next action:** Enable GPU — replace the snap Ollama:
  `! curl -fsSL https://ollama.com/install.sh | sh`
  Then re-pull: `ollama pull qwen3:4b && ollama pull qwen3:8b`.
  Verify: `ollama ps` → "PROCESSOR" should show GPU or GPU+CPU.
- **`tests/test_prompt_rules.py` is now a real, committed suite** (stdlib
  `unittest`, `make test` — see the 2026-07-10 session log entry for what it
  covers). Narrower in scope than the full end-to-end suite shelved below —
  pure prompt-text and deterministic-enforcement checks, no LLM/API calls at
  all — but genuinely committed and run, not a throwaway per-session script.
- **Still shelved (2026-07-09): a formal `tests/` end-to-end pipeline suite**
  (stdlib `unittest`, stubbing every paid API — Gemini image/video — plus
  Ollama for hermeticity) was drafted (full `pipeline.run()` drive-through +
  a scene-alignment self-heal scenario) but explicitly shelved before being
  verified or committed; the two draft files were removed per instruction, no
  trace left. Distinct from `test_prompt_rules.py` above — this would be a
  much larger, slower suite exercising the actual orchestration/gate/resume
  machinery, not just prompt text. Revisit if/when that becomes a priority —
  this project's established practice for THAT scope remains throwaway
  stubbed scripts per session (see the session log entries above for worked
  examples), not a committed suite.
- **Next up:** confirm the still-running `storyboard` live test (bundle
  location/cast data, panel consistency with the rendered location image);
  live smoke-test the multi-character-reference feature (2026-07-10 later 11)
  and the SDK video-call fixes on a real run (both currently only verified
  offline against stubs/the real SDK's Pydantic validation, per this
  project's established practice of not spending live API quota mid-
  investigation); moodboard tile auto-render (opt-in); richer ingest
  (PDF/EPUB/.fdx); draft all scenes (not just first N); edit / sound mix /
  final cut phase.

## Session log
- 2026-07-11 — **`revise` now always operates as if `--max-scenes all` had
  been used, and prints its scoping evaluation up front plus a per-
  downstream-stage indication of what's changing.** Three related requests
  in one message: (1) "revise options should always run with
  max-scenes=all", (2) "only modify relevant scenes after comparing the
  diff of what has changed" (already the core design — confirmed still
  intact, not a regression to guard against), (3) "print the evaluation of
  changed scenes as soon as the scenes are identified during revise and as
  well indicate what is changing in each downstream stage."

  (1): `_revise_loop` previously inherited `max_scenes` from
  `run_params.json` (the 2026-07-10 (later 6) fix — see that entry) —
  correct for `--resume`, but wrong for `revise`: an original run's
  `--max-scenes` reflects a PROTOTYPE-SCALE choice made at the time (often
  the default of 1), and re-capping every subsequent revision to that same
  small number means a scene the diff correctly identifies as affected can
  still be invisible to casting-image/video rendering for no reason a
  revision should ever need. Changed `_revise_loop` to hardcode
  `max_scenes = None` unconditionally, no longer reading it from
  `run_params.json` at all — `profile`/`target_duration` are still
  inherited exactly as before, only `max_scenes` changed. Scoping itself
  is untouched: `revise_keys` (not `max_scenes`) is what actually
  determines which scenes get regenerated, so this only removes a second,
  redundant, accidental cap sitting on top of that — the LLM/image/video
  calls a revision makes are exactly the same set either way, just no
  longer silently excluded from a scene the diff already said needed
  regenerating.

  (3): Previously the only "what's happening" signal was the final
  `[reel] plan: save X.json (revising: [...]), then re-run: ...` line,
  printed just before the confirm prompt — after any ripple-suggestion
  LLM call and casting identity-drift check had already run silently.
  Added a `[reel] evaluation — X.json: changed=[...] added=[...]
  removed=[...]` line printed IMMEDIATELY after the diff is computed in
  `_revise_one` (or, for `_revise_source`'s scoped path, relabeled the
  existing early "scoped: scene(s) [...] affected" print to the same
  "evaluation —" wording for consistency) — before any of those later
  side effects, so the operator sees what was identified as soon as it's
  known. Then `_run_downstream_revision` gained a one-line indication
  printed right before EACH downstream stage actually runs:
  `regenerating [...]` for a scoped subset, `full regen (no scoped subset
  applies)` when `_translate_revise_keys` returned `None` (covers both an
  upstream drastic edit and an untranslatable cross-type scope — same
  message either way, since both mean "full regen" for that one stage
  regardless of cause), and `re-rendering (image provider)` for
  `casting_images`/`moodboard_tiles` (previously silent). The existing
  `nothing to regenerate (already up to date)` (empty-scope skip) and the
  `RENDER_SKIP_STAGES` skip notice were already present and needed no
  change — this just fills in the two cases that were previously silent.

  Added `tests/test_revise_inheritance.py::TestReviseLoopAlwaysUsesMaxScenesAll`
  (3 tests, driving `_revise_loop` via mocked `input()` — pick one stage,
  then `quit` — and a mocked `cli._revise_one` call-recorder): an
  original run that used `--max-scenes 1` does NOT reach `_revise_one` as
  `1`; an original run that already used `all` stays `all`; and a
  completely missing `run_params.json` (pre-existing `--out`, or one only
  ever touched by standalone `stage`/`revise` commands) still resolves to
  `all` rather than crashing or defaulting to `1`. Added
  `tests/test_revise_scene_delete_add.py::test_evaluation_and_per_stage_printouts`
  (captures stdout via `contextlib.redirect_stdout`): confirms the
  evaluation line appears with the correct added/changed/removed content
  AND appears textually before the final `plan:` line, and that both new
  per-stage messages (`soundscape: regenerating [4]`,
  `casting_images: re-rendering (image provider)`) are present. Full suite
  now 132 tests (was 128), still fully offline, `py_compile` clean.
- 2026-07-10 (later 12) — **`continuity_mode: extend` now only fires when a
  panel's in-frame characters match the immediately preceding panel's** —
  direct follow-up request ("use extend video as long as the
  characters/properties in the subsequent panels are same") to the
  multi-character-reference work in the entry just below. Previously
  `i2v._gen_gemini`'s extend-mode eligibility check was purely mechanical —
  config `continuity_mode: extend` set, a `prev_clip` path given, and
  resolution 720p — with no awareness of WHAT was actually in the previous
  clip; a panel whose cast changed (via the multi-character-reference
  boundary logic from the entry below) would still have `prev_clip_path`
  reach `_gen_gemini`, so if reference-images ended up NOT used for that
  boundary (e.g. only one new resolvable portrait, under the 2-reference
  threshold), extend-mode could still fire and wrongly extend the PREVIOUS
  clip's now-wrong-cast continuity/audio into a shot that doesn't feature
  those characters. Fixed by extracting the existing boundary check out of
  `_resolve_panel_references` into a standalone `pipeline._char_set_changed
  (fr, prev_char_key)` (same logic, now independently callable) and having
  every caller that threads `prev_clip_path` into `_render_one_panel` null
  it out — for that one call only, not the loop's running `prev_clip_path`
  variable itself — whenever the current panel is a character boundary
  relative to the one before it: `_render_scene_frames`' normal per-panel
  walk, `rerender_panels`' `_render(pnum)` (using its existing
  `_prev_char_key(pnum)` helper), and — the part easy to miss —
  `rerender_panels`' cascade-stop bookkeeping (the panel two hops past a
  targeted re-render, whose `.hash` sidecar gets rewritten to the value it
  WOULD have if fully re-chained, without touching its actual bytes): that
  recompute's `new_clip_path` stand-in needed the identical nulling when
  the bookkeeping panel is itself a boundary relative to the freshly-
  rendered panel's new cast, or the bookkeeping hash would disagree with
  what `_render_one_panel`'s real formula produces for it.

  **PROPS deliberately got no equivalent check** — traced whether
  `visual_overview.key_props` (the storyboard's per-SCENE prop list, see
  `pipeline._panel_context`'s own docstring: "no artifact currently says
  which panel a given prop appears in") has any panel-level attribution in
  the current schema, and it doesn't: `key_props` is constant across every
  panel in a scene, so a "did props change between these two panels" check
  would always be trivially true within a scene and could only ever differ
  at a SCENE boundary — which already unconditionally resets
  `prev_clip_path`/`prev_tail` to `None` between scenes, before this
  change existed. Documented this reasoning directly in
  `_char_set_changed`'s docstring rather than adding dead code that could
  never actually fire, so a future session doesn't waste time re-deriving
  the same conclusion or "fixing" what's actually already covered.

  Verified via `tests/test_multi_character_references.py`: 4 new
  `_char_set_changed` unit cases (scene start, same cast, changed cast,
  no-characters-listed) plus a new end-to-end `_render_scene_frames` test
  (3-panel scene: panel 2 has the same cast as panel 1 and correctly
  reaches `i2v.generate_clip` with a non-`None` `prev_clip`; panel 3, where
  a second character joins, correctly reaches it with `prev_clip=None`
  even though panel 2's clip genuinely exists on disk by then). Full suite
  now 128 tests (was 123), still fully offline, `py_compile` clean.
- 2026-07-10 (later 11) — **Every character in frame at a scene/shot
  boundary now gets its own Veo identity reference, not just one** —
  Increment 5 from the 2026-07-04 "locations cast alongside characters"
  session, explicitly paused back then and picked up now on direct
  request ("while determining the scene/shot boundaries, make sure to pass
  in the reference of all the characters present in the frame as a
  reference image while rendering"). The real gap this closes:
  `pipeline._frame_char_anchor` has always only anchored the FIRST name in
  a panel's `characters_in_frame` list — a panel with two or more
  characters gave every character after the first ZERO identity grounding,
  since Veo's `image=` seed can only carry one still image. Design taken
  directly from the prior investigation already saved in project memory
  (`veo_character_consistency` — includes a live ad-hoc SDK smoke test from
  2026-07-06 confirming the request shape works): Veo 3.1's "Ingredients to
  Video" `reference_images` config field accepts up to 3 ASSET-typed
  identity images per call, but is HARD mutually exclusive with
  `image=`/`last_frame` continuity in the same call — a genuine Veo API
  constraint confirmed via live SDK introspection, not something this
  wrapper can work around. So it's used selectively, not on every panel:
  new `pipeline._resolve_panel_references(fr, prev_char_key, cast_index,
  out)` treats a panel as a "shot boundary" when it's a scene's first panel
  or its in-frame character SET differs from the immediately preceding
  panel's, and only actually returns references when 2+ of those
  characters have a resolvable casting portrait — a boundary panel with
  just one character stays on the existing single-`seed` path, since a
  literal first-frame seed is a stronger grounding for a lone subject than
  a loose "ingredient" reference is, and there's nothing "multi" about one
  character anyway. Every other (non-boundary) panel is completely
  unaffected — still chains from `prev_tail` exactly as before.

  New `gemini.generate_video_with_references` (+ `_generate_video_with_
  references_once`) mirrors `extend_video`'s shape (SDK-only — the raw
  REST `predictLongRunning` surface has never been verified to accept this
  field, unlike image-seeding/extend; same transient-error retry/backoff,
  same `_log_call` tagging). Always requests Veo's only valid duration for
  this mode (8s, per `i2v._veo_nearest_valid_duration`'s docstring, which
  had already documented this force_max case in an earlier session before
  the feature existed to use it) and `person_generation="allow_adult"`
  (the official parameter table groups reference-images with image-to-
  video/interpolation for this field, per the comment already sitting next
  to `_extend_video_once`'s own `"allow_all"` choice). `i2v._gen_gemini`
  gained a `reference_images` param, tried FIRST — ahead of extend-mode —
  since a character-boundary is exactly the moment continuity from the
  previous clip should yield to fresh identity-lock for the new cast; any
  failure (config `video.multi_character_references: false`, SDK
  unavailable, API error) falls straight through to the existing
  seed/extend logic using the SAME `images`/`prev_clip` the caller always
  passes regardless — `pipeline.py` computes the normal single-image
  `seed` unconditionally, same as before this feature existed, and passes
  `reference_images` ALONGSIDE it rather than instead of it, so a disabled/
  failed reference call never loses grounding entirely, it just falls back
  to exactly today's behavior. `generate_clip`/`_content_hash` both thread
  `reference_images` through (the hash now includes every reference
  image's bytes, so a casting-portrait revision correctly invalidates a
  boundary panel that references it, same as it already invalidates a
  panel that SEEDS from it); the manifest's per-panel record gained a
  `reference_images` field for operator visibility into which characters a
  given boundary panel was asked to lock.

  `rerender_panels` (the one-hop-cascade targeted re-render path) needed
  the identical boundary/reference computation wired in too, or its
  `_content_hash` would disagree with what a full `_render_scene_frames`
  pass computes for the same panel — the exact class of bug
  `_resolve_prev_clip_path` was already written to prevent for
  `prev_clip_path`, in the session that built the one-hop cascade
  mechanism. New `_prev_char_key(pnum)` (reads the PRECEDING panel's
  `characters_in_frame` from the storyboard, since a targeted re-render
  isn't walking sequentially) feeds the same shared
  `_resolve_panel_references` both call sites now use. The cascade-stop
  bookkeeping (the panel two hops past the last target, whose `.hash`
  sidecar gets rewritten to the value it WOULD have if fully re-chained,
  without touching its actual clip bytes) needed the same treatment — it
  was already missing `requested_seconds` from its hash recompute (a
  pre-existing gap, left alone, out of scope here) but adding
  `reference_images` there was NOT optional: without it, a boundary panel
  two hops past a cascade would get a bookkeeping hash that disagreed with
  `_render_one_panel`'s real formula, spuriously flagging it stale (or not)
  on a later pass — a real latent bug this change would otherwise have
  introduced, caught by tracing the formula through rather than by a test
  failure.

  New config `video.multi_character_references` (default `true`).

  Added `tests/test_multi_character_references.py` (16 tests, fully
  offline — no live Ollama/Gemini/Veo calls): `_resolve_panel_references`'
  boundary/cast-change/portrait-availability/3-image-cap logic directly (7
  cases); `_render_one_panel` threading `reference_images` through to
  `i2v.generate_clip` and into the hash (confirmed a revised casting
  portrait's bytes changing re-triggers a render); `i2v._gen_gemini`'s
  branch selection (reference call used when 2+ refs + config enabled;
  falls back to seed on a single ref, config-disabled, or a raised
  exception); and — notably — a test against the REAL installed
  `google-genai` SDK types (only `Client` mocked, so real Pydantic
  validation of the request shape still runs, matching how earlier
  sessions verified other SDK-shape fixes) confirming
  `GenerateVideosConfig.reference_images` actually accepts the built
  `VideoGenerationReferenceImage(image=..., reference_type=ASSET)` objects,
  `duration_seconds=8`/`person_generation="allow_adult"` land correctly,
  and more than 3 supplied images get capped; plus one end-to-end
  `rerender_panels` test (3-panel scene, Bob joins at panel 2) confirming
  the targeted boundary panel gets both characters' references while the
  one-hop cascade panel (same cast, not a boundary) correctly gets none.
  Full suite now 123 tests (was 107), all still offline, `py_compile`
  clean. Not live-tested against the real Veo API in this session
  (deliberately, per this project's established practice of stubbing first
  and spending real API quota only once, on purpose, later) — worth a
  live smoke test on the next real run, per the updated "Next up" note.

  Updated project memory `veo_character_consistency` to reflect this is
  now implemented (was "Open investigation... nothing implemented yet"),
  keeping the still-genuinely-open items (location-in-the-same-call,
  Gemini still-image multi-ref, Kling 3.0) so a future session doesn't
  have to re-derive what's actually left.
- 2026-07-10 (later 10) — **`revise` can now DELETE and ADD scenes directly
  in `scenes.json`, not just modify one — lifting the "v1 scope assumption:
  scene count/order is stable" limit documented (and deliberately left
  alone) in several earlier session-log entries.** User asked directly for
  the revise agent to be able to "delete/add/modify from the existing
  jsons/artifacts and create the revised ones." Scoped this to the one
  artifact where it's well-defined and highest-value: `scenes.json`, the
  single source of truth every other scene-keyed artifact (soundscape/
  visuals/cinematography/screenplay/storyboard) is declared to depend on —
  those artifacts still can't be independently edited to add/remove a
  scene (a mismatch there remains `drastic`, unrelated to this feature);
  they get kept in sync with scenes.json automatically.

  `artifact_diff.ARTIFACT_SHAPES["scenes"]` changed from
  `allow_add=False, allow_remove=False` to `True, True` — a scenes.json
  edit that adds or removes a scene number is no longer automatically
  `drastic`. ADD already mostly worked once this flipped:
  `revision_merge.merge_by_key` already appends a genuinely new key: the
  one gap was ordering — a new scene got appended to the END of the array
  regardless of its number, but several downstream consumers (e.g.
  `storyboard._scene_bundles`) iterate `scenes["scenes"]` in LIST order,
  not re-sorted by number — fixed by having `scenes.segment_scenes` re-sort
  the merged list by `number` after every scoped merge (a single numeric
  sort key, not a tuple, to avoid a `None < None` comparison error if more
  than one entry somehow lacks a number).

  DELETE needed real new machinery, since `merge_by_key` can only add or
  replace a key, never remove one — deliberately reusing what already
  exists rather than inventing a second mechanism: `cli._revise_one`'s
  `"scenes"` branch now computes `removed_keys` from `diff.removed`
  (previously computed but discarded — `revise_keys` only ever included
  `changed | added`) and, right after saving the edited scenes.json but
  BEFORE the normal downstream cascade, calls two new functions: (1)
  `cli._strip_removed_scenes` — for every scene-keyed stage in
  `downstream_of("scenes")`, calls `fidelity.strip_orphan_scenes` (the
  EXACT SAME deterministic primitive `pipeline.run_group` already uses to
  self-heal scene-structure alignment during a fresh run — not a second
  implementation) against that stage's current on-disk artifact and saves
  the result back; `screenplay` additionally gets `screenplay.fountain`
  regenerated (deterministic — `to_fountain`, no LLM) plus its
  `drafted_count`/`total_scenes` bookkeeping fields corrected, since a
  pure-deletion edit leaves `screenplay`'s own `revise_keys` scope empty
  and nothing else in the round would otherwise touch that file. (2) `cli.
  _strip_removed_scenes_from_video_manifest` — `pipeline._assemble_movie`
  stitches every scene `video/manifest.json` lists, in order, so a stale
  entry for a deleted scene would otherwise still end up in `movie.mp4`;
  strips it and re-assembles (the underlying clip files are left on disk,
  not deleted — non-destructive, matching this codebase's established
  "never delete, only stop referencing" convention for
  `rerender_panels`'s one-hop cascade-stop bookkeeping from an earlier
  session). Both are pure local bookkeeping (no Gemini/Veo call), so they
  run unconditionally, independent of the `render` skip-by-default flag —
  only actual media generation is gated by that flag, not cleanup.

  **A real correctness bug was found and fixed while wiring this up,
  independent of deletion specifically**: `cli._translate_revise_keys`
  (the `scenes`→`casting` cross-type scope translation, via revised
  scenes' `location` names) returned `None` — meaning "no known
  translation, fall back to a full regen" — whenever it found zero
  locations among the targeted scenes, with no distinction between "the
  input `revise_keys` was itself empty" (nothing to translate, correctly
  an empty/no-op scope) and "the input was non-empty but genuinely
  ambiguous" (correctly falls back). This was dormant before this session
  because `_revise_one` always bailed out early ("no changes detected")
  before `revise_keys` could ever reach `_run_downstream_revision` empty —
  but this session's other fix (see next paragraph) makes an empty
  `revise_keys` a normal, reachable case, which would have made EVERY
  scenes-only-deletion revision wrongly force a full `casting` regen for
  no reason. Fixed: `_translate_revise_keys` now returns `set()`
  immediately for an empty input, before ever computing `locs`.
  `cli._run_downstream_revision` gained a matching optimization for the
  general (non-`scene_render`/`casting_images`) branch: when the
  translated scope is empty-but-not-`None`, `run_stage` is skipped
  entirely for that stage rather than called and its entire response
  discarded by `merge_by_key`'s `keys_to_replace=set()` — avoids burning
  local Ollama compute on output nothing will ever use.

  The OTHER half of the "no changes detected" bailout also needed fixing:
  it checked `elif not revise_keys:`, which would have incorrectly
  swallowed a pure-deletion edit (no add/change, `revise_keys` legitimately
  empty, but `removed_keys` non-empty) as "nothing to revise" and returned
  early without ever saving the edit — changed to `elif not revise_keys and
  not removed_keys:`. This same bailout also covers casting/characters
  (name-keyed, `allow_remove=True` already before this session), so a
  pure-deletion edit there is fixed too, though — per explicit scoping
  decision — those artifacts get no equivalent downstream-stripping step
  (no per-scene structure to reconcile against): the deletion is simply
  saved as-is, and downstream stages stop seeing that name in FUTURE
  regenerations only, not retroactively cleaned up. The revision-plan
  printout (`[reel] plan: save ...`) now separately lists `revising: [...]`
  and `removing: [...]` when both apply to the same edit (e.g. a round that
  both adds scene 6 and deletes scene 3 at once — fully supported, handled
  correctly by running the strip step first, then the normal scoped
  regen for whatever was also added/changed).

  Added `tests/test_revise_scene_delete_add.py` (10 tests, all offline/
  mocked — no LLM/API/ffmpeg calls): the shape change itself (add/remove no
  longer drastic for `scenes`, still drastic for every other scene-keyed
  artifact); `segment_scenes`' merge-then-sort placing a new scene number
  in correct narrative order while leaving untouched scenes byte-identical;
  `_translate_revise_keys`'s three cases (empty input → empty output,
  `None` stays `None`, non-empty-with-no-locations still falls back); and
  three end-to-end `_revise_one("scenes", ...)` scenarios against a
  realistic on-disk fixture (5 stages + a video manifest) — deleting scene
  2 correctly strips it from soundscape.json/screenplay.json, regenerates
  screenplay.fountain with scene 2's content gone, strips it from the video
  manifest, and — the regression this was built to catch — does NOT call
  `run_stage` for soundscape at all (nothing to regenerate); a
  deletion-only edit reaches the actual apply logic instead of hitting the
  old "no changes detected" bailout; and adding scene 4 correctly scopes
  `revise_keys={4}` through to every downstream stage. Full suite now 107
  tests (was 97), still zero LLM/API calls anywhere; `py_compile` clean
  across `reel/*.py reel/agents/*.py tests/*.py`.
- 2026-07-10 (later 9) — **A source-text edit in `revise` is now SCOPED to
  the scenes it actually affects, on the largest local model tier — closing
  a gap explicitly deferred as "genuinely hard, out of scope for v1" in an
  earlier session.** User asked directly: identify the actual change in the
  story, pass the current rendered artifacts at each stage, and change only
  the scenes/shots that actually require it — using the largest available
  model so the whole thing "can be processed seamlessly." Previously ANY
  edit to the raw story text (`cli._revise_source`, the special handler for
  `stage_name in ("source", "ingest")`) unconditionally fell back to a full
  regen of every single stage, since `artifact_diff.diff_source_text`'s own
  docstring called word-level prose diffing "a genuinely hard problem...
  out of scope for v1." Built the missing piece as two deterministic
  helpers plus one LLM confirmation call, deliberately NOT "send the whole
  story twice to a big model and hope" — that would be slow, wasteful of
  context, and give the model no structural hint about where to even look.
  New in `reel/artifact_diff.py` (pure, no LLM, no new dependency — matches
  this module's existing design contract): `candidate_changed_scenes(old,
  new, scenes)` flags any scene whose stored `source_excerpt` (or
  `source_line` for a checkpoint predating that field) is no longer found
  verbatim, whitespace-normalized, in the new text — a cheap, deterministic
  pre-filter; `unified_source_diff(old, new)` turns the edit into a
  COMPACT paragraph-level diff (`difflib.unified_diff` over a new
  `_split_paragraphs` helper — blank-line paragraphs, falling back to
  single lines, falling back to sentence-splitting for a single-block story
  with no line breaks at all, so there's always real diff granularity to
  work with) instead of the full story. New `reel/agents/revision.
  identify_source_text_changes(unified_diff, scenes, candidates, profile)`
  takes both, asks the model to confirm/refine the candidate set against
  the diff's actual content and decide `drastic` (the diff implies a scene
  should be ADDED or REMOVED — the same v1 "scene count/order stays
  stable" assumption every other scene-keyed artifact diff already makes,
  intentionally not solved here either) vs. a scoped `changed_scene_
  numbers` list. FAILS SAFE at every layer, verified via 5 direct test
  cases: a malformed/unparseable response, or a `drastic: false` response
  that's internally inconsistent (no scene numbers AND no reason given),
  both force `drastic=True` rather than silently under-scoping; every
  returned scene number is sanitized against the scene numbers that
  actually exist before the caller ever sees it, so a hallucinated number
  can never leak into `revise_keys`.

  For "largest available model": bumped `agent_profiles.revision` from
  `quality` to **`quality_high`** (qwen3:30b, the biggest tier this
  hardware fits — ~7-9 tok/s, slow, but this call is infrequent — once per
  source-text revision round, not once per scene — and correctness matters
  more than speed for correlating a diff against potentially many existing
  scenes). This also upgrades `suggest_ripple_scenes`'s tier as a side
  effect (same config knob), a reasonable free improvement for a similarly
  judgment-heavy advisory call.

  Wiring it all together needed one refactor to avoid a second, parallel
  scoped-revision implementation: extracted `cli._revise_one`'s inline
  downstream-cascade loop (the `for dname in downstream: ...` block —
  `RENDER_SKIP_STAGES` check, `scene_render` routing through
  `_apply_scene_render_revision` with name→scene-number translation,
  `_translate_revise_keys` per stage) into a shared `cli.
  _run_downstream_revision(out, stage_name, downstream, revise_keys,
  edited_artifact, panel_targets, ...)`, called by BOTH `_revise_one` (its
  original call site, behavior unchanged — verified by the full existing
  suite passing unmodified) and the new `_revise_source` scoped path: once
  `identify_source_text_changes` returns a non-drastic result,
  `stages.run_stage("scenes", existing=current_scenes, revise_keys=
  changed_scene_numbers, ...)` re-runs `scenes` itself scoped (the exact
  same `existing=`/`revise_keys=` mechanism a direct `scenes.json` hand-
  edit already uses), then `_run_downstream_revision` cascades from there
  exactly like `_revise_one("scenes", ...)` would — literally the same
  code path, not a lookalike. Falls back to the original full-regen-of-
  every-`STAGES`-entry behavior in exactly two cases: no `scenes.json`
  exists yet to correlate the diff against (verified the LLM call is
  skipped entirely here, not just ignored — nothing to attempt), or the
  analysis itself says `drastic`.

  **A real regression was caught and fixed while verifying this**: the
  scoped analysis call fires whenever `scenes.json` exists — which it does
  in `tests/test_revise_inheritance.py`'s existing `test_revise_source_
  passes_inherited_attributes_to_every_stage` (its `_setup_run` writes
  `scenes.json`) — and that test didn't mock `identify_source_text_changes`,
  so running the full suite made a REAL live Ollama call, ballooning the
  suite from ~0.03s to ~68s. Fixed by adding a mock there forcing
  `drastic=True`, since that test's actual purpose (profile/max_scenes
  threading through the full-regen fallback) doesn't need or want the new
  scoped path exercised — restored to fully offline, <1s. Added
  `tests/test_revise_scoped_source.py` (15 tests) covering the diff/
  candidate helpers directly, `identify_source_text_changes`'s fail-safe
  behavior (5 cases) and profile resolution (default `quality_high`, an
  explicit override still wins), and three end-to-end `_revise_source`
  scenarios (scoped path only touches `scenes`+its real downstream — NOT
  a full-`STAGES` replay; drastic path still does the full regen; no-
  scenes.json path never even attempts the LLM call). Full suite now 97
  tests (was 82), still comfortably under a second, still zero live LLM
  calls anywhere in the suite.
- 2026-07-10 (later 8) — **`revise` now skips casting regeneration and all
  image/video rendering by default, with `--render`/`render on`/`render
  off` to opt in.** User asked for this directly, framed as a cost-control
  measure: casting-image (Gemini) and scene-video (Veo) rendering are the
  ONLY stages a revision round can trigger that cost real API spend —
  every other stage (structure/characters/scenes/soundscape/visuals/
  cinematography/screenplay/storyboard/fidelity) runs on the local, free
  Ollama models. Iterating on text content shouldn't have to pay for a
  re-render every single round. New `cli.RENDER_SKIP_STAGES = {"casting",
  "casting_images", "moodboard_tiles", "scene_render"}`, checked at the top
  of both regeneration paths: `_revise_one`'s scoped downstream loop (over
  `downstream_of(stage_name)`) and `_revise_source`'s full `STAGES` replay
  loop for a drastic source-text edit — each now prints a one-line skip
  notice per skipped stage rather than silently doing nothing, so it's
  clear from the output what didn't run and why. A skipped stage's on-disk
  checkpoint is deliberately left untouched (not deleted) — anything
  downstream that REQUIRES it (`storyboard` requires `casting`, for
  instance) still resolves correctly against the last rendered/cast state;
  that staleness is the accepted cost of free iteration, caught up whenever
  rendering is turned back on. Three ways to opt in, all threading a single
  `render: bool` parameter through `_revise_loop`/`_revise_one`/
  `_revise_source`: the standalone command's new `--render` flag (sets the
  session default); typing `render on`/`render off` at the interactive
  stage-picker prompt at any point, no restart needed (the menu header now
  echoes the live setting, e.g. "[render: off — casting/rendering skipped
  (type 'render on' to include them)]"); or directly picking `casting`
  itself from the menu to hand-edit it — that edit always happens
  regardless of the flag (it's an explicit, deliberate choice, not an
  incidental downstream cascade), only ITS OWN downstream
  `casting_images`/`scene_render` still respect the setting. Added
  `tests/test_revise_render_skip.py` (5 tests): confirms the exact stage
  set, that `_revise_one` selectively skips only the rendering stages while
  still regenerating non-rendering downstream stages (screenplay/
  storyboard/fidelity) under the default, that `render=True` correctly
  includes `casting_images`/`_apply_scene_render_revision`, and the same
  pair of checks for `_revise_source`'s full-regen path. Full suite now 82
  tests (was 77), still comfortably under a second, still zero LLM/API
  calls; the two existing `test_revise_inheritance.py` end-to-end tests
  (which exercise `_revise_one`/`_revise_source` without passing `render=`,
  so now implicitly exercise the new default-skip behavior too) continued
  passing unchanged, confirming no regression to the attribute-inheritance
  work from the entry below.
- 2026-07-10 (later 7) — **`soundscape.json` gained a `score_direction`
  field, fixing a real pre-existing bug: `storyboard.py`'s `_build_scene_board`
  has always read `bundle.audio.score_direction` into `audio_overview.
  score_cue` (the field `pipeline._panel_video_prompt` asserts as Veo's
  music directive), but `soundscape.py`'s schema never asked the model to
  produce that field — every scene's score direction has always been
  silently empty.** Found while implementing a larger, user-requested
  architectural change this session (moving character extraction/casting to
  the very start of the pipeline before a single scene exists, with
  rendering happening immediately; a scenes.py rule forcing a new scene at
  each character's first story appearance; a "keep background score
  subtle" config toggle). That work was reviewed carefully first (see the
  review below) — one real bug was found and fixed during the review
  (`stages.py`'s registry list order, reordered to cosmetically match the
  new pipeline sequence, broke `cli._revise_source`'s full-regen loop,
  which treats that list as a topological execution order) — but the user
  then asked to discard all the new architectural changes entirely and
  keep only genuine bug fixes. Reverted `pipeline.py`, `stages.py`,
  `casting.py`, `scenes.py`, and `config/models.yaml` to their pre-session
  state via `git checkout`; kept only the `score_direction` schema field
  (a real, independently-verifiable bug — storyboard.py's own MERGE SOURCE
  block already documented reading this exact field), stripped of the
  "subtle score" config-flag wrapping it had been bundled with (that
  wrapping was the architectural feature being discarded, not the bug fix
  itself). Added one regression test (`score_direction` present in the
  rendered schema) to `tests/test_prompt_rules.py`; suite is 77 tests
  (was 76 before this session's work began), full package compiles clean.
  The character-first pipeline reordering, the character-introduction
  scene-boundary rule, and the subtle-score toggle remain unimplemented —
  revisit as a fresh, focused piece of work if still wanted, informed by
  the review findings above (the two-phase casting split needs the
  `_revise_source` ordering issue solved differently next time, e.g. by
  NOT reordering `stages.py`'s registry to match, exactly as this review
  concluded).
- 2026-07-10 (later 6) — **`revise` now inherits the original run's
  attributes (profile, max_scenes, target_duration, genre/moodboard
  steering) instead of silently reverting to bare defaults.** User asked
  directly to "make sure the revise inherit all attributes from original
  run." Investigated by reading `cli.py`'s `_revise_one`/`_revise_source`/
  `_revise_loop` and every `stages.run_stage(...)` call site within them —
  found FOUR distinct, real gaps, all stemming from the same root cause:
  `revise` is a separate process invocation from the `pipeline.run()` that
  produced the checkpoints it edits, so nothing about that original run's
  configuration carries over automatically unless explicitly reloaded and
  threaded through. (1) **Steering was completely absent**: `llm.
  set_direction`'s process-wide directive starts unset in a fresh process —
  `revise` never called it at all, so every regenerated stage during a
  revision session ran with NO genre/moodboard creative direction, even
  though genre.json/moodboard.json exist on disk from the original run.
  This was the most consequential gap, silently undermining the whole
  "genre and moodboard STEER every stage" architecture for any revised
  content. (2) **`profile` was never passed** to any `run_stage(...)` call
  in the revise flow, so a stage regenerated via `revise` always fell back
  to its own config default tier rather than honoring e.g. a `--profile
  fast` override the original run used. (3) **`max_scenes` wasn't threaded
  through at all** — `run_stage`'s own default is `1` (intentional for
  standalone single-stage testing, per an earlier session's design), so any
  revise-triggered downstream regeneration would have silently re-capped
  casting-image/video rendering back down to 1 scene, even after a
  completed `--max-scenes all` run. (4) **`--target-duration`'s scene/shot-
  count guidance had no pathway into `revise` at all** — `stages.py`'s
  `_scenes`/`_cinematography` wrappers didn't even expose params for it.
  Fixed all four: extracted `pipeline.compose_direction(genre_spec,
  moodboard_spec)` from `run()`'s previously-inline `apply_direction()`
  closure (a pure refactor, verified behavior-identical via a dedicated
  test) so `cli.py` can reuse the exact same composition; new
  `cli._restore_direction(out)`, called once at the top of `_revise_loop`
  (covering both the standalone `revise` command and the post-run "revise
  now?" offer, since both funnel through that one function), reloads
  genre.json/moodboard.json and re-applies the steering before any stage
  regenerates. Reused the EXISTING `cli._load_run_params(out)` (already
  built for `--resume` inheritance in an earlier session) to recover
  `profile`/`max_scenes`/`target_duration`, threaded through every
  `run_stage(...)` call in `_revise_one` and `_revise_source`. `max_scenes`
  needed one deliberate exception — new `cli._effective_max_scenes
  (stage_name, max_scenes)` forces `screenplay` to always get `None`
  (uncapped) regardless of the inherited value, since a uniform pass-through
  would have reintroduced the exact cap the "screenplay always drafts
  everything, only rendering stages are capped" policy (fixed in an earlier
  session) deliberately removed — caught by directly re-reading that
  policy's own PROGRESS.md entry before writing the fix, not from memory.
  `target_duration` needed new plumbing: `stages.py`'s `_scenes`/
  `_cinematography` wrappers gained `target`/`shots_guidance` params
  (falsy-default, so every pre-existing call site — including a fresh
  `pipeline.run()`, which computes and passes these itself — is
  unaffected), fed by new `cli._duration_kwargs(stage_name, out,
  target_duration)`, which reloads the CURRENT scene count from disk for
  cinematography's guidance (since `scenes` may have just been regenerated
  earlier in the same revision round, and the guidance needs the up-to-date
  count, not the original run's). Added a new, separate test file (distinct
  from `test_prompt_rules.py`, which is scoped to prompt text — this is
  about CLI/orchestration attribute threading): `tests/
  test_revise_inheritance.py`, 13 tests — `compose_direction`'s composition
  logic in isolation, `_restore_direction` actually changing `llm.
  direction()`'s process-wide state (and degrading gracefully when
  genre.json/moodboard.json are absent), `_effective_max_scenes`'s
  screenplay exception, `_duration_kwargs`'s per-stage guidance (including
  the current-scene-count reload), and two end-to-end tests
  (`_revise_one`/`_revise_source` driven via their existing `edited_override`/
  `auto_confirm` testing hooks, with `stages.run_stage` mocked to a call
  recorder) confirming the inherited profile/max_scenes genuinely reach
  every downstream stage — including the screenplay exception holding even
  when every OTHER stage correctly receives the real inherited value. Full
  suite now 76 tests (was 63), still comfortably under a second, still zero
  LLM/API calls. All passed on the first run after implementation — no bugs
  found by the new tests themselves this time, unlike the prompt-rules
  suite's own first draft earlier today.
- 2026-07-10 (later 5) — **Fixed a real, reported bug: a JSON-schema
  EXAMPLE in casting.py's prompt leaked into actual model output.** User
  reported "Marcel's pocket watch" — the illustrative example casting.py's
  PROMPT used to show what a `kind: "prop"` entry looks like — showing up as
  a genuinely derived prop even though no such object exists in the actual
  story being processed. Root cause, confirmed by checking whether "Marcel"
  is a real character: it IS, in `samples/sample_story.2.txt` (0 hits in
  `sample_story.txt`, 2 in `.2.txt`) — the example wasn't a random invented
  name, it happened to collide with a real character already present in the
  same prompt's context, so the model had no clean signal that "Marcel's
  pocket watch" was a hypothetical illustration rather than a genuine
  suggestion tied to a character it could already see. Audited every
  PROMPT-facing string (not docstrings — those never reach the model) across
  all 13 agent files for the same pattern (a concrete, narrative-sounding
  proper noun used as an "e.g." example) and found exactly two files with
  the risk, both introduced in the props/casting work from earlier today:
  `casting.py`'s LOCATION and PROP schema examples ("Rusty Anchor Bar",
  "Marcel's pocket watch") and `scenes.py`'s rule 8 (`location`) and rule 10
  (`props`) inline examples (same two terms, plus "Lumen Field"). Fixed by
  replacing the two JSON-schema "name" field examples in casting.py with
  clearly generic, bracket-style placeholders ("LOCATION NAME (must exactly
  match a name in LOCATIONS INPUT below — this is a placeholder illustrating
  the shape, not a real place to include)" / the equivalent for PROP NAME) —
  matching the same unambiguous-placeholder convention already used
  elsewhere in this codebase (`characters.py`'s "NAME", `storyboard.py`'s
  "CHARACTER_A") rather than inventing a new pattern; and by swapping the
  inline "e.g." examples in scenes.py / casting.py's docstring for generic,
  possessive-free, non-proper-noun alternatives ("the harbor tavern", "a
  corner café", "a tarnished pocket watch") that read unambiguously as
  format illustrations rather than real suggested content. Every other
  agent prompt (soundscape/visuals/cinematography/screenplay/storyboard/
  characters/structure/genre/moodboard/fidelity/revision) was checked and
  found already clean — either no invented-name examples at all, or already
  using the generic placeholder convention. Added a permanent regression
  test, `TestNoStoryLikeExamplesInPrompts` in `tests/test_prompt_rules.py`:
  renders every prompt in the codebase and asserts none contain the terms
  involved in this incident ("Marcel", "Rusty Anchor Bar", "Lumen Field"),
  plus a check that casting.py's two schema examples now use the generic
  placeholder form — a cheap, permanent tripwire against the same mistake
  recurring if an "e.g." list is ever expanded without thinking through
  this risk. Suite now 63 tests (was 61), still <1s, still zero LLM/API
  calls. One docstring-only mention of "Marcel" in `scenes.py`'s
  `_validate` (an unrelated whitespace-line-wrap-matching illustration,
  never sent to any model) was deliberately left alone — out of scope
  since it has zero LLM exposure.
- 2026-07-10 (later 4) — **`make demo`/`make run` now depend on `make test`.**
  User asked for the prompt-rule suite to run before the demo, to make sure
  the codebase is intact first. Added `test` as a Makefile prerequisite to
  both `demo` and `run` (not just `demo` — the same rationale applies
  identically to running the pipeline against a user's own story) — `make`
  stops on a non-zero exit by default, so a failing test now blocks the
  actual pipeline invocation before it spends any real time or API quota,
  rather than surfacing a prompt regression only after a run is already
  underway. Verified via `make -n demo`/`make -n run` (dry-run, confirms
  `test`'s command prints first without executing the pipeline) and a
  throwaway deliberately-failing test in a scratch directory (confirmed
  `python -m unittest discover` exits non-zero, which is what makes the
  dependency actually block `make`).
- 2026-07-10 (later 3) — **Added a real, committed unittest suite
  (`tests/test_prompt_rules.py`, `make test`) validating that agent
  prompts actually follow this project's established prompting rules, and
  that rules enforced deterministically (not just by prompt instruction)
  do what they claim.** User asked directly for this after several rounds
  of ad-hoc, throwaway smoke-testing (`.format()` + substring checks run
  by hand each time a prompt changed) — converted that pattern into a
  proper, permanent suite instead of re-deriving it every session. Scoped
  deliberately narrower than the `pipeline.run()` end-to-end suite shelved
  in an earlier session (see the "Still shelved" bullet above) — pure
  prompt-text and deterministic-logic checks, zero LLM/API calls, so it's
  fully offline and runs in well under a second (61 tests, ~0.01s).
  Two kinds of checks, mirroring how this project's prompting conventions
  actually work: (1) PROMPT TEXT — every one of the seven "sandwiched"
  prompts (scenes/casting/soundscape/visuals/cinematography/screenplay/
  storyboard, per the 2026-07-09 audit) is asserted to actually contain
  its "Before you respond" reminder block at the TRUE end of the rendered
  string (not just present somewhere) with zero leftover unresolved
  `{placeholder}`s, plus every high-stakes rule keyword this session
  added or audited (MINIMIZE SCENE COUNT, SOURCE OVER COVERAGE,
  TIE-BREAKER, ISOLATION covering props, PROPS, SELF-CONTAINED VS. BRIEF,
  the "at least 1 shot" floor) is confirmed present in the actual text
  sent to the model, not just described in a docstring; a companion
  regression guard confirms the four short prompts (characters/structure/
  genre/moodboard) and two grader prompts (fidelity/revision) remain
  deliberately un-sandwiched, so a future edit can't silently drift either
  way without a test noticing. (2) ENFORCEMENT — the deterministic
  functions that catch what prompting alone doesn't reliably guarantee are
  tested directly: `scenes._validate` (drops a hallucinated `source_line`,
  keeps a real quote spanning a hard line-wrap), `scenes.
  _attach_source_excerpts` (contiguous non-overlapping partition, ordered
  by actual source position not scene `number`, graceful degradation for
  an unfindable line), `scenes._reconcile_character_names` (fixes the
  exact "Woman"/"Young Woman" class of near-miss, leaves a genuinely
  ambiguous or unrelated name alone), `casting._location_entries`'s
  `recurring_props` heuristic and `casting._prop_entries`'s global
  (not location-scoped) recurrence requirement, `storyboard.
  _build_scene_board`'s deterministic `key_props` copy (including the
  empty-list-not-missing-key edge case), and `duration_budget.
  suggest_shots_per_scene`'s 1-shot floor. A separate
  `TestProviderPolicySteeringSplit` class statically inspects (via
  `inspect.getsource`, no live calls) every creative-agent module for an
  `llm.generate(` call (steered) and every grader/direction-setting module
  for `models.text(` while asserting `llm.generate(` is ABSENT — turning
  ARCHITECTURE.md's provider-policy prose ("Gemini only for image/video;
  graders judge neutrally") into an actual regression-tested invariant
  rather than a convention that could silently drift. Caught one real bug
  in the suite's own first draft before it ever ran clean: a stale expected
  reminder-block tail string in the scenes.py test (written before this
  session's earlier prop-rule addition extended that same reminder block
  with one more bullet) — fixed by re-deriving the exact current tail from
  the file rather than trusting memory, the same "verify against current
  state, not what you remember" discipline the sandwiched prompts
  themselves now enforce on the model. Wired into `Makefile` as `make test`.
- 2026-07-10 (later 2) — **Recurring props now get a locked, very
  descriptive `visual_prompt` — the same identity-consistency treatment
  characters and locations already have — so the same object renders
  consistently across every scene it appears in, in both the casting image
  render and the actual Veo video prompt.** Direct follow-up to the prop
  wiring in the entry just below: that fix got a prop's plain NAME into the
  Veo Context section, but a bare name repeated across separately-generated
  clips still isn't enough for visual consistency — Veo has no cross-
  generation memory, so "a brass mirror" mentioned in three different scene
  renders could plausibly come out as three different-looking mirrors,
  exactly the failure mode already solved for characters/locations via a
  locked casting `visual_prompt`. Extended `casting.py` with the same
  mechanism for props: new `_prop_entries(scenes)` finds every prop name
  (exact-match — same known limitation as `_map_chunks`'s source_line
  matching, not solved here) appearing in 2+ DISTINCT scenes ANYWHERE in
  the story (deliberately not scoped to one location, unlike
  `_location_entries`'s existing `recurring_props` — a portable prop like a
  character's watch travels across locations, so location-scoping would
  miss it entirely); a single-scene prop isn't cast, since there's no
  repetition to keep consistent and casting every one-off object would be
  excessive cost/clutter. New `kind: "prop"` casting entries (no `actor`
  layer, same as locations) with a `character.visual_prompt` PROMPT rule
  demanding real prop-master specificity — exact material, color/finish,
  size, condition, distinguishing marks/engravings — under the same strict
  ISOLATION discipline as a character portrait (no scene/hands/other-
  objects baked in), extended the ISOLATION rule's own stated scope to
  cover props explicitly (previously scoped to "person/animal/bird/
  creature/group entries only"). Because `pipeline._render_casting_images`
  was already fully kind-agnostic (confirmed by reading it directly before
  writing any code — it just reads `character.visual_prompt` off any
  entry), a cast prop gets an actual rendered reference PNG for free, with
  zero changes to that function; the pipeline's active-names computation
  (which caps portrait rendering to `--max-scenes`) gained prop names
  alongside characters/locations so capped runs don't skip them. The
  deliberate scope boundary: a prop's rendered PNG is NOT wired into Veo's
  `reference_images`/seed-image mechanism the way a character's portrait
  is — only the prop's TEXT description reaches the render prompt, per
  what was actually asked (a textual definition "embedded in video or
  image prompts"), not a new identity-seeding channel. On the render side:
  `pipeline._panel_context` (already carrying the prop NAME from the prior
  fix) now resolves each name against `casting_lookup` — the exact same
  dict `_panel_subject` already uses for characters — swapping in the
  prop's locked, richly descriptive `visual_prompt` when a cast entry
  exists (kind == "prop"), falling back to the bare name for an uncast
  single-scene prop. Verified via direct unit tests: `_prop_entries`
  correctly found only the one prop recurring across two DIFFERENT
  locations (proving the deliberately-global, not-location-scoped
  aggregation works) while ignoring a single-scene prop entirely; a full
  stubbed `cast_characters()` call confirmed the real prompt actually
  contains "PROPS INPUT" with only the qualifying prop, and the parsed
  result carries the prop's cast entry through; `_panel_context` correctly
  swaps in the rich description for a cast prop, falls back to the bare
  name for an uncast one, and degrades gracefully (bare name, no crash) for
  a cast-but-visual_prompt-missing edge case; a full `_five_part_veo_prompt`
  assembly confirmed the rich description — not just the name — lands in
  the final Veo prompt string.
- 2026-07-10 (later) — **Added the concept of a scene-level `prop`, and fixed
  a real, previously-invisible gap: `visuals.py`'s `key_props` never actually
  reached a rendered video frame.** User asked for props to be added at the
  scene level, with the prop threaded into the location "in some form" so it
  gets used during rendering. Traced the existing prop-adjacent machinery
  first: `visuals.py` already had a `key_props` field (prop + dramatic
  `function`), but following it all the way to the real Veo render path
  (`pipeline._five_part_veo_prompt` → `_panel_context`) showed it was NEVER
  read there — `_panel_context` only ever built Context from the location's
  cast `visual_prompt` plus the panel's `composition` note; `key_props` (and
  `visual_moments`) were only ever referenced by storyboard's rarely-used
  LLM-feedback path's free-text `image_prompt`, not the structured render
  path every real render actually takes. So a prop visuals.py flagged as
  dramatically significant had, in practice, zero effect on any rendered
  clip. Fixed with three coordinated changes, in pipeline order: (1)
  `scenes.py` gained a new source-grounded `props` field (STRICT RULE 10,
  parallel to `characters`/`location`) — the earliest, most-authoritative
  point a prop can be identified, grounded in the actual source text rather
  than invented later; (2) `casting.py`'s `_location_entries` aggregates
  every scene's `props` per `location` into `recurring_props` — a prop
  appearing in MORE THAN ONE scene sharing that location (or the location's
  only scene, if it never recurs) is a plausible fixed fixture, deliberately
  excluding a single-scene-only prop at a multi-scene location, since baking
  a transient/action prop into a location's reused reference image would
  wrongly force it into every OTHER scene there too — the PROMPT still tells
  the model to treat these as candidates, keeping only genuinely
  architectural ones; (3) `visuals.py`'s scene list now includes each
  scene's `props` as grounding context for its own `key_props` judgment
  (still a creative "which props carry real dramatic weight" call, just no
  longer invented from nothing). The actual render-path fix:
  `storyboard.py`'s `_build_scene_board` (deterministic path) now copies
  prop NAMES from `bundle.art.key_props` into a new
  `visual_overview.key_props` field (also added to the PROMPT schema +
  MERGE SOURCE block for the LLM-feedback path); `pipeline._panel_context`
  gained a `key_props` param and folds them into the Context section
  ("visible in the scene: ..."), applied scene-wide (every panel in the
  scene gets the same list) since no artifact attributes a prop to one
  specific panel. Because `visual_overview` already flowed unchanged from
  `_render_scene_frames` down to `_five_part_veo_prompt`, only
  `_panel_context` and its one caller needed the actual code change — no
  signature changes needed anywhere else in the render chain. Verified via
  direct unit tests at every layer: `_location_entries`'s recurring-vs-
  transient heuristic (a prop in both of a 2-scene location's scenes
  survives, a prop in only one of them doesn't; a 1-scene location's props
  all survive since there's nothing to compare against), `_panel_context`'s
  junk-filtering (empty/None entries dropped) and correct Context-string
  assembly, `_build_scene_board`'s key_props copy (drops the `function`
  commentary, filters empty prop names), a full `_five_part_veo_prompt`
  assembly confirming a prop genuinely appears in the final assembled Veo
  prompt string, and a stubbed `segment_scenes()` round-trip confirming
  `props` survives alongside the existing `source_excerpt` machinery.
- 2026-07-10 — **Scene segmentation now actively minimizes scene count (fewer
  scenes → fewer rendered video clips downstream), and every scene captures
  its own precise source-text excerpt + word-count metadata.** User framed
  scene generation as "the critical phase for all other downstream actions"
  and asked for two things: (1) keep the scene count to a minimum, since
  scene count drives how many videos get rendered; (2) have each scene
  capture the actual portion of the story that defined it, plus metadata.
  For (1): added STRICT RULE 9 (MINIMIZE SCENE COUNT) to `scenes.py`'s
  prompt — merge consecutive beats sharing a `location` and continuous time
  into ONE scene, splitting only on a real location/time/purpose change, not
  for every new beat or line — reinforced in the prompt's task-intro line and
  its "before you respond" sandwich block (a new merge-check bullet). This is
  explicitly a pacing/cost bias, not a fidelity relaxation — rules 1-2/7
  (no invented scenes, no unnecessary repeats) still stand, so the model
  can't drop real events to save a scene, only stop over-fragmenting a single
  continuous moment. `segment_scenes`'s standalone default `target` changed
  from the flat "8-14 scenes" to "as few scenes as the story can be told
  in — often 3-6 for a short story", so a bare `stage scenes` run (no
  duration-budget guidance) gets the same bias. For (2): new deterministic
  (no LLM, so no hallucination risk) `_attach_source_excerpts` partitions the
  source text into one contiguous, non-overlapping span per scene — from
  that scene's already-validated `source_line` position to the next scene's
  — and attaches it as `source_excerpt` plus a `word_count` metadatum, a
  strict superset of the short 5-15-word `source_line` anchor that only ever
  existed for hallucination-checking and chunk-matching. Wired through:
  `ingest.scene_source_context` gained a `source_excerpt` param it prefers
  over the older, coarser chunk-based join (which only approximates scene
  boundaries at ~3000-char chunk granularity and can pull in neighboring
  scenes' text); `screenplay.py`/`storyboard.py`'s per-scene context lookups
  now pass it through, falling back gracefully for checkpoints predating this
  field. Known, documented limitation shared with the pre-existing
  `_map_chunks`: if a `source_line` phrase genuinely repeats verbatim
  elsewhere in the story, `str.find` locates its first occurrence, which may
  not be the one a given scene actually covers — inherent to anchoring on
  short quoted text, not new to this change. Verified via a full stubbed
  `segment_scenes()` round-trip (contiguous partition boundaries correct,
  last scene runs to end of text, unfindable `source_line` degrades to an
  empty excerpt rather than raising) and the scoped-revision path
  (`revise_keys`-untouched scenes keep their OLD `source_excerpt` byte-
  identical via `merge_by_key`, the revised scene gets a freshly computed
  one) — both against real `.venv` execution, no live Ollama call needed
  since these are pure deterministic post-processing steps.
- 2026-07-09 (later 7) — **Removed the hardcoded "at least 2 shots per scene"
  floor — a scene now only needs a minimum of 1 shot/panel.** User pointed
  out a scene may only need one shot. Found the floor in two places:
  `cinematography.py`'s PROMPT rule ("Each scene should have at least 2
  shots") and, more consequentially, `duration_budget.suggest_shots_per_scene`
  — its `est_total_shots`/`per_scene` math used `max(scene_count * 2, ...)`/
  `max(2, ...)`, so even a very short `--target-duration` (which should
  imply ~1 shot/scene at the ~6s/shot planning assumption) was silently
  doubled in the guidance text fed to cinematography.py, working against the
  user's own target. Both floors dropped to 1. Verified `_build_scene_board`/
  `rerender_panels` already handle a single-panel scene correctly with no
  code change needed (`is_last` is true for a lone panel;
  `rerender_panels`'s one-hop cascade already treats "no next panel" as a
  no-op, not an error). Verified via `.format()` smoke test (rule text now
  says "at least 1 shot") and `suggest_shots_per_scene(10, 10)` now
  correctly returns "about 1 shot(s) per scene" instead of being floored to 2.
- 2026-07-09 (later 6) — **Prompt-pitfall audit: lost-in-the-middle sandwiching +
  priority-conflict tie-breakers across every agent prompt.** User asked
  directly whether any prompt suffers from "lost in the middle" (Liu et al.
  2023 — LLMs attend to a long context's start/end far better than its
  middle) or unresolved "priority conflicts" (two rules pulling opposite
  ways with no stated precedence). Audited all 13 `reel/agents/*.py` prompt
  templates. The seven long-context, multi-rule prompts (`scenes.py`,
  `casting.py`, `soundscape.py`, `visuals.py`, `cinematography.py`,
  `screenplay.py`, `storyboard.py` — each with 6-17 rule bullets ahead of a
  large data block) each gained a short "Before you respond, re-check
  against [data] above" block at the true end of the prompt (after the
  last data placeholder), restating the highest-stakes rules a second time
  where recency actually helps — verified against this project's own prior
  incident history in the session log (dropped voiceover lines, unused
  visual_filter/sound_events/key_props, hallucinated `source_line`s) as
  exactly the class of drift this mitigates. Found and fixed three real
  priority conflicts along the way: `cinematography.py`'s "motifs should
  recur/develop across scenes" directly opposed "vary angle/lens for a
  shared location's establishing shot" for that one shot type — added an
  explicit TIE-BREAKER (location-distinctness wins for establishing/wide
  shots at a shared location; motif development happens in the scene's
  other shots). `screenplay.py`'s camera-coverage-derived shots had no
  stated precedence against source-material fidelity if the two disagreed
  — added SOURCE OVER COVERAGE (adapt or drop the shot rather than invent
  content to fill a suggested camera set-up). `casting.py`'s "invent the
  actor" instruction read as if it might license inventing physical
  attributes too, in tension with the adjacent STORY FIDELITY rule —
  clarified as two different scopes (invent WHO plays the role; ground
  WHICH attributes that person has in the source) rather than an actual
  conflict. Also clarified a false-conflict in `storyboard.py`: "image_prompt
  is self-contained" vs. "don't re-describe the location fully in every
  panel" — self-containment covers character look/camera grammar/action;
  the location gets a brief anchor after its first-panel establishment, and
  that's still fully self-contained for rendering purposes. Reviewed but
  deliberately left unchanged: the four short single-call prompts
  (`characters.py`, `structure.py`, `genre.py`, `moodboard.py`, source
  text capped 2.5-12k chars, minimal rule counts, already schema/rules-
  then-data-last) and the two grader prompts (`fidelity.py`, `revision.py`,
  same shape) — none has the rule-count/context-length combination the
  lost-in-the-middle literature actually warns about, so editing them would
  have been unmotivated churn. `ingest.py` has no LLM prompt at all (pure
  text extraction/chunking), not audited for this reason. Every changed
  file verified via `py_compile` plus a direct `.format()` smoke test
  confirming the reminder block renders and is the actual last content in
  the assembled prompt string. Nothing committed yet at time of writing.
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
