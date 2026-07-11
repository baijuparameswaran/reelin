# Agents

> Reference for every module in `reel/agents/` — what it takes, what it
> produces, whether it's steered/graded, and whether it calls an LLM at all.
> Architecture/pipeline-wiring lives in [ARCHITECTURE.md](ARCHITECTURE.md);
> current status and history live in [PROGRESS.md](PROGRESS.md).

## How to read this file
- **Steered** = receives the process-wide creative direction (`llm.set_direction`,
  composed from `genre.guidance()` + `moodboard.guidance()`) prepended to its
  system message. Creative agents calling `llm.generate` directly are steered;
  graders calling `reel.models.text` pass `steer=False` and stay neutral.
- **Scoped revision** = accepts `existing: dict | None` + `revise_keys: set | None`
  kwargs (the `revision_merge.merge_by_key` pattern) so `python -m reel.cli revise`
  can regenerate just the targeted scene numbers / character-or-location names,
  splicing everything else back in byte-identical from `existing`.
- **Sandwiched prompt** = the prompt template ends with a "Before you respond,
  re-check against [data] above" block restating its highest-stakes rules a
  second time, in addition to stating them up front — mitigates the
  lost-in-the-middle effect (Liu et al. 2023) for prompts with a large data
  block and many rules. See the 2026-07-09 "Prompt-pitfall audit" entry in
  PROGRESS.md for which prompts needed this and why.

## Pipeline order

`ingest → genre → structure ‖ characters → moodboard → scenes → casting →
soundscape ‖ visuals ‖ cinematography → screenplay → storyboard → render`

(`‖` = runs concurrently when `runtime.max_parallel_agents` allows it —
see ARCHITECTURE.md's "Conventions & decisions" for the constraint.)

## Reference table

| Agent | File | LLM? | Steered? | Graded? | Scoped revision? | Sandwiched? |
|---|---|---|---|---|---|---|
| ingest | `ingest.py` | no | — | — | — | — (no prompt) |
| genre | `genre.py` | yes (open) | n/a (sets direction) | n/a (is the grader for others) | no | no (short, data-last) |
| structure | `structure.py` | yes (open) | yes | yes (fidelity) | no | no (short, data-last) |
| characters | `characters.py` | yes (open) | yes | yes (fidelity) | yes (`name`) | no (short, data-last) |
| moodboard | `moodboard.py` | yes (open) | n/a (sets direction) | no | no | no (short, data-last) |
| scenes | `scenes.py` | yes (open) | yes | yes (fidelity + scene-alignment) | yes (`number`) | **yes** |
| casting | `casting.py` | yes (open) | yes | yes (fidelity) | yes (`name`) | **yes** |
| soundscape | `soundscape.py` | yes (open) | yes | yes (fidelity + scene-alignment) | yes (`scene_number`) | **yes** |
| visuals | `visuals.py` | yes (open) | yes | yes (fidelity + scene-alignment) | yes (`scene_number`) | **yes** |
| cinematography | `cinematography.py` | yes (open) | yes | yes (fidelity + scene-alignment) | yes (`scene_number`) | **yes** |
| screenplay | `screenplay.py` | yes (open) | yes | yes (fidelity + scene-alignment) | yes (`number`, per-scene loop) | **yes** |
| storyboard | `storyboard.py` | only if `feedback` given | yes (LLM path only) | yes (fidelity + scene-alignment) | yes (`scene_number`, per-scene loop) | **yes** (LLM path) |
| fidelity | `fidelity.py` | yes (open, qualitative check) + deterministic (scene-alignment check) | no (neutral grader) | — (is the grader) | n/a | no (short, data-last) |
| revision | `revision.py` | yes (`suggest_ripple_scenes`) + deterministic (`is_drastic_identity_change`) | no (neutral) | — | n/a | no (short, data-last) |

Every LLM call in this table routes through the **open** Ollama models
(`reel/llm.py` / `reel/models.py`) — Gemini is used only for image (`imagegen.py`)
and video (`i2v.py`/`gemini.py`), never for text. See ARCHITECTURE.md's
provider-policy bullet.

## Per-agent notes

- **`ingest.py`** — no LLM. `ingest(path)` loads/chunks the source (`.txt`,
  `.pdf` via `_extract_pdf`); `chunk_text`/`scene_source_context` give every
  later per-scene agent the specific passage a scene maps to, instead of a
  generic head-truncation.
- **`genre.py`** — `resolve_genre` fixes one genre for the whole run
  (priority: `--genre` > config > auto-inferred from the story), then
  `guidance()` feeds the steering hook and `enforce_stage()`/`score_pipeline()`
  grade every other stage's genre alignment. See ARCHITECTURE.md for the
  full steer-vs-grade split.
- **`structure.py`** — one call, no rules list to speak of (logline/genre/
  themes/tone/three-act beats/central conflict). Runs concurrently with
  `characters.py`.
- **`characters.py`** — cast extraction including non-human characters
  (animal/bird/creature), collapsing only genuinely undetailed background
  figures into a `"group"` entry.
- **`moodboard.py`** — the film-wide visual-tone bible (palette/light/
  texture/atmosphere/influences), run once right after `structure`; its
  render-ready `tiles` are capped to `--max-scenes` since they're an actual
  media-generation step (unlike its film-wide aesthetic fields).
- **`scenes.py`** — segments the story into a numbered scene list; the single
  most consequential stage for downstream render cost, since every scene
  becomes at least one rendered video clip. Source text is the *only*
  authority (rule 1), structural beats are a secondary ordering hint that
  loses on conflict (rule 6), and rule 9 (MINIMIZE SCENE COUNT) actively
  steers toward the fewest scenes that can still tell the story faithfully —
  merging consecutive beats that share a location and continuous time,
  splitting only on a real location/time/purpose change. Every scene also
  carries a `location` string, identical across every scene set in the same
  place — the anchor `casting.py` uses to cast locations — and a
  deterministically-computed (no LLM) `source_excerpt` + `word_count`
  (`_attach_source_excerpts`): the actual contiguous portion of the story
  text that scene covers, a strict superset of the short `source_line`
  anchor. `ingest.scene_source_context` prefers `source_excerpt` over the
  coarser chunk-based join whenever it's present, so `screenplay.py`/
  `storyboard.py` ground their per-scene prompts in the precise passage
  rather than an approximate chunk. `_reconcile_character_names` is a
  deterministic safety net on top of the prompt instruction to reuse
  `characters.json`'s settled names. Every scene also carries `props`
  (rule 10) — notable physical objects the source text explicitly mentions
  for that scene — the earliest, source-grounded point a prop enters the
  pipeline; see `casting.py` and `visuals.py` below for how it's used
  downstream. Its scoped-revision support (`existing`/`revise_keys`) is the
  one scene-keyed artifact where `revise_keys` can ADD a genuinely new scene
  number, not just modify an existing one — `merge_by_key` appends it, then
  `segment_scenes` re-sorts the merged list by `number` so it lands in
  narrative order rather than always at the end. DELETING a scene isn't
  handled inside `segment_scenes` itself — that's `cli._revise_one`'s
  `"scenes"` branch (see ARCHITECTURE.md's "A DIRECT hand-edit to
  `scenes.json`..." bullet), which strips the deleted number out of every
  OTHER scene-keyed artifact directly, since `merge_by_key` can only add or
  replace a key, never remove one.
- **`casting.py`** — locks each character's on-screen visual form: an
  `actor` block (the performer's own intrinsic look) plus a `character`
  block (that actor aged/costumed into the role). Also casts every distinct
  `location` from `scenes.json` (`kind: "location"`, no `actor` layer — the
  inverse isolation rule: show the space's architecture, not a person).
  `character.visual_prompt` is the identity seed image generation renders
  and Veo anchors to — its ISOLATION rule (no scene/prop/location baked in)
  is the single most load-bearing rule in this codebase's prompts.
  `_location_entries` aggregates every scene's `props` per location into
  `recurring_props` — a prop mentioned in more than one scene at that place
  (or the only scene, if it never recurs) is a plausible FIXED fixture
  (a bar's brass mirror), unlike a single-scene-only prop at a
  multi-scene location, which is presumed transient/action-specific and
  deliberately excluded so it doesn't get wrongly baked into a reference
  image reused by every OTHER scene at that place. The PROMPT still tells
  the model to treat `recurring_props` as candidates, not a mandate — only
  genuinely architectural items should make it into the rendered
  `visual_prompt`. Also casts recurring PROPS in their own right
  (`kind: "prop"`, `_prop_entries` — any prop appearing in 2+ scenes
  ANYWHERE in the story, not scoped to one location, since a portable prop
  like a character's watch travels): a VERY DESCRIPTIVE, isolation-strict
  `visual_prompt` (exact material/color/size/condition/marks) is the
  text-only substitute for Veo's lack of cross-generation memory — without
  a locked description, the same named prop can render as a visibly
  different object every time it recurs. Gets a rendered reference image
  too, for free, via the already kind-agnostic `pipeline._render_casting_images`.
- **`soundscape.py` / `visuals.py` / `cinematography.py`** — the three
  per-scene "design" agents (score/ambient, color/lighting/props, camera
  coverage respectively), each processing every scene in one call for
  cross-scene continuity. All three share the same location-consistency
  rule (scenes sharing a `location` share its base look/sound; only
  mood/specific events vary) — `cinematography.py`'s version has an explicit
  TIE-BREAKER against its own motif-development rule (see the audit entry
  in PROGRESS.md). `visuals.py`'s `key_props` (props with genuine dramatic/
  thematic weight, each with a `function`) is now grounded by `scenes.py`'s
  plain `props` inventory when present, rather than invented from nothing —
  and, as of 2026-07-10, actually reaches the rendered video: `storyboard.py`
  copies prop NAMES into `visual_overview.key_props`, which
  `pipeline._panel_context` resolves against `casting.py`'s cast prop
  entries (falling back to the bare name for an uncast, single-scene prop)
  and folds into every panel's Veo Context section. Before that wiring, a
  prop identified here never left `visuals.json` at all.
- **`screenplay.py`** — drafts Fountain-formatted shots + fully attributed
  dialogue, one LLM call **per scene** (not all-scenes-in-one-call like the
  three above), favoring voice-over over on-screen dialogue economy. Takes
  `casting` for a locked on-screen-look block so action stays true to what's
  actually rendered. SOURCE OVER COVERAGE tie-breaker: camera coverage never
  licenses inventing content the source doesn't support.
- **`storyboard.py`** — the fusion stage; merges every upstream artifact
  into a production-ready board. **Deterministic by default** (see its
  module docstring's "BUILD PATH" section) — `_build_scene_board` constructs
  the entire scene from structured fields with zero LLM calls, since a
  fusion stage isn't supposed to make new creative decisions. The LLM path
  (`_llm_generate_scene`, the original full-generation prompt) only runs
  when `feedback` (a directed creative note) is given, since applying free
  text needs actual judgment the deterministic builder can't provide.
  `image_prompt` here is a fallback only — the real Veo prompt for an actual
  render is reconstructed from structured fields by
  `pipeline._five_part_veo_prompt`, not read from here.
- **`fidelity.py`** — two distinct, deliberately separate checks: story
  fidelity (`check_stage`/`check_alignment`/`score_pipeline`, a qualitative
  LLM judgment of drift/omissions/contradictions vs. the source) and
  scene-structure alignment (`check_scene_alignment`/`strip_orphan_scenes`,
  deterministic — does this stage's scene-keyed data actually match
  `scenes.json`) which self-heals automatically inside `pipeline.run_group`
  before the operator ever sees the gate.
- **`revision.py`** — powers `python -m reel.cli revise`. `identify_source_text_changes`
  scopes a raw story-TEXT edit down to the scene numbers it actually
  affects (given `artifact_diff`'s deterministic candidate pre-filter +
  compact paragraph diff), deciding `drastic` (implies a scene should be
  added/removed) vs. a scoped `changed_scene_numbers` list — fails safe to
  `drastic` on any malformed/ambiguous response, and sanitizes every
  returned number against scenes that actually exist. Runs on
  `agent_profiles.revision`, the largest local tier (`quality_high`) by
  default, since this needs to correlate a diff against every existing
  scene reliably. `suggest_ripple_scenes` is a separate, advisory-only LLM
  call flagging other scenes that might depend on an edit (never
  auto-applied); `is_drastic_identity_change` is a deterministic `difflib`
  heuristic that decides whether an edited character/location description
  drifted enough to warrant re-casting vs. preserving the existing locked
  identity/image.

## Adding a new agent

1. Module in `reel/agents/`, `SYSTEM` + `PROMPT` constants, a public
   `do_thing(...)` function calling `llm.generate`/`models.text`.
2. If it's a per-scene or per-name artifact, add it to
   `reel/artifact_diff.py`'s `SCENE_KEYED_ARTIFACTS`/`NAME_KEYED_ARTIFACTS`
   and give it `existing=`/`revise_keys=` kwargs following the pattern in
   any of the scene-keyed agents above (splice via `revision_merge.merge_by_key`).
3. Register it as a `Stage` in `reel/stages.py` (inputs, produces) so
   `run_stage`/`downstream_of`/the `revise` command all see it.
4. Wire it into `reel/pipeline.py`'s `run()` (a `_spec(...)` entry in the
   appropriate concurrency group) if it's part of the main pipeline.
5. If its prompt has more than ~5 rules ahead of a large data block, sandwich
   it (see the "Sandwiched prompt" note above) — cheaper to do at write time
   than to rediscover the need later.
