# reel

A multi-modal, agentic pipeline that turns source material (a book, short story,
or script) into production-ready creative assets — ultimately a movie / reel /
web-series episode — navigating the phases of adaptation, some in parallel.

Built to run on **locally-hosted open LLMs** (via [Ollama](https://ollama.com)),
developed in slow, steady iterations.

## Setup — from a new, clean environment

### 1. Prerequisites

- **Python 3.10+** (any recent CPython; no other packaging tooling is used —
  `requirements.txt` / `requirements-image.txt` + a plain `venv` is the whole
  story, see below).
- **[Ollama](https://ollama.com)** — runs every text/LLM agent locally.
  - **If you have an NVIDIA GPU, install via the official installer script,
    not your distro's `snap`/`apt` package.** The Ubuntu snap build uses strict
    confinement that blocks `/dev/nvidia*`, so it silently falls back to
    100%-CPU inference with no error:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
  - After pulling a model (next section), confirm the GPU is actually being
    used: `ollama ps` — look for a non-zero "Size VRAM" against the loaded
    model. All-CPU still works, just much slower (a few tok/s vs. tens+ on GPU).
- **`ffmpeg`** on `PATH` — optional but recommended. Needed for stitching
  rendered clips into `output/video/movie.mp4` and for subtitle/shot-label
  overlays; without it those two steps are skipped with a warning, everything
  else is unaffected. `apt install ffmpeg` / `brew install ffmpeg` / etc.
- **A Gemini API key** — optional, only for character-image + Veo video
  rendering (see below). The text pipeline (ingest → … → screenplay/storyboard)
  runs completely without one.

### 2. Install

```bash
git clone <this-repo-url> reel && cd reel
make setup          # creates .venv/, installs requirements.txt, then pulls the local models
source .venv/bin/activate
make models         # verify: shows which installed model each profile ('fast'/'quality'/…) resolves to
```

`make setup` only touches `.venv/` and installs from `requirements.txt` — it
never writes a lockfile or a second dependency manifest; that's the single
source of truth for what the environment needs.

It then matches each profile's model to your GPU/RAM and pulls the result.
`config/models.yaml` is tuned for one specific machine (8 GB VRAM, 24 GB RAM),
so on a smaller box `make setup` writes the deltas to a gitignored
`config/models.local.yaml` — e.g. `quality_high` drops off the 19 GB
`qwen3:30b` — and pulls that instead. It only ever picks from a profile's own
declared fallbacks, and on a host the tracked config already fits it writes
nothing. Run it alone with `make hardware-config` (add nothing to see the plan:
`python -m reel.hardware_config`), and delete the file to go back to the
tracked defaults.

The pull itself is a multi-GB download — `make setup-models` runs just that
part, `SKIP_MODELS=1 make setup` skips it. It's skipped with a hint if `ollama`
isn't installed yet, since installing Ollama needs sudo and stays your call:

```bash
curl -fsSL https://ollama.com/install.sh | sh   # NOT the snap — it can't reach the GPU
make setup-models
```

### 3. Optional: Gemini for image + video generation

Everything else in the pipeline runs on local open models; **only** character
portraits and Veo video clips use Gemini, and only if a key is present:

```bash
make secrets CMD=set   # paste your key once; stored chmod-600 at ~/.config/reel/gemini_key
make secrets           # CMD defaults to 'status' — check whether a key is set
```

(Or set the env var directly: `export GEMINIAPIKEY=…` — also accepts
`GEMINI_API_KEY` / `GOOGLE_API_KEY`.) Without a key, the casting/video stages
no-op with a clear hint and the rest of the run is unaffected.

### 4. Optional: local (non-Gemini) image generation

Only needed if you set `image.backend: diffusers` in `config/models.yaml`
instead of the default `gemini` backend:

```bash
make setup-image    # installs diffusers/torch/etc. (needs a GPU to be practical)
```

### 5. Run

```bash
make demo                # bundled sample story, 1 scene, fast profile
make demo SCENES=all     # every scene the sample story has
make demo RESUME=1       # continue a paused/failed run from its last stage
make run SRC=path/to/story.txt SCENES=3
```

The pipeline is **model-agnostic**: agents request a *profile* (`fast` /
`quality` / …), never a model name — if the preferred model isn't pulled, it
falls back to whatever's installed (see `config/models.yaml`). On a
single-GPU/CPU-only host, `--profile fast` (one small model, no reloads
between agents) is much faster than the mixed default.

### Testing

```bash
make test   # or: python -m unittest discover -s tests -v
```

`tests/test_prompt_rules.py` validates every agent prompt against this
project's prompting conventions (sandwiching, tie-breakers, schema fields)
and the deterministic logic that backs some of those rules — no LLM or API
calls, runs in well under a second. `make demo` and `make run` both depend
on `test`, so a broken prompt/rule is caught before either spends any real
time or API quota on an actual run.

## Local models & the update cadence

Preferred models are **Qwen3 4B** (fast) and **Qwen3 8B** (quality); larger
profiles (`synthesis`, `quality_high`) may be configured for bigger hosts — see
`config/models.yaml`. Pull/refresh them and keep them current:

```bash
make update         # pull preferred models + smoke test  (the cadence job)
make update-all     # also pull fallback models
make install-cron   # weekly + monthly auto-update jobs
```

`scripts/update-models.sh` checks the Ollama version, detects your hardware and
prefers models that actually fit it (`reel/manifest.py --hardware`
/`--runnable-only`), pulls every model the agents depend on, runs a smoke test
so an update can't silently break the agents, and logs to
`scripts/model-updates.log`.

## Hardware notes

- **GPU strongly recommended** for the LLM stages (first-token latency ~2–5s on
  GPU vs. ~30–60s on CPU) and required in practice for Veo/diffusers video
  generation. CPU-only still works for the text pipeline — just run it as a
  batch, not interactively.
- Generation is **streamed**, so `runtime.request_timeout_seconds` in
  `config/models.yaml` is an *inactivity* window (max gap to the next token),
  not a cap on total time — a slow stage like `storyboard` runs as long as it
  needs. Raise it (or set `0` = wait forever) if a stage still trips it, then
  `--resume`.
- Each stage's artifact is written to `output/` the moment you approve it, so a
  failure or timeout in a later stage never discards work already done —
  re-run with `--resume` to continue.
- `num_ctx` and which models each profile targets are tuned per-profile in
  `config/models.yaml` for the RAM/VRAM available; see that file's comments
  before changing them for a very different host.

See `CLAUDE.md` (entry point) — or directly `ARCHITECTURE.md` for this
project's specific dev host and conventions, `AGENTS.md` for a per-agent
reference, and `PROGRESS.md` for current status and session-to-session
history.

## Iteration 1 — "screenplay material"

The agent set converts raw text into screenplay material plus a full creative
design (cast look, score, art, camera, and a per-moment storyboard).

**The creative flow** — each box is an agent; same-column branches run
concurrently where the host allows it (structure ‖ characters;
soundscape ‖ visuals ‖ cinematography). `scenes` and `casting` run in sequence,
not concurrently — casting also locks a visual reference per distinct scene
*location*, so it needs scenes' output first:

```
ingest ─┬─▶ structure ─▶ moodboard ─┬─▶ scenes ─▶ casting ─┬─▶ soundscape ─────┐
        └─▶ characters ──────────────┘                     ├─▶ visuals ─────────┼─▶ storyboard ─┐
                                                             └─▶ cinematography ──┘               ├─▶ render ─▶ assemble
                                          screenplay ◀─ scenes + all designs ──────────────────────┘
```

**Cross-cutting agents** — set once, they shape and police *every* stage above
(not per-scene; they thread through the whole run):

```
genre     ─ chosen up front (--genre │ config │ auto-from-story)  ─┐  steer every creative
moodboard ─ visual-tone bible, fixed right after structure        ─┴─▶ stage's prompt (one
                                                                       shared "direction")
genre     ─ scores each stage's genre alignment  ┐  shown at the review gate; below
fidelity  ─ scores each stage vs the original story ┴─ threshold → re-iterate hint
```

Genre and the moodboard **steer** generation (their conventions are injected into
each creative stage), while genre and fidelity **grade** each stage — and the
graders judge neutrally, unaffected by the steering. See
[Genre alignment](#genre-alignment), [Moodboard](#moodboard-film-wide-visual-tone-bible),
and [Story fidelity](#story-fidelity-consistency-scoring).

| Agent | Role | Output |
|-------|------|--------|
| **ingest** | — | normalized text + metadata (`title`, word count) |
| **genre** | showrunner | one genre for the film (explicit / config / inferred from the story) + its conventions; **steers every stage** and **scores per-stage alignment** |
| **structure** | story analyst | logline, genre, themes, tone, three-act beat sheet |
| **moodboard** | production designer | film-wide visual-tone bible (color story, palette, lighting mood, textures, atmosphere, influences, render-ready tiles); **steers all downstream stages** |
| **characters** | script analyst | every character — humans **and** animals/birds/creatures — each defined individually (kind, role, want, arc, appearance, voice, mannerisms); undetailed background masses collapse to one `group` |
| **scenes** | screenwriter | numbered scene list (sluglines, summaries, purpose, and a `location` — the plain name of the physical setting, identical across every scene set in the same place) |
| **casting** | casting director | two layers per character — an **actor** (own role-independent look) and the **character** (that actor aged/costumed into the role); also casts each distinct scene `location` (no actor layer — the space itself). Both are rendered to an image via Gemini (see [Character image generation](#character-image-generation-gemini)) and used as the video identity reference |
| **soundscape** | sound / score | per-scene ambient bed, audio cues, silence, emotional function |
| **visuals** | art production | per-scene color palette, lighting, filters, key props |
| **cinematography** | director of photography | per-scene shot list (type, angle, movement, lens, framing) |
| **screenplay** | screenwriter | Fountain draft — numbered shots, attributed dialogue, V.O. — carrying the **locked on-screen look** (casting) plus every design above |
| **storyboard** | storyboard artist | a frame per moment that **fuses the full detail** of every artifact (locked cast look + voice, complete art/sound/camera design, and the screenplay's own shots + dialogue) into a self-contained, render-ready prompt — this is what **drives video generation** |

> The screenplay and storyboard are the two artifacts that **drive video
> generation**, so they deliberately capture the full detail of every upstream
> stage — nothing is dropped on the way to the renderer.

Artifacts land in `output/`: `genre.json`, `moodboard.json`, `structure.json`,
`characters.json`, `casting.json`, `scenes.json`, `soundscape.json`,
`visuals.json`, `cinematography.json`, `storyboard.json`, `screenplay.fountain`,
per-stage genre/fidelity reports under `output/genre/` and `output/fidelity/`
(aggregates `genre_alignment.json` + `fidelity.json`), per-character files under
`output/characters/`, character images under `output/casting/`, scene clips under
`output/video/`, and a combined `project.json`.

### Human-in-the-loop review

After each LLM stage the pipeline pauses at a **review gate**: it prints a
summary — including that stage's **story-fidelity score** (see
[Story fidelity](#story-fidelity-consistency-scoring)) so you can judge whether
the output still matches the source — and waits for you to either approve
(press Enter), type feedback, type **`view`** to open the full output (not
just the summary, which can be truncated) in `$EDITOR`/`$VISUAL` (default
`vim`), or `stop` to pause.

`view` is both read and edit in one: close the editor without saving and the
gate just reprompts with the same result unchanged. Save a real, valid change
and it's applied as a new candidate result, **re-checked against
fidelity/genre** before the same gate comes back up (approve / feedback /
`view` again / `stop`) — an edit is never auto-approved on its own. A save
with no actual change, or content that isn't valid JSON, is discarded with a
message and the original output is left untouched.

The auto-approve timeout only counts down while waiting at the prompt — it
isn't running at all while you're inside the editor, and restarts fresh once
the gate reprompts. Type feedback and the stage re-runs with your notes
appended to its prompt — iterate until you approve. Parallel branches are
gated one after another once they finish.

The gate is controlled in `config/models.yaml` under `hitl`:

```yaml
hitl:
  enabled: true          # false → fully automated, no prompts
  timeout_seconds: 900   # auto-approve after N idle seconds (0 = wait forever)
```

Set `enabled: false` for unattended / batch runs.

### Pause & resume

Long runs are interruptible. At any review gate, type **`stop`** (or press
**Ctrl-C**) to pause — every stage you've already approved stays written in
`output/`. Pick up where you left off with `--resume`:

```bash
make run SRC=story.txt            # ... type 'stop' at a gate, or Ctrl-C
make run SRC=story.txt RESUME=1   # reloads finished stages, continues from the next
make demo RESUME=1                # same, for the bundled sample run
# or directly:
python -m reel.cli story.txt --out output --resume
```

Without `--resume` (plain `make demo` / `make run`) the pipeline starts fresh
from stage 1 and overwrites the existing checkpoints. With `--resume` it loads
each completed `output/<stage>.json` and only recomputes the first stage that
isn't done yet (and everything after it). A stage that was mid-flight when you
stopped is never half-saved — it simply re-runs.

`--max-scenes`/`--profile`/`--no-render` are **inherited automatically** on
`--resume` if you don't repeat them — the effective values from the run being
resumed are saved to `output/run_params.json` and reused, so `--resume` alone
(as printed at the pause prompt) never silently drops back to
`--max-scenes`'s default of `1` even if the original run used `--max-scenes
all`, and never silently starts rendering (spending real API quota) partway
through a run that was explicitly started with `--no-render`. Passing any of
these flags explicitly on the resume command always overrides the inherited
value (and becomes the new one remembered for any later resume). `--genre`
doesn't need this — it's already checkpointed to `output/genre.json` and
reloaded automatically on `--resume`.

### Session tracking

A full story-to-video run (ingest through the final render) is one **session**,
identified by a generated id (`<timestamp>-<random>`) written to
`output/session.json` (`session_id`, `source`, `started_at`, `status` —
`running`/`complete`/`paused`/`failed`, `resumes`). A plain `python -m reel.cli
story.txt` mints a new session; `--resume` and standalone `stage`/`render`
invocations against the same `--out` reattach to whatever session is already
there instead of minting a new one, so a multi-day run across several
`--resume`s stays one session. Every line in `output/logs/gemini_api.log` and
every `output/logs/scene_NN_veo_prompts.txt` is tagged with the active session
id, so if the same `--out` dir is ever reused for a different story, its API
calls and prompt logs are still attributable to the run that made them.

## Character + location image generation (Gemini)

The casting stage generates **one image per casting entry** — per character
(the character representation) **and** per distinct scene location — via the
[Gemini image API](https://ai.google.dev/gemini-api/docs/image-generation),
from each entry's `visual_prompt`. A character's portrait is isolated (plain
neutral backdrop, no scene/props/location baked in — it's an identity reference
reused across every scene); a location's reference is the inverse (it *should*
show the place's own architecture/decor, just with no people or scene-specific
mood baked in — it's a background plate reused across every scene set there).
This is the only image generation in the pipeline, and the resulting
`output/casting/<name>.png` files are the **identity/background references**
handed to the video stage.

```yaml
image:
  enabled: true
  backend: gemini             # gemini | diffusers | auto1111 | none
  model: gemini-3.1-flash-image   # gemini-3-pro-image | gemini-2.5-flash-image
  aspect_ratio: "3:4"
  image_size: "2K"            # 512 | 1K | 2K | 4K
```

The Gemini backend needs **no extra Python deps** (stdlib REST) but does need an
API key in the environment:

```bash
export GEMINIAPIKEY=…         # or GEMINI_API_KEY / GOOGLE_API_KEY
```

Best-effort: with no key the run continues and keeps each character's text
`visual_prompt`. (The casting data still models *actor vs. character* — see the
casting agent — but only the character is rendered.) The `diffusers`/`auto1111`
backends remain available for local/self-hosted image models
(`pip install -r requirements-image.txt` for diffusers).

Both this and video rendering (below) can also be skipped for a single run
without touching config, via `python -m reel.cli story.txt --no-render` —
every design/planning stage still runs normally (structure, characters,
scenes, soundscape, visuals, cinematography, screenplay, storyboard), only
the two stages that spend real Gemini/Veo API quota are skipped. Equivalent
in effect to setting `image.enabled: false` / `video.enabled: false`, but
scoped to just this invocation instead of a persistent config change —
useful for iterating on a story's text before committing to a render.
Inherited across `--resume` like `--max-scenes`/`--profile` (see
[Pause & resume](#pause--resume)).

## Scene rendering (image-to-video, Veo)

Once the storyboard and screenplay are done, the pipeline renders **scenes frame
by frame with continuity** via the [Gemini Veo API](https://ai.google.dev/gemini-api/docs/video).
For each storyboard frame it generates a short **clip** (image-to-video):

- the **first frame of a scene** is seeded from the in-frame character's
  representation image (`output/casting/<name>.png`) — the identity reference;
- **later frames** are seeded from the **previous frame's last image**, so motion
  is continuous within the scene. A scene boundary resets the chain (a cut);
- a "shot boundary" frame with **more than one character in it** (a scene's
  opening frame, or any frame whose in-frame cast changes from the one
  before it) instead uses Veo's `reference_images` — up to 3 identity-lock
  portraits, one per character — so every character gets grounded, not just
  whichever one a single seed image can carry. This trades away frame-to-
  frame continuity for that one frame (Veo can't do both in the same call),
  falls back to the normal single-seed path automatically if it's disabled
  or fails, and leaves every other frame unaffected. Toggle with config
  `video.multi_character_references` (default on).
- if `video.continuity_mode` is set to `extend` (native video-to-video scene
  extension, carries ambient/music audio forward too — see the config
  comment), it's only ever used when a frame's in-frame characters are the
  SAME as the previous frame's; a frame where the cast changes falls back
  to seeding/reference-images instead, so a scene never extends the wrong
  characters' continuity into a shot that doesn't feature them.

`--max-scenes` (default 1, prototype; pass `all` for every scene) limits how
many scenes get **rendered** — casting-image generation and this video render
step. It does NOT limit drafting: screenplay, storyboard, soundscape, visuals,
and cinematography always process every scene in the story, since those are
design/planning stages, not media generation — only the actual rendering
stages (image/video API calls) restrict themselves to `max_scenes`. Within a
rendered scene, **every shot is always rendered** (the storyboard emits one
frame per camera shot; the renderer never caps shots).

**`--target-duration N`** (default 45 seconds) sets a target total runtime for
the rendered movie. It's engine-independent guidance — Veo is only one of
several supported video backends (`reel/i2v.py` also supports diffusers-based
LTX/Wan/CogVideoX and a remote comfyui/http endpoint) — so it never forces an
exact scene/shot count; `reel/duration_budget.py` converts it into planning
text ("aim for about N scenes", "about M shots per scene") fed to the scenes
and cinematography stages, which still make the actual creative call. The
storyboard's own per-panel duration estimate then becomes each rendered
clip's *requested* length, translated by whichever backend is configured:
Veo 3.1 only accepts exactly 4, 6, or 8 seconds (not a continuous range, and
must be 8 outside 720p resolution), so the gemini/veo backend rounds to the
nearest valid value; other backends honor the request directly as a frame
count. The gate for the storyboard stage shows the plan's estimated total
runtime against the target, flagging when they're more than ~15% apart —
informational only, it never blocks. Omitted on `--resume`, it inherits
whatever the run being resumed actually used, same as `--max-scenes`/
`--profile`.

**Every prompt actually sent to Veo follows a fixed five-part formula, always
in this order** (per Google's official Veo 3.1 prompting guide —
[cloud.google.com/blog/.../ultimate-prompting-guide-for-veo-3-1](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1)):
`[Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]`.
Each section is assembled from an explicit dict built from **structured
data** — casting.json's locked character/location descriptions, the scene's
`visual_overview`, the panel's own camera fields — not the storyboard agent's
free-text `image_prompt` (which has no guaranteed internal order; that field
still exists and is used as a fallback when no casting context is available,
e.g. a standalone `gen-video` prompt). Depth-of-field terms ("deep focus"/
"shallow focus") are intentionally not asserted anywhere in this assembly —
left to whatever the source content says. See `pipeline._five_part_veo_prompt`.

Output lands in `output/video/scene_NN/frame_MM.mp4` plus a `manifest.json`. After
the clips are rendered they are **stitched into a single movie** —
`output/video/movie.mp4` — concatenated in scene-then-frame order with ffmpeg
(fast lossless stream-copy, falling back to a re-encode if the clips don't match;
Veo's native audio is preserved). The `movie` path is recorded in `manifest.json`.

You can also render **straight from the finished artifacts**, without re-running
any LLM stage — it builds a camera-directed plan from `screenplay.fountain` +
`cinematography.json` (every drafted scene, every shot, no caps), renders it, and
stitches the movie:

```bash
python -m reel.cli render            # render the whole drafted story to video + movie.mp4
python -m reel.cli render --fresh    # re-render (old clips backed up to output/video_prev)
python -m reel.cli stitch            # just re-stitch existing clips → movie.mp4 (no render)
```

```yaml
video:
  enabled: true
  backend: gemini             # gemini/veo | diffusers (GPU) | comfyui/http | none
  model: veo-3.1-fast-generate-preview
  aspect_ratio: "16:9"
  resolution: "720p"
  continuity: true            # chain each clip from the previous frame's last image
```

Veo uses the same `GEMINIAPIKEY`. The preview tier rate-limits aggressively, so the
client **retries with backoff** on HTTP 429/5xx *and* on transient Veo operation
errors (internal/unavailable). Best-effort: with no key the run finishes without
clips. The `diffusers` (LTX-Video/Wan via `pipeline_class`) and `comfyui`/`http`
backends remain as self-hosted/remote-GPU alternatives — see `reel/i2v.py`.

## Running individual stages

Every stage is declared once in a registry (`reel/stages.py`) with the inputs it
depends on, so you can run **one stage on its own** instead of the whole pipeline.
Each stage loads its inputs from prior `output/<input>.json` checkpoints (ingesting
the source on demand) and writes its own artifact.

```bash
python -m reel.cli stages                      # list stages + their inputs
python -m reel.cli stage structure story.txt   # ingest + run just 'structure'
python -m reel.cli stage scenes                # uses output/source.json + structure.json
python -m reel.cli stage screenplay --feedback "more voice-over"
python -m reel.cli stage fidelity              # check the draft vs the original story
```

Equivalently in Python:

```python
from reel.stages import run_stage
run_stage("casting", out="output")             # needs structure + characters; picks up
                                                # scenes.json too if present (locations)
run_stage("structure", input_path="story.txt") # ingests the source first
```

Stage runs are direct (no review gate; pass `--feedback`/`feedback=` for a
revision note). Routing follows the provider policy — text stages on the open
models, image/video stages on Gemini when a key is set (see below). `reel.pipeline.run`
still runs the full pipeline with the HITL gate, concurrency, and `--resume`.

## Revising a completed run

`--feedback` (above) always regenerates a stage's *entire* output. For a
targeted change — fix one scene, tweak one character's look, adjust one video
panel — after a run has finished (or paused), use the interactive revision
loop instead:

```bash
python -m reel.cli revise --out output
```

You don't have to run this as a separate command, either: once a full
`python -m reel.cli story.txt` run finishes (video render included, if
enabled), it offers to drop you straight into the same revision loop —
type `revise` to start editing right away, or just press Enter to exit.
(Skipped automatically for unattended/non-interactive runs — `hitl.enabled:
false` in config, or no TTY.)

Each round: pick any stage (or `source` for the raw ingested story text) to
hand-edit in `$EDITOR` — the same "view/edit" mechanism the live pipeline gate
already uses. On save, reel figures out **exactly what changed** (which scene
numbers, character/location names, or storyboard panels — a deterministic
diff for any JSON stage, no LLM needed) and proposes re-running only the
stages that are actually downstream of the edit, scoped to just the changed
keys — not the whole pipeline from scratch. You confirm before anything runs.

A few things this is deliberately careful about:

- **Character/location visual identity is preserved by default.** Editing one
  character's casting entry doesn't cause the others to be silently reworded
  and re-rendered — every downstream agent still gets full story context (for
  coherence), but only the entries you actually targeted are trusted from its
  response; everything else comes back byte-for-byte identical, so an
  untouched character's rendered reference image is reused, not regenerated.
  If your edit changes a character/location's description enough that it
  might imply a different look, you get a warning and the OLD image is kept
  by default — reworking it into the new look isn't attempted automatically.
- **Even within a scene/character that IS being regenerated, only the
  fields that actually changed are accepted — field by field, not the
  whole entry wholesale.** The model still sees full-story context and can
  incidentally reword a field that had nothing to do with your actual
  edit; each field of its response is compared against the existing value,
  and only genuinely different ones are taken — everything else keeps its
  old value verbatim, so a targeted edit to one field of a scene/character
  doesn't quietly cause every other field on it to drift too.
- **Video panels re-render cheaply and predictably.** Editing one storyboard
  panel re-renders that panel plus exactly the one immediately after it (to
  keep that seam visually smooth), then stops — it does not cascade through
  the rest of the scene, so the API cost of a small edit stays small. Start/
  end frames for every rendered panel are recorded in
  `output/video/manifest.json`, which is what makes this targeted re-render
  (and re-stitching it back into the scene/movie) possible.
- **You can add or delete a scene directly in `scenes.json`, not just edit
  one.** Hand-editing `scenes` to insert a new scene number scopes exactly
  like any other edit — it (and whatever's downstream of it) gets
  generated, everything else stays untouched. Deleting a scene number is
  propagated too: it's stripped out of every other stage that tracks scenes
  (soundscape/visuals/cinematography/screenplay/storyboard),
  `screenplay.fountain` is regenerated to match, and — if that scene was
  already rendered — it's dropped from the video manifest and the final
  movie is re-stitched without it (the old clip files themselves are left
  on disk, not deleted, in case you want them back). None of this costs an
  API call: it's local bookkeeping, so it happens even in a `render: off`
  session. Reordering scenes (renumbering an existing one) isn't supported
  — that's still treated as a larger ("drastic") change, falling back to a
  full downstream regenerate with a warning and confirmation prompt, not a
  silent partial revision.
- **Editing the raw story text is scoped too, not just a full regen.**
  Hand-edit `source` and reel first figures out WHICH scenes the change
  actually touches — a deterministic pre-check (does each scene's own
  portion of the story still appear, word for word, in the new text?)
  followed by an LLM pass that reads a compact diff of just the changed
  paragraphs (not the whole story twice) and confirms/refines the affected
  scene numbers, on the largest local model available for reliability. Only
  those scenes — and whatever's actually downstream of them — get
  regenerated; an edit that reads as adding or removing an event still
  falls back to the full regen (this scoped path doesn't yet infer a
  concrete new/deleted scene from prose the way a direct `scenes.json` edit
  above does).
- **One session, many rounds.** The revision loop reattaches to the run's
  existing `output/session.json` (see [Session tracking](#session-tracking)
  above) and keeps it `running` across every round; it's only marked
  `complete`/`paused` when you type `quit`/`exit` or `pause` (or Ctrl-C).
- **The original run's attributes carry over automatically** — `--profile`,
  `--target-duration`, and the genre/moodboard creative direction are all
  reloaded from the completed run before any stage regenerates, so a
  revision doesn't silently drop back to slower/default models or lose the
  genre steering that shaped the rest of the film.
- **A revision always operates as if `--max-scenes all` had been used**,
  regardless of what the original run's `--max-scenes` actually was — it's
  the one attribute deliberately NOT inherited. Which scenes actually get
  regenerated is still controlled entirely by the diff-based scoping above
  (`revise_keys`), not by `--max-scenes`; this just removes an unrelated
  cap that would otherwise make a scene the diff correctly identifies as
  affected invisible to rendering just because the original run used a
  smaller prototype-scale `--max-scenes N`. (Screenplay was already never
  capped by `--max-scenes`, in a fresh run or a revision either way.)
- **The scope evaluation prints as soon as it's known, and each downstream
  stage says what it's doing.** Right after a diff (or, for a source-text
  edit, the LLM confirmation pass) identifies which scenes/names are
  affected, that's printed immediately — before any ripple-suggestion
  prompts or the final confirm — so you see the scope up front, not just
  at the very end. Then, as each downstream stage runs, it prints a line
  saying what it's about to do: `regenerating [...]` for a scoped subset,
  `full regen` when no scoped translation applies, `nothing to
  regenerate` when a stage has nothing left to do (e.g. only scenes were
  deleted), or a skip notice for a rendering stage while `render` is off.
- **Casting + all image/video rendering are skipped by default.** Those are
  the only stages a revision can trigger that cost real API spend (Gemini
  image, Veo video) — every other stage is local-LLM-only and free, so
  iterating on text content doesn't pay for a re-render on every round.
  Enable it with `revise --render`, or type `render on` / `render off`
  inside the loop at any point to toggle it mid-session (the menu header
  always shows the current setting). A skipped stage's existing output is
  left as-is, not deleted, so anything downstream that needs it still works
  — just against the last rendered state until you turn rendering back on.

## Story fidelity (consistency scoring)

As the pipeline transforms the source through structure → … → screenplay →
storyboard, drift can creep in. For **each** stage, a fidelity agent compares that
stage's output back to the **original story** and scores it — so you can see
exactly where (and how badly) an adaptation diverges.

The score is computed **before each review gate** and shown in the gate readout,
so it directly informs your approve / re-iterate decision:

```
  story fidelity: DRIFTING  62/100  ⚠ below 70 — consider re-running with feedback
    drift: villagers' nickname dropped; Edith's secret choice softened
```

(The threshold for the hint is `fidelity.min_score`, default 70.)

- Per stage → `output/fidelity/<stage>.json`: a `fidelity_score` (0-100) plus
  `drift` / `omissions` / `contradictions` and a `verdict`.
- Aggregate → `output/fidelity.json` (and `project.json`): the **pipeline score**

  ```
  overall = round( 0.5 · mean(stage scores)  +  0.5 · min(stage scores) )
  ```

  i.e. half the average quality, half the weakest stage (one badly drifting stage
  caps story consistency). Verdict bands: **≥85 aligned · 70–84 mostly aligned ·
  50–69 drifting · <50 misaligned**.

The fidelity agent runs on the **open models** (never Gemini, per the provider
policy). It's best-effort (a failed check never blocks the run) and adds one
model call per stage — toggle it off for faster runs:

```yaml
fidelity:
  per_stage: true      # false to skip the per-stage consistency checks
```

You can also run it standalone: `python -m reel.cli stage fidelity` (a holistic
screenplay+storyboard-vs-story check).

## Genre alignment

The pipeline fixes **one genre** for the run and keeps every department true to it.
The genre comes from (in priority order): the `--genre` flag, the config
`genre.value`, or — when that is `auto` — it is **inferred from the storyline**
itself before any creative stage runs.

```bash
python -m reel.cli story.txt --genre "noir thriller"   # force a genre
python -m reel.cli story.txt                           # config / auto-detect
```

The genre agent then does two things:

- **Steers** every creative stage — its conventions (tone, visual/sound language,
  pacing, dialogue) are injected into each stage's prompt so generation leans into
  the genre. (Steering is invisible to the graders: fidelity and the genre check
  itself judge neutrally.)
- **Enforces** alignment — after each stage, it scores how on-genre the output is,
  shown in the review gate next to the fidelity score:

  ```
    genre [noir thriller]: MOSTLY ON-GENRE  74/100
    story fidelity: ALIGNED  88/100
  ```

  Below `genre.min_score` (default 70) flags a re-iterate hint, exactly like
  fidelity. Per-stage reports → `output/genre/<stage>.json`; the resolved spec →
  `output/genre.json`; the aggregate (same `0.5·mean + 0.5·min` score) →
  `output/genre_alignment.json` and `project.json`.

Runs on the **open models** (never Gemini, per policy). Best-effort and
toggleable:

```yaml
genre:
  value: auto        # explicit genre name, or "auto" to infer from the storyline
  steer: true        # inject genre conventions into each creative stage's prompt
  enforce: true      # per-stage genre-alignment checks
  min_score: 70
```

## Moodboard (film-wide visual-tone bible)

Right after structure, a **moodboard** agent fixes the film's single aesthetic —
color story, palette, lighting mood, textures, atmosphere, visual influences,
wardrobe and sound mood, plus render-ready reference `tiles`. It sits one level
*above* the per-scene `visuals` stage: like genre, it's one cross-cutting
reference, reviewed at its own gate and saved to `output/moodboard.json`.

It then **steers every downstream creative stage** — casting, soundscape, visuals,
cinematography, screenplay, storyboard — by folding its directive into the same
creative-direction hook the genre uses, so they all compose toward one look
(no per-agent changes; the graders stay neutral). Runs on the **open models**.

Its reference **tiles** (capped to `--max-scenes`) are **rendered to images** via
the image backend — **Gemini when a key is configured**, else the open image
backend — into `output/moodboard/tile_NN.png` (the board's palette + lighting are
appended to each tile prompt so they cohere). Per policy only the *tiles* (images)
use the image provider; the moodboard spec itself stays on the open text models.
Best-effort: with no image backend the run keeps the tile prompts. Render them
standalone with `python -m reel.cli stage moodboard_tiles`.

```yaml
moodboard:
  enabled: true      # false to skip the stage
  steer: true        # fold the moodboard into the creative-direction steering
```

Standalone: `python -m reel.cli stage moodboard`.
