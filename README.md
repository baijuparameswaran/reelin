# reel

A multi-modal, agentic pipeline that turns source material (a book, short story,
or script) into production-ready creative assets — ultimately a movie / reel /
web-series episode — navigating the phases of adaptation, some in parallel.

Built to run on **locally-hosted open LLMs** (via [Ollama](https://ollama.com)),
developed in slow, steady iterations.

## Iteration 1 — "screenplay material"

The agent set converts raw text into screenplay material plus a full creative
design (cast look, score, art, camera, and a per-moment storyboard).

**The creative flow** — each box is an agent; same-column branches run
concurrently where the host allows it (structure ‖ characters; scenes ‖ casting;
soundscape ‖ visuals ‖ cinematography):

```
ingest ─┬─▶ structure ─▶ moodboard ─┬─▶ scenes ─┬─▶ soundscape ─────┐
        └─▶ characters ──────────────┘           ├─▶ visuals ─────────┼─▶ storyboard ─┐
                              casting ◀ characters└─▶ cinematography ──┘               ├─▶ render ─▶ assemble
                                          screenplay ◀─ scenes + all designs ──────────┘
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
| **casting** | casting director | two layers per character — an **actor** (own role-independent look) and the **character** (that actor aged/costumed into the role); the **character** is rendered to an image via Gemini (see [Character image generation](#character-image-generation-gemini)) and used as the video identity reference |
| **scenes** | screenwriter | numbered scene list (sluglines, summaries, purpose) |
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
the output still matches the source — and waits for you to either approve (press
Enter) or type feedback. Type feedback and the stage re-runs with your notes
appended to its prompt — iterate until you approve. Parallel branches are gated
one after another once they finish.

The gate is controlled in `config/models.yaml` under `hitl`:

```yaml
hitl:
  enabled: true          # false → fully automated, no prompts
  timeout_seconds: 120   # auto-approve after N idle seconds (0 = wait forever)
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

## Character image generation (Gemini)

The casting stage generates **one image per character — the character
representation** — via the [Gemini image API](https://ai.google.dev/gemini-api/docs/image-generation)
from each character's `visual_prompt`. This is the only image generation in the
pipeline, and the resulting `output/casting/<name>.png` is the **identity
reference** handed to the video stage.

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

## Scene rendering (image-to-video, Veo)

Once the storyboard and screenplay are done, the pipeline renders **scenes frame
by frame with continuity** via the [Gemini Veo API](https://ai.google.dev/gemini-api/docs/video).
For each storyboard frame it generates a short **clip** (image-to-video):

- the **first frame of a scene** is seeded from the in-frame character's
  representation image (`output/casting/<name>.png`) — the identity reference;
- **later frames** are seeded from the **previous frame's last image**, so motion
  is continuous within the scene. A scene boundary resets the chain (a cut).

`--max-scenes` (default 1, prototype) limits how many **scenes** are drafted and
rendered — but **every shot within each rendered scene is always rendered** (the
storyboard emits one frame per camera shot; the renderer never caps shots).

Output lands in `output/video/scene_NN/frame_MM.mp4` plus a `manifest.json`.

You can also render **straight from the finished artifacts**, without re-running
any LLM stage — it builds a camera-directed plan from `screenplay.fountain` +
`cinematography.json` (every drafted scene, every shot, no caps) and renders it:

```bash
python -m reel.cli render            # render the whole drafted story to video
python -m reel.cli render --fresh    # re-render (old clips backed up to output/video_prev)
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
run_stage("casting", out="output")             # needs structure + characters checkpoints
run_stage("structure", input_path="story.txt") # ingests the source first
```

Stage runs are direct (no review gate; pass `--feedback`/`feedback=` for a
revision note). Routing follows the provider policy — text stages on the open
models, image/video stages on Gemini when a key is set (see below). `reel.pipeline.run`
still runs the full pipeline with the HITL gate, concurrency, and `--resume`.

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

```yaml
moodboard:
  enabled: true      # false to skip the stage
  steer: true        # fold the moodboard into the creative-direction steering
```

Standalone: `python -m reel.cli stage moodboard`.

## Quick start

```bash
make setup          # venv + deps
make models         # show which local models each profile resolves to
make demo           # run on the bundled sample story (fast profile)
make demo RESUME=1  # continue a paused/failed sample run from its last stage
make run SRC=path/to/story.txt SCENES=3
```

The pipeline is **model-agnostic**: agents request a *profile* (`fast` /
`quality`), not a model name. If the preferred model isn't pulled, it falls back
to whatever's installed (see `config/models.yaml`). On this CPU-only host,
`--profile fast` (one model, no reloads) is much faster than the mixed default.

## Local models & the update cadence

Preferred models are **Qwen3 4B** (fast) and **Qwen3 8B** (quality). Pull/refresh
them and keep them current:

```bash
make update         # pull preferred models + smoke test  (the cadence job)
make update-all     # also pull fallback models
make install-cron   # weekly + monthly auto-update jobs
```

`scripts/update-models.sh` checks the Ollama version (Qwen3 needs a newer Ollama
than 0.6.5), pulls every model the agents depend on, runs a smoke test so an
update can't silently break the agents, and logs to `scripts/model-updates.log`.

## Hardware notes (this host)

CPU-only, **~12 GB RAM in WSL** (16 GB laptop; raised from the original 7.6 GB
cap via `%UserProfile%\.wslconfig` → `[wsl2]` / `memory=12GB`, then
`wsl --shutdown`). Implications:

- Inference is slow (a few tok/s); run as a batch, not interactively.
- Generation is **streamed**, so the request timeout is an *inactivity* window
  (max gap to the next token), not a cap on total time — a slow stage like
  `storyboard` runs as long as it needs. The first token is the long pole (it
  includes loading the model into RAM and prefilling the prompt on CPU), so the
  default `runtime.request_timeout_seconds` is **600**; raise it (or set 0 =
  wait forever) in `config/models.yaml` if a stage still trips it, then
  `--resume`.
- Each stage's artifact is written to `output/` the moment you approve it, so a
  failure or timeout in a later stage never discards the work already done —
  re-run with `--resume` to continue.
- `num_ctx` is capped at 4096 to fit comfortably in RAM. Larger inputs are
  truncated for now (chunking is a later iteration); at 12 GB, raising to 8192
  is now plausible but untested.
- The `quality` tier (8B, ~5.7 GB) now loads without swapping, so it's
  comfortable at 12 GB RAM.

See `CLAUDE.md` for project vision, decisions, and session-to-session state.
