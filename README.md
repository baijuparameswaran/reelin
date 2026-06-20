# reel

A multi-modal, agentic pipeline that turns source material (a book, short story,
or script) into production-ready creative assets — ultimately a movie / reel /
web-series episode — navigating the phases of adaptation, some in parallel.

Built to run on **locally-hosted open LLMs** (via [Ollama](https://ollama.com)),
developed in slow, steady iterations.

## Iteration 1 — "screenplay material"

The agent set converts raw text into screenplay material plus a full creative
design (cast look, score, art, camera, and a per-moment storyboard):

```
ingest ─┬─▶ structure ──┐
        └─▶ characters ──┴─▶ scenes ─┬─▶ soundscape ─────┐
                        └─▶ casting   ├─▶ visuals ─────────┼─▶ storyboard ─┐
                                      └─▶ cinematography ──┘               ├─▶ assemble
                                                          screenplay ──────┘
```

Branches on the same column run concurrently where the host allows it
(structure ‖ characters; scenes ‖ casting; soundscape ‖ visuals ‖ cinematography).

| Agent | Role | Output |
|-------|------|--------|
| **ingest** | — | normalized text + metadata (`title`, word count) |
| **structure** | story analyst | logline, genre, themes, tone, three-act beat sheet |
| **characters** | script analyst | every character — humans **and** animals/birds/creatures — each defined individually (kind, role, want, arc, appearance, voice, mannerisms); undetailed background masses collapse to one `group` |
| **casting** | casting director | each character's locked on-screen visual form + a text-to-image prompt (people, animals, and groups alike) |
| **scenes** | screenwriter | numbered scene list (sluglines, summaries, purpose) |
| **soundscape** | sound / score | per-scene ambient bed, audio cues, silence, emotional function |
| **visuals** | art production | per-scene color palette, lighting, filters, key props |
| **cinematography** | director of photography | per-scene shot list (type, angle, movement, lens, framing) |
| **storyboard** | storyboard artist | a visual image per moment, fusing cast + art + camera + score, each with emotional & audio attributes |
| **screenplay** | screenwriter | Fountain-formatted draft pages (informed by every design above) |

Artifacts land in `output/`: `structure.json`, `characters.json`, `casting.json`,
`scenes.json`, `soundscape.json`, `visuals.json`, `cinematography.json`,
`storyboard.json`, `screenplay.fountain`, per-character files under
`output/characters/`, and a combined `project.json`.

### Human-in-the-loop review

After each LLM stage the pipeline pauses at a **review gate**: it prints a
summary and waits for you to either approve (press Enter) or type feedback. Type
feedback and the stage re-runs with your notes appended to its prompt — iterate
until you approve. Parallel branches are gated one after another once they finish.

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
