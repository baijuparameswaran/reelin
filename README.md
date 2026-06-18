# reel

A multi-modal, agentic pipeline that turns source material (a book, short story,
or script) into production-ready creative assets — ultimately a movie / reel /
web-series episode — navigating the phases of adaptation, some in parallel.

Built to run on **locally-hosted open LLMs** (via [Ollama](https://ollama.com)),
developed in slow, steady iterations.

## Iteration 1 — "screenplay material"

The first agent set converts raw text into screenplay material:

```
ingest ─▶ structure ─┐
                     ├─▶ scenes ─▶ screenplay ─▶ assemble
         characters ─┘
```

| Agent | Output |
|-------|--------|
| **ingest** | normalized text + metadata (`title`, word count) |
| **structure** | logline, genre, themes, tone, three-act beat sheet |
| **characters** | cast with roles, wants, arcs, traits |
| **scenes** | numbered scene list (sluglines, summaries, purpose) |
| **screenplay** | Fountain-formatted draft pages |

Artifacts land in `output/`: `structure.json`, `characters.json`, `scenes.json`,
`screenplay.fountain`, and a combined `project.json`.

## Quick start

```bash
make setup          # venv + deps
make models         # show which local models each profile resolves to
make demo           # run on the bundled sample story (fast profile)
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

CPU-only, ~7.6 GB RAM in WSL (16 GB laptop). Implications:

- Inference is slow (a few tok/s); run as a batch, not interactively.
- `num_ctx` is capped at 4096 to fit a 7B model in RAM. Larger inputs are
  truncated for now (chunking is a later iteration).
- The `quality` tier (8B) is tight at current RAM. **Recommended:** raise the WSL
  memory limit to ~12 GB via `%UserProfile%\.wslconfig` then `wsl --shutdown`.
  See `CLAUDE.md` for the exact steps.

See `CLAUDE.md` for project vision, decisions, and session-to-session state.
