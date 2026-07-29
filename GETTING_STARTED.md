# Getting started

Everything needed to go from a fresh clone to a finished run, in order. Each
step ends with a way to check it actually worked, so a problem surfaces where
it happened rather than three stages later.

The [README](README.md) is the feature reference — what every stage, flag, and
config block does. This file is just the path in.

**Shortest possible version**, if your machine already has Python 3.10+ and
[Ollama](https://ollama.com):

```bash
git clone <this-repo-url> reel && cd reel
make setup            # venv + deps + the local models for YOUR hardware
make demo             # bundled sample story, one scene, review gate on
```

---

## 1. Prerequisites

| | Needed for | Notes |
|---|---|---|
| **Python 3.10+** | everything | Plain `venv` + `requirements.txt`. No poetry/uv/lockfile. |
| **[Ollama](https://ollama.com)** | every text stage | Runs all the LLM agents locally. Install below. |
| **`ffmpeg`** | stitching clips into one movie | Optional. Without it, clips still render; only the stitch and overlay steps are skipped, with a warning. |
| **A Gemini API key** | character images + Veo video | Optional. The whole text pipeline (ingest → … → storyboard) runs without one. |

### Installing Ollama

If you have an NVIDIA GPU, **use the official installer, not your distro's
snap/apt package**. The Ubuntu snap build is strictly confined and can't open
`/dev/nvidia*`, so it silently falls back to 100%-CPU inference with no error —
the pipeline still works, roughly 10x slower.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Check it:** after `make setup` pulls a model, run a stage and then `ollama ps`
— the loaded model should show a non-zero **`SIZE VRAM`**. All zero means it's
on CPU.

---

## 2. Install

```bash
git clone <this-repo-url> reel && cd reel
make setup
```

`make setup` does three things:

1. **Creates `.venv/`** and installs `requirements.txt` (PyYAML, pypdf,
   google-genai). It never writes a lockfile or a second manifest —
   `requirements.txt` is the single source of truth.
2. **Matches each profile's model to your hardware.** `config/models.yaml` is
   tuned for one specific machine (8 GB VRAM / 24 GB RAM), so on a smaller box
   the deltas are written to a gitignored `config/models.local.yaml` — e.g.
   `quality_high` steps down off the 19 GB `qwen3:30b`. It only ever picks from
   a profile's own declared fallbacks, and on a host the tracked config already
   fits it writes nothing at all.
3. **Pulls those models** via `ollama pull`. This is the slow part — several GB.

Useful variations:

```bash
SKIP_MODELS=1 make setup       # venv + deps only, no multi-GB download
make hardware-config           # just re-run step 2 (after a hardware change)
python -m reel.hardware_config # preview step 2's decisions, change nothing
make setup-models              # just step 3
```

If `ollama` isn't installed yet, step 3 prints the installer one-liner and
skips — installing it needs sudo, so that stays your call. Run
`make setup-models` once it's in place.

**Check it:**

```bash
source .venv/bin/activate   # see the note below
make models                 # which installed model each profile resolves to
make test                   # the offline suite — no LLM/API calls, ~1s
```

### About the venv

`make` targets call `.venv/bin/python` directly, so **they work without
activating anything**. Activate when you want to run `python -m reel.cli …`
yourself:

```bash
source .venv/bin/activate     # deactivate with: deactivate
```

If `.venv/` is missing or you get `ModuleNotFoundError: No module named 'yaml'`,
it was never created (or got deleted) — run `make setup` again. It's gitignored,
so a fresh clone never has one.

---

## 3. Optional: a Gemini API key

Only character/location images and Veo video clips use Gemini. Everything else
runs on the local open models.

```bash
make secrets CMD=set    # paste the key once; stored chmod-600 at ~/.config/reel/gemini_key
make secrets            # CMD defaults to 'status' — is a key set?
```

(Or `export GEMINIAPIKEY=…`; `GEMINI_API_KEY`/`GOOGLE_API_KEY` also work.)

Without a key the image/video stages no-op with a clear hint and the rest of
the run is unaffected — so it's entirely reasonable to do your first few runs
with no key at all, and add one when you want to see actual footage.

**Real money:** every image and every video clip is a paid API call. Two ways
to stay in control — `--no-render` skips both stages for a single run, and
`python -m reel.cli spend` prints estimated spend from the call log at any
time (it also prints automatically after each run).

---

## 4. Your first run

```bash
make demo
```

That runs the bundled sample story end to end: one scene rendered, every design
stage processing the whole story, the human-in-the-loop gate on. How long it
takes depends entirely on your hardware and profile — the LLM stages dominate,
and `PROFILE=fast` below is much quicker than the mixed default.

```bash
make demo SCENES=all             # render every scene, not just the first
make demo NORENDER=1             # no image/video calls at all — no API spend
make demo PROFILE=fast           # one small model throughout; fastest option
make run SRC=path/to/story.txt   # your own .txt or .pdf
```

### What happens at each stage

After every stage the run **pauses at a review gate** showing a summary plus
that stage's fidelity and genre-alignment scores. Your options:

| Input | Effect |
|---|---|
| **Enter** | approve, continue to the next stage |
| any text | feedback — re-runs this stage with your note appended to its prompt |
| `view` | open the full output in `$EDITOR` (default `vim`); save a valid edit to apply it, or quit without saving to leave it unchanged |
| `stop` | pause the run (Ctrl-C does the same) |

Idle at a gate and it auto-approves after `hitl.timeout_seconds` (default 900).
The countdown only runs while you're at the prompt, not while you're in the
editor. For unattended runs set `hitl.enabled: false` in `config/models.yaml`,
or pass `--gate-timeout N`.

### Resuming

Nothing is lost when you stop — each approved stage is already written to
`output/`.

```bash
make demo RESUME=1
make run SRC=story.txt RESUME=1
python -m reel.cli story.txt --resume
```

`--resume` reloads every finished stage and recomputes from the first
unfinished one. `--max-scenes`, `--profile`, `--target-duration` and
`--no-render` are all inherited from the run being resumed, so `--resume` on
its own never silently reverts to defaults or starts spending API quota
partway through a `--no-render` run.

### What you get

Artifacts land in `output/` as they're approved — `structure.json`,
`characters.json`, `scenes.json`, `casting.json`, `soundscape.json`,
`visuals.json`, `cinematography.json`, `storyboard.json`,
`screenplay.fountain`, character/location portraits under `output/casting/`,
clips under `output/video/`, and the stitched `output/video/movie.mp4`.

---

## 5. Where to go next

- **Iterate on a finished run** without redoing it: `python -m reel.cli revise`
  hand-edits any stage and regenerates only what's genuinely downstream of the
  change. A completed run offers this automatically.
- **Run one stage on its own:** `python -m reel.cli stages` to list them,
  `python -m reel.cli stage scenes` to run one against existing checkpoints.
- **Render from finished artifacts** without re-running any LLM stage:
  `python -m reel.cli render`.
- **Keep models current:** `make update` (pull + smoke test), or
  `make install-cron` for weekly/monthly jobs.

Full detail on all of the above — plus genre steering, the moodboard, fidelity
scoring, Veo prompt construction, and every config block — is in the
[README](README.md). Project internals live in
[ARCHITECTURE.md](ARCHITECTURE.md) (conventions and decisions),
[AGENTS.md](AGENTS.md) (per-agent reference), and [PROGRESS.md](PROGRESS.md)
(current status and history).

---

## Troubleshooting

**`make: .venv/bin/python: No such file or directory`**
The venv doesn't exist. `make setup` (or `SKIP_MODELS=1 make setup` if you
already have the models).

**A stage times out or hangs.**
Generation is streamed, so `runtime.request_timeout_seconds` in
`config/models.yaml` is an *inactivity* window — the gap allowed before the
next token — not a cap on total time. A slow stage takes as long as it takes.
If a stage still trips it, raise the value (or set `0` for no limit) and
`--resume`.

**Everything is extremely slow.**
Check `ollama ps` for non-zero `SIZE VRAM` (see step 1 — the snap package is
the usual cause on Ubuntu). Then try `--profile fast` / `make demo
PROFILE=fast`: one small model for every agent means no model reload between
stages, which on a single-GPU host is a bigger win than it sounds.

**A model won't fit / gets OOM-killed.**
`python -m reel.hardware_config` shows what it thinks fits and why; `make
hardware-config` applies it. To override by hand, edit
`config/models.local.yaml` — it's merged field by field over
`config/models.yaml`, so naming just a `model:` keeps that profile's other
settings. Delete it to return to the tracked defaults.

**`ollama pull` fails on a model.**
`scripts/update-models.sh` logs each pull to `scripts/model-updates.log` and
carries on past a failure rather than aborting. A too-old Ollama is the common
cause (Qwen3 needs a recent version) — the script prints the required minimum
and the upgrade command.

**No images or clips were produced.**
Expected without a Gemini key — check with `make secrets`. Also check you
didn't pass `--no-render` (or `NORENDER=1`), and note `--max-scenes` defaults
to `1`, so only the first scene renders unless you ask for more.

**`movie.mp4` is missing but the clips exist.**
`ffmpeg` isn't on `PATH`. Install it, then `python -m reel.cli stitch` — no
re-render, no API spend.
