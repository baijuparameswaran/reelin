# Reel Factory — Agentic Video Reel Pipeline

Generate short-form video reels (9:16) from a simple text script using a multi-agent AI pipeline.

**Works offline by default** — uses local LLMs via [Ollama](https://ollama.com) with automatic model lifecycle management (auto-upgrade deprecated models, remove obsolete ones). Cloud APIs (OpenAI, Anthropic) available as optional fallback.

> **Verified:** End-to-end pipeline tested with Ollama v0.30.7 on CPU (15.5 GB RAM). Total model disk: ~15.7 GB (4 unique models).

## Quick Start

```bash
# Install (pipenv)
pipenv install       # Creates virtualenv + installs all dependencies
pipenv install --dev # Include dev tools (pytest, ruff)
pipenv shell         # Activate the virtual environment

# Or traditional install
pip install -e .

# One-command local LLM setup (installs Ollama + pulls all models)
./setup_local_llms.sh

# Or manually:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama serve
# reel-factory models pull

# Create a reel (uses local models, no API keys needed)
reel-factory create --script "A cat discovers it can fly" --genre comedy

# Fully automatic (no prompts)
reel-factory create -s "Two robots debate philosophy" -g drama -m auto

# Use cloud models instead
export OPENAI_API_KEY="sk-..."
reel-factory create -s "A superhero origin story" -g action --llm-mode cloud_only
```

## LLM Provider Modes

| Mode | Behavior | API Keys Needed |
|------|----------|-----------------|
| `local_first` (default) | Ollama first, cloud fallback | No (optional) |
| `local_only` | Ollama only, fully offline | No |
| `cloud_first` | Cloud first, Ollama fallback | Yes |
| `cloud_only` | Cloud APIs only | Yes |

Set via `--llm-mode` flag, `REEL_LLM_MODE` env var, or `config/pipeline.yaml`.

## Local Model Assignments

Each pipeline stage uses an optimized local model:

| Stage | Local Model | Size | Cloud Fallback |
|-------|-------------|------|----------------|
| Screenplay Writer | llama3:8b | 4.7 GB | gpt-4o |
| Character Designer | llama3:8b | 4.7 GB | gpt-4o |
| Genre Styler | mistral:7b-instruct | 4.4 GB | gpt-4o |
| Visual Renderer | mistral:7b-instruct | 4.4 GB | runway-gen3 |
| Audio Composer | mistral:7b-instruct | 4.4 GB | suno-v4 |
| Effects Processor | qwen2:7b | 4.4 GB | gpt-4o |
| Video Assembler | phi3:mini | 2.2 GB | gpt-4o |
| Quality Reviewer | llama3:8b | 4.7 GB | gpt-4o |

### Automatic Model Refresh

Models are automatically managed on each pipeline run:
- **Deprecated models** are upgraded to their recommended successor
- **Obsolete models** are removed and replaced
- **Missing models** for required stages are pulled
- State tracked in `~/.reel-factory/model_state.json`

```bash
# Check model health
reel-factory models status

# Manual refresh (upgrade deprecated, remove obsolete)
reel-factory models refresh

# Preview changes without acting
reel-factory models refresh --dry-run

# Keep old models while pulling upgrades
reel-factory models refresh --keep-deprecated

# View full model lifecycle registry
reel-factory models registry
```

## Pipeline Stages

```
Script Input
    |
    v
[1] Screenplay Writer ---- llama3:8b / gpt-4o
    Converts idea -> structured scenes, dialogue, timing
    |
    v
[2] Character Designer --- llama3:8b / gpt-4o
    Creates visual prompts, consistency tags, voice casting
    |
    v
[3] Genre Styler --------- mistral:7b / gpt-4o
    Color palette, typography, pacing, music, filters
    |
    v
[4] Visual Renderer ------ mistral:7b / runway-gen3
    Keyframe prompts, camera plans, interpolation specs
    |
    v
[5] Audio Composer ------- mistral:7b / suno-v4
    Music prompts, voiceover direction, SFX, mix levels
    |
    v
[6] Effects Processor ---- qwen2:7b / gpt-4o
    Color grading, filters, text overlays, transitions
    |
    v
[7] Video Assembler ------ phi3:mini / gpt-4o
    Final 9:16 MP4 assembly with mixed audio
    |
    v
[8] Quality Reviewer ----- llama3:8b / gpt-4o
    Scores quality, suggests improvements
    -> If score < threshold: loop back to stage 4
```

## Architecture

### Agent Design

Each agent:
- Has a **system prompt** defining its role and expertise (see `src/prompts/templates.py`)
- Receives a **formatted user prompt** with pipeline state variables injected
- Outputs **structured JSON** parsed and validated by the agent
- Tracks **token usage** per call via shared `PipelineTokenTracker`
- Supports **feedback injection** — user feedback is included in re-prompts
- Has **confidence scoring** — heuristic assessment of output completeness
- Uses the **LLM Router** which auto-selects local or cloud provider

### LLM Routing

The `LLMRouter` (`src/llm/router.py`) handles provider selection:

1. Checks provider mode (local_first / cloud_first / local_only / cloud_only)
2. For local calls: resolves best available Ollama model for the stage
3. For cloud calls: detects provider from model name prefix (gpt- -> OpenAI, claude- -> Anthropic)
4. Automatic fallback between local and cloud on failure

### Model Lifecycle

The `ModelRefreshManager` (`src/llm/model_registry.py`) maintains a registry of models with status:
- **recommended** — actively suggested for use
- **deprecated** — works but a better successor exists (auto-upgraded on refresh)
- **obsolete** — no longer functional (auto-removed on refresh)

Upgrade paths are tracked (e.g., `llama2:7b -> llama3:8b-instruct-q5_K_M`).

### Stage Boundaries

Every stage has a strict contract in `src/pipeline/stage_defaults.py`:

| Stage | Required Inputs | Produced Outputs |
|-------|----------------|-----------------|
| screenplay | `script` | `screenplay` |
| character_design | `screenplay` | `characters` |
| genre_style | `screenplay` | `style_guide` |
| visual_rendering | `screenplay`, `characters`, `style_guide` | `visual_clips` |
| audio_music | `screenplay`, `style_guide` | `audio_tracks` |
| effects_filters | `visual_clips`, `style_guide` | `processed_clips` |
| assembly | `processed_clips`, `audio_tracks`, `screenplay` | `final_output` |
| review | `final_output`, `screenplay`, `style_guide` | `review_score` |

### Token Usage Tracking

Every LLM call is tracked via `PipelineTokenTracker`:
- Per-call: model, provider, stage, input/output tokens, latency, cost
- Per-stage: aggregated totals
- Pipeline total: cumulative across the run
- Budget limits: configurable max tokens per stage and per run

## User Intervention

At each stage boundary, the user can:

| Action | Effect |
|--------|--------|
| **approve** (default) | Accept output, continue |
| **modify** | Provide feedback, agent re-runs incorporating it |
| **skip** | Use current output regardless of confidence |
| **override** | Swap model for this stage and re-run |
| **abort** | Stop pipeline |

### Interaction Modes

| Mode | Behavior |
|------|----------|
| `prompt` | Pause at every stage with timeout countdown |
| `auto` | Never pause, use defaults throughout |
| `minimal` | Only pause when confidence < threshold |

### Timeout Auto-Continue

Each stage has a configurable timeout. When it expires:
- Default action is applied (approve for all stages except review)
- Review stage always waits for user decision

## Configuration

All settings in `config/pipeline.yaml`:

```yaml
llm_mode: local_first  # local_first | cloud_first | local_only | cloud_only

local_models:
  screenplay: llama3:8b-instruct-q5_K_M
  character_design: llama3:8b-instruct-q5_K_M
  genre_style: mistral:7b-instruct-q5_K_M
  visual_rendering: mistral:7b-instruct-q5_K_M
  audio_music: mistral:7b-instruct-q5_K_M
  effects_filters: qwen2:7b-instruct-q5_K_M
  assembly: phi3:mini
  review: llama3:8b-instruct-q5_K_M

cloud_models:
  screenplay_model: gpt-4o
  character_model: gpt-4o
  review_model: gpt-4o

ollama:
  host: http://localhost:11434
  auto_pull: true
  auto_refresh: true  # Upgrade deprecated models on each run

user_interaction: prompt
max_iterations: 3

token_limits:
  max_per_stage: 50000
  max_per_pipeline_run: 300000
```

## CLI Reference

```bash
# Create a reel
reel-factory create -s "SCRIPT" -g GENRE [-m MODE] [-r RENDERING] [--llm-mode MODE]

# Model management
reel-factory models status          # Show installed vs required
reel-factory models pull [--stage]  # Download models
reel-factory models list            # List all Ollama models
reel-factory models refresh         # Upgrade deprecated, remove obsolete
reel-factory models refresh --dry-run
reel-factory models registry        # Show full model lifecycle table

# Pipeline info
reel-factory stages                 # Show all stages with config
```

## Project Structure

```
reel/
+-- Pipfile                        # Pipenv dependency specification
+-- Pipfile.lock                   # Locked dependency versions
+-- setup_local_llms.sh            # One-command local LLM setup
+-- config/
|   +-- pipeline.yaml              # All pipeline configuration
+-- src/
|   +-- cli.py                     # CLI entry point (Click + Rich)
|   +-- config/
|   |   +-- settings.py            # YAML config loader -> PipelineConfig
|   +-- llm/
|   |   +-- router.py              # LLM router (local-first provider selection)
|   |   +-- ollama_provider.py     # Ollama HTTP client + model resolution
|   |   +-- model_registry.py      # Model lifecycle, deprecation, auto-refresh
|   +-- pipeline/
|   |   +-- orchestrator.py        # Pipeline orchestrator (execution, timeouts, retries)
|   |   +-- state.py               # Typed state flowing between stages
|   |   +-- stage_defaults.py      # Stage boundary contracts + defaults
|   +-- agents/
|   |   +-- base_agent.py          # Abstract base with LLM routing + prompt formatting
|   |   +-- screenplay_agent.py    # [1] Script -> structured screenplay
|   |   +-- character_agent.py     # [2] Screenplay -> character visual refs
|   |   +-- genre_agent.py         # [3] Genre -> comprehensive style guide
|   |   +-- renderer_agent.py      # [4] Scenes -> rendering plan with keyframes
|   |   +-- audio_agent.py         # [5] Style + screenplay -> audio plan
|   |   +-- effects_agent.py       # [6] Clips -> post-processing plan
|   |   +-- assembly_agent.py      # [7] All assets -> assembly plan
|   |   +-- review_agent.py        # [8] Final reel -> quality evaluation
|   +-- prompts/
|   |   +-- templates.py           # Centralized prompt templates for all agents
|   +-- utils/
|       +-- token_tracker.py       # Token usage tracking + cost estimation
+-- tests/
+-- docs/
|   +-- HOWTO.md                   # Detailed usage guide
+-- pyproject.toml                 # Package config + dependencies
+-- .gitignore
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLMs (Local) | Ollama (Llama 3, Mistral, Qwen 2, Phi 3) |
| LLMs (Cloud) | OpenAI GPT-4o, Anthropic Claude (optional) |
| Image Generation | Flux 1.1, DALL-E 3, Stable Diffusion |
| Video Generation | Runway Gen-3, HeyGen, D-ID |
| Voice/TTS | ElevenLabs |
| Music | Suno, Udio |
| Video Processing | FFmpeg, MoviePy, Pillow |
| CLI | Click + Rich |
| LLM Routing | Custom router with Ollama + OpenAI + Anthropic |
| Model Management | Built-in registry with lifecycle tracking |
| Token Tracking | Built-in PipelineTokenTracker |
| Environment | Pipenv (Python 3.14 virtualenv) |

## Extending

### Add a new agent

1. Create `src/agents/my_agent.py`, subclass `BaseReelAgent`
2. Add prompt templates in `src/prompts/templates.py`
3. Implement `execute(state, feedback)` and `stage_name` property
4. Add boundary in `src/pipeline/stage_defaults.py`
5. Add defaults in `STAGE_DEFAULTS`
6. Register in `orchestrator.py` `_build_stages()`
7. Add local model assignment in `STAGE_LOCAL_MODELS`

### Add a new local model to the registry

Edit `src/llm/model_registry.py`:
```python
MODEL_REGISTRY["new-model:7b"] = ModelEntry(
    name="new-model:7b",
    status="recommended",
    added_date="2026-06-01",
    size_gb=4.0,
    capabilities=["json", "creative"],
)
```

### Deprecate a model

```python
MODEL_REGISTRY["old-model:7b"].status = "deprecated"
MODEL_REGISTRY["old-model:7b"].successor = "new-model:7b"
MODEL_REGISTRY["old-model:7b"].reason = "New model has better JSON output"
```

## Environment Variables

```bash
REEL_LLM_MODE=local_first      # Provider mode
OLLAMA_HOST=http://localhost:11434  # Ollama URL
OPENAI_API_KEY=sk-...           # Optional cloud fallback
ANTHROPIC_API_KEY=sk-ant-...    # Optional cloud fallback
```


## Known Limitations

- **crewai** is listed in `pyproject.toml` but excluded from Pipfile — its numpy
  dependency does not yet support Python 3.14. Install manually once upstream adds support.
- CPU-only Ollama inference is slow (~2 min per stage with 8B models). GPU acceleration
  dramatically improves throughput.
- WSL: Ollama server must be restarted each session (`ollama serve &`).
