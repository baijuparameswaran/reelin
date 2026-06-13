# HOWTO: Using Reel Factory

## Prerequisites

- Python 3.11+
- FFmpeg installed and on PATH
- httpx Python package (`pip install httpx`)

### For Local/Offline Mode (Recommended - No API Keys Needed)

- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Required models pulled (see below)

**Quickest setup** — run the included setup script:

```bash
./setup_local_llms.sh
```

This automatically:
1. Downloads and installs Ollama (if not present)
2. Starts the Ollama server
3. Pulls all 4 required models (~15.7 GB total)
4. Verifies everything works

### For Cloud Mode (Optional)

- API keys for OpenAI and/or Anthropic
- Optional: Runway Gen-3, HeyGen/D-ID, ElevenLabs, Suno API keys

## Installation

```bash
cd reel
pip install -e .

# With cloud provider support
pip install -e ".[openai]"      # OpenAI only
pip install -e ".[anthropic]"   # Anthropic only
pip install -e ".[all]"         # All providers

# Development mode
pip install -e ".[dev]"
```

## Setting Up Local Models (Offline Mode)

Reel Factory defaults to **local-first** mode using Ollama. No API keys required.

### 1. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.com/download
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Pull Required Models

```bash
# Pull all recommended models for every stage (~20GB total)
reel-factory models pull

# Or pull for a specific stage
reel-factory models pull --stage screenplay
reel-factory models pull --stage review
```

### 4. Check Status

```bash
reel-factory models status
```

Output:
```
Ollama is running (4 models available)

Stage Model Assignments
 Stage            Required Model               Status   Resolved To
 screenplay       llama3:8b-instruct-q5_K_M    Ready    llama3:8b-instruct-q5_K_M
 character_design llama3:8b-instruct-q5_K_M    Ready    llama3:8b-instruct-q5_K_M
 genre_style      mistral:7b-instruct-q5_K_M   Ready    mistral:7b-instruct-q5_K_M
 visual_rendering mistral:7b-instruct-q5_K_M   Ready    mistral:7b-instruct-q5_K_M
 audio_music      mistral:7b-instruct-q5_K_M   Ready    mistral:7b-instruct-q5_K_M
 effects_filters  qwen2:7b-instruct-q5_K_M     Missing  -
 assembly         phi3:mini                     Ready    phi3:mini
 review           llama3:8b-instruct-q5_K_M    Ready    llama3:8b-instruct-q5_K_M
```

### Recommended Local Models Per Stage

| Stage | Model | Size | Why |
|-------|-------|------|-----|
| screenplay | llama3:8b | 4.7 GB | Strong creative writing + JSON structure |
| character_design | llama3:8b | 4.7 GB | Detailed visual descriptions |
| genre_style | mistral:7b-instruct | 4.4 GB | Aesthetic reasoning, style coherence |
| visual_rendering | mistral:7b-instruct | 4.4 GB | Technical rendering plans |
| audio_music | mistral:7b-instruct | 4.4 GB | Audio composition planning |
| effects_filters | qwen2:7b | 4.4 GB | Technical FFmpeg parameters |
| assembly | phi3:mini | 2.2 GB | Simple technical specs (lightweight) |
| review | llama3:8b | 4.7 GB | Evaluation and scoring |

**Total unique models: 4 (~15.7 GB disk space)**

> Verified with Ollama v0.30.7 on CPU-only (x86_64, 15.5 GB RAM).
> First inference per model takes ~5s to load into memory, subsequent calls are faster.

### Customizing Local Models

Edit `config/pipeline.yaml`:

```yaml
local_models:
  screenplay: llama3.1:8b          # Use newer Llama
  character_design: nous-hermes:7b  # Alternative model
  genre_style: mistral:7b-instruct
  # ... etc
```


### Model Lifecycle & Auto-Refresh

The pipeline automatically refreshes models at the start of each run:
- Detects deprecated models and pulls their recommended successor
- Removes obsolete models that are no longer functional
- Pulls missing models required for stages
- Tracks state in `~/.reel-factory/model_state.json`

```bash
# Preview what would change
reel-factory models refresh --dry-run

# Execute refresh (pull upgrades, remove obsolete)
reel-factory models refresh

# Pull upgrades but keep old models too
reel-factory models refresh --keep-deprecated

# View the full model registry with lifecycle status
reel-factory models registry
```

The model registry (`src/llm/model_registry.py`) defines upgrade paths:
- `llama2:7b` → `llama3:8b` (obsolete → recommended)
- `mistral:latest` → `mistral:7b-instruct` (deprecated → recommended)
- `qwen:7b` → `qwen2:7b` (obsolete → recommended)

## LLM Provider Modes

Control where LLM calls are routed:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `local_first` (default) | Try Ollama, fall back to cloud | Best of both worlds |
| `local_only` | Only use Ollama, fail if unavailable | Fully offline/air-gapped |
| `cloud_first` | Try cloud APIs, fall back to Ollama | When you want best quality |
| `cloud_only` | Only use cloud APIs | When Ollama isn't installed |

### Set mode via CLI

```bash
reel-factory create -s "..." --llm-mode local_only
reel-factory create -s "..." --llm-mode cloud_first
```

### Set mode via config

```yaml
# config/pipeline.yaml
llm_mode: local_first
```

### Set mode via environment variable

```bash
export REEL_LLM_MODE=local_only
```

## Environment Variables

```bash
# LLM provider mode
export REEL_LLM_MODE="local_first"  # local_first|cloud_first|local_only|cloud_only

# Ollama (local models)
export OLLAMA_HOST="http://localhost:11434"  # Custom Ollama URL

# Cloud APIs (optional, for fallback or cloud mode)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Media generation APIs (optional, for full rendering)
export ELEVENLABS_API_KEY="..."
export RUNWAY_API_KEY="..."
export HEYGEN_API_KEY="..."
export SUNO_API_KEY="..."
```

## Basic Usage

### Create a reel (uses local models by default)

```bash
reel-factory create --script "A detective finds a clue in an unexpected place" --genre drama
```

### Create fully offline

```bash
reel-factory create -s "A cat learns to skateboard" -g comedy --llm-mode local_only
```

### Create with cloud models

```bash
reel-factory create -s "Space explorers find alien life" -g action --llm-mode cloud_only
```

### Auto mode (no interaction)

```bash
reel-factory create -s "A cat learns to skateboard" -g comedy -m auto
```

## Interacting at Stage Checkpoints

When a stage completes in `prompt` mode:

```
+--- Pipeline Checkpoint ---+
| Stage Complete: screenplay |
| Confidence: 80%            |
| Auto-continuing in 45s...  |
+----------------------------+
{
  "title": "The Flying Cat",
  "scenes": [...],
  "total_duration_seconds": 28
}
Options: approve (default) | modify | skip | override | abort
Action [approve]:
```

### Actions

| Action | Result |
|--------|--------|
| `approve` / Enter | Accept output, continue |
| `modify` | Provide feedback, re-run stage |
| `skip` | Skip to next stage |
| `override` | Change model for this stage |
| `abort` | Stop pipeline |

## Token Usage Tracking

After every run, see which provider and model served each stage:

```
Per-Stage Breakdown
 Stage              Model                       Provider  Tokens  Calls  Latency
 screenplay         llama3:8b-instruct-q5_K_M   ollama    3,200   1      2,850ms
 character_design   llama3:8b-instruct-q5_K_M   ollama    2,100   1      1,620ms
 genre_style        mistral:7b-instruct          ollama      800   1        310ms
 review             gpt-4o                       openai    1,500   1        650ms
```

## Viewing Stages

```bash
reel-factory stages
```

Shows all 8 stages with local model, cloud fallback, timeouts, and retry config.

## Troubleshooting

### "Ollama is not running"
Start Ollama: `ollama serve`

### "No local model available for stage"
Pull models: `reel-factory models pull`

### Pipeline falls back to cloud unexpectedly
Check `reel-factory models status` - a model may not be pulled. Use `--llm-mode local_only` to force local and see errors.

### Slow local inference
- Use smaller quantizations (q4_K_M instead of q5_K_M)
- Use phi3:mini for more stages
- Ensure GPU acceleration is enabled in Ollama

### Token budget exceeded
Increase limits in `config/pipeline.yaml` under `token_limits`.

## Architecture

Key files:
- `src/llm/router.py` — LLM router with local-first provider selection (ProviderMode enum)
- `src/llm/ollama_provider.py` — Ollama HTTP client, model resolution, pull/list
- `src/llm/model_registry.py` — Model lifecycle management, deprecation, auto-refresh
- `src/prompts/templates.py` — Centralized prompt templates for all 8 agents
- `src/agents/base_agent.py` — Abstract base with call_llm(), format_prompt(), parse_json_response()
- `src/pipeline/orchestrator.py` — Execution loop, model refresh, timeouts, retries
- `src/pipeline/state.py` — Typed state flowing between agents
- `src/pipeline/stage_defaults.py` — Stage boundary contracts and default configs
- `src/utils/token_tracker.py` — Token tracking with provider attribution and cost estimation
- `config/pipeline.yaml` — All configuration (models, timeouts, thresholds, ollama settings)
- `setup_local_llms.sh` — One-command Ollama + model setup script
