"""Model-agnostic Ollama client shared by every agent.

Design goals:
  * Zero hard-coded model names in agent code — agents pick a *profile*.
  * Always-runnable: if a profile's preferred model isn't pulled, fall back to
    the first installed model in its `fallbacks` list.
  * Zero non-stdlib deps for the transport (urllib), so there's little to break.

Timeouts: generation streams token-by-token, so the configured timeout is an
*inactivity* window (max gap between tokens), not a cap on total generation
time. A slow-but-progressing model on a CPU host never trips it; only a hung or
crashed daemon does. Tune via `runtime.request_timeout_seconds` in models.yaml.
"""
from __future__ import annotations

import base64
import json
import re
import socket
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
# Per-host overlay, gitignored, written by `reel.hardware_config` (make setup).
# Deep-merged over CONFIG_PATH by `config()` — see `local_overrides`.
LOCAL_CONFIG_PATH = CONFIG_PATH.with_name("models.local.yaml")

# Legacy flat cap. Retained as the safe floor returned for an unknown or
# malformed profile, and as the value an undeclared `num_ctx` derives back to
# — but agents should call `max_chars(profile)`, which derives the real budget
# from the model actually serving the stage.
MAX_CHARS = 12_000

# ── local source-budget derivation (from each profile's own num_ctx) ─────────
# `num_ctx` is this project's hardware knob: it's chosen per profile against
# what fits in 8 GB VRAM plus the CPU-RAM KV cache (see the profile comments
# in config/models.yaml), so deriving the source budget from it IS the
# hardware derivation. A profile that can afford more context should be
# allowed to read more of the story; a flat constant across all of them threw
# that away — `synthesis` runs at 16K ctx and was capped as if it were 8K.
#
# num_ctx is a TOTAL token budget (prompt + response), so two reserves come
# off the top before the remainder is spent on source text:
_PROMPT_RESERVE_TOKENS = 3_000   # the agent's own rules/scaffold. Sized on the
                                 # largest of the three whole-source prompts —
                                 # `scenes` measures ~2,900 tokens — so the
                                 # budget is safe for all of them, not just the
                                 # average one.
_OUTPUT_RESERVE_TOKENS = 2_000   # room for the JSON response. Matches what was
                                 # measured left over at the old flat cap, so
                                 # output space doesn't shrink as input grows.
_CHARS_PER_TOKEN = 4             # English prose approximation, deliberately on
                                 # the generous side of the usual 3.5-4.
_DEFAULT_NUM_CTX = 8_192         # for a profile that declares no num_ctx —
                                 # every profile in this config declares one,
                                 # and 8192 derives back to ~the old flat cap,
                                 # so an undeclared profile behaves as before
                                 # rather than silently collapsing to a floor.
_MIN_SOURCE_CHARS = 4_000        # floor for a genuinely tiny num_ctx, where the
                                 # reserves would otherwise leave nothing at all.

# Source-text budget per provider, for providers whose context dwarfs anything
# local. Not the model's full context: the binding limit for a stage like
# `scenes` is the OUTPUT cap (65K tokens on the Gemini frontier models, a few
# hundred scenes' worth) and cost, not input. 600K chars ~ 150K tokens covers
# a full novel while staying well inside a 1M-token window.

_PROVIDER_MAX_CHARS: dict[str, int] = {"gemini": 600_000}

# ── hardware detection ────────────────────────────────────────────────────────

# Approximate VRAM/RAM footprint in MB for common models (quantized weights).
# Used when a model isn't yet pulled and its size can't be queried from Ollama.
# Sizes assume the default q4_K_M (or equivalent) quantization tier.
_MODEL_SIZE_MB: dict[str, int] = {
    "qwen3:0.6b":          520,
    "qwen3:1.7b":        1_100,
    "qwen3:4b":          2_560,
    "qwen3:8b":          5_200,
    "qwen3:14b":         9_000,
    "qwen3:30b":        19_000,
    "qwen3:32b":        20_000,
    "qwen2.5:latest":    4_700,
    "qwen2.5:7b":        4_700,
    "qwen2:7b":          4_100,
    "gemma3:4b":         3_300,
    "gemma3:12b":        8_100,
    "gemma3:27b":       17_000,
    "llama3:8b":         4_700,
    "llama3.1:8b":       4_700,
    "llama3.2:3b":       2_000,
    "mistral:latest":    4_100,
    "mistral:7b-instruct": 4_100,
    "phi3:mini":         2_300,
    "phi3:latest":       2_300,
    "phi4:latest":       9_100,
    "deepseek-r1:7b":    4_700,
    "deepseek-r1:14b":   9_000,
    "deepseek-r1:32b":  19_500,
    "qwq:32b":          19_500,
}


@lru_cache(maxsize=1)
def gpu_vram_mb() -> int | None:
    """Total GPU VRAM in MB (sum across all GPUs), or None when nvidia-smi is absent."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
        return sum(vals) if vals else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def system_ram_mb() -> int:
    """Total system RAM in MB (from /proc/meminfo MemTotal)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 8_192  # conservative fallback


def hardware_summary() -> str:
    """One-line hardware description for log messages."""
    vram = gpu_vram_mb()
    ram = system_ram_mb()
    gpu_str = f"GPU {vram} MB VRAM" if vram else "no GPU detected"
    return f"{gpu_str}, RAM {ram} MB"


@lru_cache(maxsize=1)
def _installed_model_sizes() -> dict[str, int]:
    """Installed Ollama model tag → file size in MB."""
    try:
        tags = _api("/api/tags", method="GET")
        return {m["name"]: m["size"] // (1024 * 1024) for m in tags.get("models", [])}
    except Exception:
        return {}


def _model_size_mb(tag: str) -> int | None:
    """Approximate VRAM/RAM footprint of a model in MB.

    Tries installed Ollama models first (accurate), then our built-in table
    (for models not yet pulled), then a family-name prefix match.
    """
    installed = _installed_model_sizes()
    if tag in installed:
        return installed[tag]
    for name, size in installed.items():
        if name.startswith(tag):
            return size
    if tag in _MODEL_SIZE_MB:
        return _MODEL_SIZE_MB[tag]
    base = tag.split(":")[0]
    for known, size in _MODEL_SIZE_MB.items():
        if known.startswith(base + ":"):
            return size
    return None  # unknown — caller should treat as "assume fits"


def can_run_model(
    tag: str,
    *,
    vram_mb: int | None = None,
    ram_mb: int | None = None,
) -> bool:
    """True if the model's estimated size fits within GPU VRAM + 80% of system RAM.

    Models that exceed this are not impossible to run (Ollama CPU-offloads the
    overflow), but they'll be slow or OOM.  Unknown sizes pass through so we
    never silently drop a model we have no data on.
    """
    size = _model_size_mb(tag)
    if size is None:
        return True
    if vram_mb is None:
        vram_mb = gpu_vram_mb() or 0
    if ram_mb is None:
        ram_mb = system_ram_mb()
    usable = vram_mb + int(ram_mb * 0.80)
    return size <= usable


@dataclass
class Profile:
    name: str
    model: str
    fallbacks: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    think: bool | None = None  # per-profile override; None = use runtime.think global
    # "ollama" (default, every existing profile) or "gemini" (hosted frontier
    # tier — see reel.gemini.generate_text). Keeping the provider on the
    # PROFILE rather than on the agent is what preserves this project's
    # "agents pick a profile, never a model name" rule: opting a stage into a
    # hosted model is a one-line `agent_profiles` change, and no agent module
    # learns a provider or model name.
    provider: str = "ollama"
    # Local profile to degrade to when a non-Ollama provider isn't usable (no
    # SDK, no credentials). `fallbacks` can't serve this purpose — those are
    # Ollama model TAGS resolved by `resolve_model`, not profile names.
    fallback_profile: str = "quality_high"


def _deep_merge(base: dict, overlay: dict) -> dict:
    """`overlay` layered over `base`, recursing into nested mappings.

    Field-level, not block-level, so an overlay naming just
    `profiles.quality_high.model` keeps that profile's tracked `fallbacks`,
    `options`, and comments-in-spirit rather than replacing the whole entry.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def local_overrides() -> dict:
    """The gitignored per-HOST overlay (`config/models.local.yaml`), or `{}`.

    `config/models.yaml` is tracked, hand-tuned, and heavily commented against
    ONE machine's hardware (see ARCHITECTURE.md's "Hardware reality"), so it's
    the wrong place for a different host's model choices — rewriting it would
    destroy the commentary and leave every clone with a dirty worktree.
    `reel.hardware_config` writes this file instead.

    Never raises: a malformed or unreadable overlay yields `{}`, so the tracked
    config alone still runs the pipeline. Matches how every other config read in
    this module degrades.
    """
    try:
        if not LOCAL_CONFIG_PATH.exists():
            return {}
        data = yaml.safe_load(LOCAL_CONFIG_PATH.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def base_config() -> dict:
    """The TRACKED config alone, with no per-host overlay applied.

    Everything that runs the pipeline wants `config()` instead. This exists for
    `reel.hardware_config`, which decides whether a host needs an override at
    all: judged against the merged config, a previously-written override would
    become the new baseline and always look like it already fits — so a host
    that grew (or a downgrade written from a bad reading) could never be
    restored to the tracked model.
    """
    return yaml.safe_load(CONFIG_PATH.read_text())


@lru_cache(maxsize=1)
def config() -> dict:
    # Cached for the process, so the per-host overlay is read exactly once,
    # alongside the tracked file. Anything that changes either path at runtime
    # (tests, `hardware_config.apply`) must call `config.cache_clear()`.
    cfg = base_config()
    overrides = local_overrides()
    return _deep_merge(cfg, overrides) if overrides else cfg


def host() -> str:
    return config().get("ollama_host", "http://localhost:11434")


# ── config schema validation ─────────────────────────────────────────────────
# A lightweight, dependency-free shape check for config/models.yaml — NOT a
# full JSON-schema (this file's own philosophy is "kept intentionally
# minimal," PyYAML is already the sole non-stdlib runtime dep, so a schema
# library is disproportionate for one config file). Every block in
# models.yaml is read elsewhere in this codebase via defensive `.get(key,
# default)` calls, which is robust to a MISSING key but silent on a
# MISSPELLED one — an operator typo (`mutli_character_references`,
# `videos:` instead of `video:`) currently just silently falls back to the
# default everywhere it's read, with no signal anything was wrong. This
# only catches that class of mistake: unknown top-level/nested keys (one
# level deep — deep enough for the flat-ish blocks this file actually has)
# and a handful of the most consequential fields' basic types. It never
# blocks anything — `validate_config` returns warnings, callers print and
# continue, matching this codebase's established best-effort philosophy
# everywhere else a config value is read.
_KNOWN_TOP_KEYS = {
    "ollama_host", "hitl", "image", "video", "fidelity", "genre", "moodboard",
    "revision", "critique", "duration", "runtime", "min_ollama_version",
    "profiles", "agent_profiles",
}

# Known sub-keys for each dict-valued top-level block, one level deep —
# matches exactly the keys config/models.yaml itself documents and the keys
# actually `.get()`-read elsewhere in this codebase (see ARCHITECTURE.md's
# `image`/`video`/`fidelity`/`genre`/`moodboard`/`revision`/`duration`/
# `runtime` blocks). `video.audio`/`video.overlays` are nested one level
# further — checked separately below rather than folded into this flat map,
# since this validator is deliberately one-level-deep, not fully recursive.
_KNOWN_SUB_KEYS: dict[str, set[str]] = {
    "hitl": {"enabled", "timeout_seconds"},
    "image": {"enabled", "backend", "open_backend", "model", "timeout_seconds",
             "host", "steps", "size", "width", "height", "guidance_scale",
             "negative_prompt", "style_suffix", "img2img_strength"},
    "video": {"enabled", "backend", "open_backend", "model", "aspect_ratio",
             "resolution", "continuity", "continuity_mode",
             "multi_character_references", "multi_segment_prompting",
             "style_suffix", "poll_seconds",
             "timeout_seconds", "audio", "pipeline_class", "host", "endpoint",
             "seconds", "fps", "size", "overlays"},
    "fidelity": {"per_stage", "min_score"},
    "genre": {"value", "steer", "enforce", "min_score"},
    "moodboard": {"enabled", "steer"},
    "revision": {"identity_drift_threshold"},
    "critique": {"enabled", "iterations", "stages"},
    "duration": {"target_seconds"},
    "runtime": {"max_parallel_agents", "request_timeout_seconds", "think",
               "num_gpu", "escalate_after", "escalate_score_gap"},
}
_KNOWN_PROFILE_KEYS = {"model", "fallbacks", "options", "think",
                       "provider", "fallback_profile"}
_KNOWN_PROVIDERS = {"ollama", "gemini"}

_KNOWN_VIDEO_AUDIO_KEYS = {"no_background_music", "room_tone", "no_subtitles"}
_KNOWN_VIDEO_OVERLAY_KEYS = {"enabled", "subtitles", "shot_info", "font_size",
                            "subtitle_color", "label_color"}

# A handful of the most consequential fields' expected Python type(s) — not
# exhaustive (this validator's whole point is catching typos/shape drift
# cheaply, not replacing a real schema), just the fields most likely to
# silently misbehave if given the wrong type via a YAML quoting mistake
# (e.g. `enabled: "true"` — a non-empty STRING, which is truthy in Python,
# so a `.get("enabled", True)` check wouldn't even flag it as falsy).
_KNOWN_TYPES: dict[tuple[str, str], type | tuple[type, ...]] = {
    ("hitl", "enabled"): bool,
    ("hitl", "timeout_seconds"): (int, float),
    ("image", "enabled"): bool,
    ("video", "enabled"): bool,
    ("video", "continuity"): bool,
    ("video", "multi_character_references"): bool,
    ("video", "multi_segment_prompting"): bool,
    ("fidelity", "per_stage"): bool,
    ("fidelity", "min_score"): (int, float),
    ("genre", "steer"): bool,
    ("genre", "enforce"): bool,
    ("genre", "min_score"): (int, float),
    ("moodboard", "enabled"): bool,
    ("moodboard", "steer"): bool,
    ("critique", "enabled"): bool,
    ("critique", "iterations"): int,
    ("duration", "target_seconds"): (int, float),
    ("runtime", "max_parallel_agents"): int,
    ("runtime", "think"): bool,
}


def validate_config(cfg: dict | None = None) -> list[str]:
    """Return a list of human-readable warnings for likely config mistakes
    in `cfg` (defaults to the real `config()`) — unknown top-level or
    nested keys, and a wrong type on the handful of fields in
    `_KNOWN_TYPES`. Never raises; a malformed `cfg` (not even a dict) just
    yields one warning describing that, rather than crashing whatever
    caller invoked this as a startup sanity check."""
    cfg = config() if cfg is None else cfg
    warnings: list[str] = []
    if not isinstance(cfg, dict):
        return [f"config root is not a mapping (got {type(cfg).__name__}) — models.yaml may be malformed"]

    for key in cfg:
        if key not in _KNOWN_TOP_KEYS:
            warnings.append(f"unknown top-level config key '{key}' — check for a typo")

    for block, known in _KNOWN_SUB_KEYS.items():
        value = cfg.get(block)
        if value is None:
            continue
        if not isinstance(value, dict):
            warnings.append(f"config '{block}' should be a mapping (got {type(value).__name__})")
            continue
        for sub in value:
            if sub not in known:
                warnings.append(f"unknown config key '{block}.{sub}' — check for a typo")

    video = cfg.get("video")
    if isinstance(video, dict):
        for nested_block, known in (("audio", _KNOWN_VIDEO_AUDIO_KEYS),
                                    ("overlays", _KNOWN_VIDEO_OVERLAY_KEYS)):
            nested = video.get(nested_block)
            if isinstance(nested, dict):
                for sub in nested:
                    if sub not in known:
                        warnings.append(f"unknown config key 'video.{nested_block}.{sub}' — check for a typo")

    for (block, sub), expected in _KNOWN_TYPES.items():
        value = cfg.get(block, {})
        if not isinstance(value, dict) or sub not in value:
            continue
        if not isinstance(value[sub], expected):
            expected_name = (expected.__name__ if isinstance(expected, type)
                            else "/".join(t.__name__ for t in expected))
            warnings.append(f"config '{block}.{sub}' should be {expected_name} "
                           f"(got {type(value[sub]).__name__}: {value[sub]!r})")

    # `critique.stages` — a mapping of STAGE NAME to a per-stage override.
    # Validated separately (and one level deeper than the generic loop above)
    # because a typo'd stage name here is completely silent: the override
    # simply never matches, and the stage keeps the global setting while the
    # operator believes they changed it. Checked against the real stage
    # registry, imported lazily so this module stays free of the
    # reel.stages -> agents -> llm import cycle.
    crit = cfg.get("critique")
    if isinstance(crit, dict) and crit.get("stages") is not None:
        per_stage = crit["stages"]
        if not isinstance(per_stage, dict):
            warnings.append("config 'critique.stages' should be a mapping "
                           f"(got {type(per_stage).__name__})")
        else:
            try:
                from . import stages as _stages
                known_stages = set(_stages.names())
            except Exception:
                known_stages = set()
            for stage_name, override in per_stage.items():
                if known_stages and stage_name not in known_stages:
                    warnings.append(f"config 'critique.stages.{stage_name}' is not a "
                                   f"known stage — known: {sorted(known_stages)}")
                if not isinstance(override, dict):
                    warnings.append(f"config 'critique.stages.{stage_name}' should be a "
                                   f"mapping (got {type(override).__name__})")
                    continue
                for k, v in override.items():
                    if k not in ("enabled", "iterations"):
                        warnings.append(f"unknown config key "
                                       f"'critique.stages.{stage_name}.{k}' — "
                                       "check for a typo")
                    elif k == "enabled" and not isinstance(v, bool):
                        warnings.append(f"config 'critique.stages.{stage_name}.enabled' "
                                       f"should be bool (got {type(v).__name__}: {v!r})")
                    elif k == "iterations" and not isinstance(v, int):
                        warnings.append(f"config 'critique.stages.{stage_name}.iterations' "
                                       f"should be int (got {type(v).__name__}: {v!r})")

    # Per-profile keys. Unlike the blocks above, `profiles` is a mapping of
    # user-chosen NAMES to profile bodies, so the one-level-deep loop can't be
    # reused — the check has to descend one further, into each body. Worth
    # having specifically because of `provider`: a misspelled `provider:` key
    # silently routes a stage back to Ollama, which looks like the hosted model
    # simply performing badly rather than never having been called.
    profiles = cfg.get("profiles")
    if isinstance(profiles, dict):
        for pname, body in profiles.items():
            if not isinstance(body, dict):
                warnings.append(f"config 'profiles.{pname}' should be a mapping "
                               f"(got {type(body).__name__})")
                continue
            for sub in body:
                if sub not in _KNOWN_PROFILE_KEYS:
                    warnings.append(f"unknown config key 'profiles.{pname}.{sub}' "
                                   "— check for a typo")
            provider = body.get("provider", "ollama")
            if provider not in _KNOWN_PROVIDERS:
                warnings.append(f"config 'profiles.{pname}.provider' is "
                               f"'{provider}' — known providers: "
                               f"{sorted(_KNOWN_PROVIDERS)}")
            fb = body.get("fallback_profile")
            if fb is not None and fb not in profiles:
                warnings.append(f"config 'profiles.{pname}.fallback_profile' "
                               f"references undefined profile '{fb}'")

    agent_profiles = cfg.get("agent_profiles")
    if isinstance(agent_profiles, dict) and isinstance(profiles, dict):
        for stage, tier in agent_profiles.items():
            if tier not in profiles:
                warnings.append(f"agent_profiles.{stage} references undefined profile '{tier}' "
                               f"— known profiles: {sorted(profiles)}")

    return warnings


# ── creative direction (e.g. genre steering) ─────────────────────────────────
# A process-wide directive prepended to the system message of *steered* generations
# so every creative stage leans the same way (set by the pipeline from the genre
# agent). Grader/checker calls (fidelity, genre enforcement) go through
# `models.text`, which disables steering, so they stay neutral.
_DIRECTION: str | None = None


def set_direction(text: str | None) -> None:
    """Set (or clear with None) the global creative direction injected into steered
    generations."""
    global _DIRECTION
    _DIRECTION = (text or "").strip() or None


def direction() -> str | None:
    return _DIRECTION


def request_timeout() -> float | None:
    """Inactivity timeout (seconds) for a single socket read during generation.

    Because generation is streamed, this is the maximum gap *between tokens*, not
    a ceiling on total generation time. `0` (or null) means wait indefinitely.
    """
    val = config().get("runtime", {}).get("request_timeout_seconds", 300)
    return val if val else None


def think_enabled() -> bool:
    """Whether to let thinking models emit their reasoning trace.
    Only applied when the resolved model actually supports thinking (see
    `_is_thinking_model`). Override with config `runtime.think: true/false`."""
    return bool(config().get("runtime", {}).get("think", False))


def unload_model(profile: str) -> None:
    """Force-unload a model from Ollama memory before a feedback retry.

    Sets keep_alive=0 on a no-op generate call, which tells Ollama to evict the
    model immediately.  This clears the KV cache so the next call loads a
    completely fresh context with no residue from the previous attempt.
    Best-effort: silently ignored if Ollama is unreachable or the call fails.
    """
    try:
        p = get_profile(profile)
        model = resolve_model(p)
        _api("/api/generate", {"model": model, "prompt": "", "keep_alive": 0})
    except Exception:
        pass


def _api(
    path: str,
    payload: dict | None = None,
    method: str = "POST",
    timeout: float | None = 30,
) -> dict:
    """Non-streaming JSON call (used for quick endpoints like /api/tags)."""
    url = host() + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise RuntimeError(
            f"Ollama returned HTTP {e.code} for {path}: {body}\n"
            "(A 500 here is often out-of-memory loading a model on low-RAM hosts.)"
        ) from e
    except (TimeoutError, socket.timeout) as e:
        raise RuntimeError(
            f"Ollama timed out on {path} after {timeout}s ({e})."
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host()} ({e}). "
            "Is the daemon running?  Try:  ollama serve"
        ) from e


@lru_cache(maxsize=1)
def installed_models() -> tuple[str, ...]:
    try:
        tags = _api("/api/tags", method="GET")
        return tuple(m["name"] for m in tags.get("models", []))
    except Exception:
        return tuple()


def get_profile(name: str, cfg: dict | None = None) -> Profile:
    """The named profile, read from `cfg` (defaults to the merged `config()`).

    `cfg` exists so `hardware_config` can read profiles out of `base_config()`
    — the pre-overlay view it has to judge against.
    """
    profiles = (config() if cfg is None else cfg)["profiles"]
    if name not in profiles:
        raise KeyError(f"Unknown profile {name!r}; available: {list(profiles)}")
    p = profiles[name]
    think = p.get("think")  # explicit bool in yaml → per-profile override
    return Profile(name, p["model"], p.get("fallbacks", []), p.get("options", {}),
                   think=think,
                   provider=p.get("provider", "ollama"),
                   fallback_profile=p.get("fallback_profile", "quality_high"))


def agent_profile(agent: str) -> str:
    """The default profile name configured for a given agent."""
    return config().get("agent_profiles", {}).get(agent, "fast")


def local_max_chars(p: Profile) -> int:
    """Source-text budget for one LOCAL profile, derived from its `num_ctx`.

    `num_ctx` is the hardware knob in this project — each profile's value is
    chosen against what fits in 8 GB VRAM plus the CPU-RAM KV cache — so
    deriving from it makes the source cap hardware-derived per model rather
    than one flat number for every tier. It's a TOTAL (prompt + response)
    budget, so the agent's own rules and room for the JSON reply come off the
    top first; whatever remains is spent on story text.

    A profile declaring no `num_ctx` derives back to roughly the old flat cap
    rather than collapsing to the floor — every profile in this config
    declares one, so that path is for a hand-added profile, where behaving as
    before is less surprising than silently truncating. Never raises: a
    non-numeric `num_ctx` falls back to the same default."""
    try:
        num_ctx = int((p.options or {}).get("num_ctx", _DEFAULT_NUM_CTX))
    except (TypeError, ValueError):
        num_ctx = _DEFAULT_NUM_CTX
    usable = num_ctx - _PROMPT_RESERVE_TOKENS - _OUTPUT_RESERVE_TOKENS
    return max(_MIN_SOURCE_CHARS, usable * _CHARS_PER_TOKEN)


def max_chars(profile: str | None = None) -> int:
    """How much SOURCE TEXT a stage may send, given the model actually
    serving it. Agents should call this rather than reading `MAX_CHARS`
    directly — that constant is only the local default.

    Why per-profile: `MAX_CHARS` (12,000 chars ≈ 3,000 tokens) is sized for a
    local profile at `num_ctx: 8192`, where the scenes prompt's own rules
    already consume ~2,900 tokens. On a 1M-context hosted model that same cap
    silently truncates the story to roughly the first 2,000 words — the model
    has room for a whole novel and never sees past chapter one. Switching a
    stage's profile alone doesn't fix that, because the truncation happens in
    the agent, before the prompt is ever built.

    Resolution order:
      1. `options.max_source_chars` on the profile — explicit always wins.
      2. Local (ollama) profiles → derived from that profile's own
         `num_ctx` (`local_max_chars`), so a tier configured for more context
         is actually allowed to read more of the story.
      3. A hosted profile whose provider ISN'T usable → the budget of its
         `fallback_profile`. This is the case that matters most: `llm.generate`
         degrades such a profile to a local model, so handing back a hosted
         budget would push 600K chars at a model with an 8K context. The
         truncation decision and the routing decision must agree.
      4. Otherwise the provider's budget, defaulting to `MAX_CHARS`.

    Never raises — an unknown or malformed profile yields `MAX_CHARS`, the
    safe floor, matching how every other config read in this module degrades.
    """
    if not profile:
        return MAX_CHARS
    try:
        p = get_profile(profile)
    except Exception:
        return MAX_CHARS

    explicit = (p.options or {}).get("max_source_chars")
    if explicit:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass

    if p.provider == "ollama":
        return local_max_chars(p)

    if not _provider_usable(p.provider):
        # Guard against a fallback_profile that points at itself or loops.
        if p.fallback_profile and p.fallback_profile != p.name:
            return max_chars(p.fallback_profile)
        return MAX_CHARS

    return _PROVIDER_MAX_CHARS.get(p.provider, MAX_CHARS)


def _provider_usable(provider: str) -> bool:
    """Whether a non-Ollama provider can actually serve a call right now —
    the same question `_generate_hosted` asks before routing, kept in sync so
    the source-truncation budget can't disagree with where the call lands."""
    if provider == "gemini":
        try:
            from . import gemini
            return gemini.available()
        except Exception:
            return False
    return False


def resolve_model(profile: Profile) -> str:
    """Preferred installed model that fits in available GPU VRAM + RAM.

    Selection order:
      1. Profile's preferred model — if installed AND fits.
      2. First fallback that is installed AND fits.
      3. Preferred model installed but too large (slow CPU offload, warns user).
      4. First installed fallback regardless of size.
      5. Any installed model (last resort).
    """
    have = installed_models()
    vram_mb = gpu_vram_mb() or 0
    ram_mb = system_ram_mb()

    def present(tag: str) -> bool:
        return any(m == tag or m.startswith(tag) for m in have)

    def fits(tag: str) -> bool:
        return can_run_model(tag, vram_mb=vram_mb, ram_mb=ram_mb)

    # First pass: installed + fits in hardware.
    if present(profile.model) and fits(profile.model):
        return profile.model
    for fb in profile.fallbacks:
        if present(fb) and fits(fb):
            return fb

    # Second pass: installed but oversized — Ollama will CPU-offload the overflow.
    if present(profile.model):
        return profile.model
    for fb in profile.fallbacks:
        if present(fb):
            return fb
    if have:
        return have[0]
    raise RuntimeError(
        "No Ollama models are installed. Run scripts/update-models.sh first."
    )


_THINKING_MODELS = ("qwen3", "deepseek-r1", "qwq")

def _is_thinking_model(model: str) -> bool:
    return any(m in model.lower() for m in _THINKING_MODELS)


_VISION_MODELS = ("gemma3", "llava", "bakllava", "qwen2-vl", "minicpm-v", "moondream")

def _is_vision_model(model: str) -> bool:
    return any(m in model.lower() for m in _VISION_MODELS)


# Ordered quality tiers — lower index = lighter/faster, higher = heavier/better.
# "thinking" sits between quality and quality_high: same qwen3:8b but with reasoning
# traces enabled, giving much better synthesis for multi-artifact stages.
_PROFILE_TIERS: tuple[str, ...] = ("fast", "quality", "synthesis", "quality_high")


def next_profile(name: str) -> str | None:
    """Return the next higher quality profile, or None if already at the top."""
    try:
        idx = _PROFILE_TIERS.index(name)
        nxt = idx + 1
        return _PROFILE_TIERS[nxt] if nxt < len(_PROFILE_TIERS) else None
    except ValueError:
        return None


_warned_providers: set[str] = set()


def _generate_hosted(p: Profile, prompt: str, *, sys_msg: str | None,
                     as_json: bool, schema: dict | None) -> str | None:
    """Route one generation to a non-Ollama provider, or return None to tell
    the caller to fall through to the local path.

    Returning None rather than raising is the whole point: a missing SDK or
    missing credentials must degrade to `p.fallback_profile` and keep the run
    going, exactly the way the Gemini image/video path no-ops into the open
    backend without `GEMINIAPIKEY`. An UNKNOWN provider name degrades the same
    way — a typo in `provider:` should cost a warning, not the run.

    A failure from a provider that IS configured and reachable does propagate
    — that's a real error (bad request, refusal, truncation), not a
    fall-back-to-local situation, and silently re-running a scene breakdown on
    a small local model after paying for a hosted call would hide it."""
    if p.provider != "gemini":
        if p.provider not in _warned_providers:
            _warned_providers.add(p.provider)
            print(f"[reel] profile '{p.name}' declares unknown provider "
                  f"'{p.provider}' — falling back to '{p.fallback_profile}'")
        return None

    from . import gemini
    if not gemini.available():
        if p.provider not in _warned_providers:
            _warned_providers.add(p.provider)
            print(f"[reel] profile '{p.name}' wants Gemini but no API key is set "
                  f"({gemini.key_hint()}) — falling back to '{p.fallback_profile}'")
        return None

    opts = p.options or {}
    return gemini.generate_text(
        prompt,
        system=sys_msg,
        model=p.model,
        as_json=as_json,
        schema=schema,
        max_output_tokens=opts.get("max_output_tokens"),
        temperature=opts.get("temperature"),
    )


def generate(
    prompt: str,
    *,
    profile: str = "fast",
    system: str | None = None,
    as_json: bool = False,
    steer: bool = True,
    images: list | None = None,
    schema: dict | None = None,
) -> str:
    """Single-turn generation against a local model selected by `profile`.

    Streams the response so the socket timeout acts as an inactivity window
    (max gap between tokens) rather than a cap on total generation time — slow
    CPU inference can take as long as it needs, as long as tokens keep arriving.

    When a global creative `direction()` is set (e.g. genre) and `steer` is True,
    it is prepended to the system message so the stage leans that way. Pass
    `steer=False` for neutral calls (graders/checkers) — `models.text` does.

    `images` is an optional list of file paths or raw bytes to attach as vision
    inputs. Only sent when the resolved model supports vision; ignored silently
    for text-only models (avoids Ollama 400 errors).

    A profile declaring a non-Ollama `provider` (see `Profile`) routes to that
    provider instead, degrading to its `fallback_profile` when the provider
    isn't usable. `schema` (a JSON Schema) is honored only on providers that
    support structured outputs — it's ignored on Ollama, which has `as_json`'s
    coarser "must be valid JSON" mode and no schema enforcement. No agent
    passes one yet; the parameter exists so a stage can adopt it without
    another change here.
    """
    p = get_profile(profile)
    steer_text = direction() if steer else None
    sys_msg = "\n\n".join(s for s in (steer_text, system) if s) or None

    # Steering is composed BEFORE the provider branch on purpose: a hosted
    # creative stage must receive the same genre+moodboard direction a local
    # one does, or it would silently be the one unsteered stage in the run.
    if p.provider != "ollama":
        hosted = _generate_hosted(p, prompt, sys_msg=sys_msg,
                                  as_json=as_json, schema=schema)
        if hosted is not None:
            return hosted
        p = get_profile(p.fallback_profile)  # degraded — continue locally

    model = resolve_model(p)
    messages: list[dict] = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})

    user_msg: dict = {"role": "user", "content": prompt}
    # Attach images only when the resolved model is vision-capable.
    if images and _is_vision_model(model):
        encoded: list[str] = []
        for img in images:
            if isinstance(img, (str, Path)):
                p_img = Path(img)
                if p_img.exists():
                    encoded.append(base64.b64encode(p_img.read_bytes()).decode())
            elif isinstance(img, (bytes, bytearray)):
                encoded.append(base64.b64encode(img).decode())
        if encoded:
            user_msg["images"] = encoded
    messages.append(user_msg)

    # Merge profile options with the global GPU knob (profile takes precedence).
    opts = dict(p.options)
    num_gpu = config().get("runtime", {}).get("num_gpu")
    if num_gpu is not None and "num_gpu" not in opts:
        opts["num_gpu"] = num_gpu

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": opts,
    }
    # Per-profile think override takes precedence over the global runtime.think flag.
    # Only send `think` for models that actually support it — Ollama returns HTTP
    # 400 for non-thinking models (e.g. mistral) even with think=False.
    think = p.think if p.think is not None else think_enabled()
    if think and _is_thinking_model(model):
        payload["think"] = True
    if as_json:
        payload["format"] = "json"

    url = host() + "/api/chat"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    timeout = request_timeout()
    parts: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:  # each read resets the inactivity timeout
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                msg = chunk.get("message") or {}
                if msg.get("content"):
                    parts.append(msg["content"])
                if chunk.get("done"):
                    break
    except (TimeoutError, socket.timeout) as e:
        raise RuntimeError(
            f"Ollama produced no token for {timeout}s and was treated as hung "
            f"({e}). The first token can be slow — it includes loading the model "
            "into RAM and prefilling the prompt on CPU (worst for the storyboard "
            "stage). Raise runtime.request_timeout_seconds in config/models.yaml "
            "(or set it to 0 to wait indefinitely), or check the daemon "
            "(ollama serve / ollama ps). Re-run with --resume to continue."
        ) from e
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise RuntimeError(
            f"Ollama returned HTTP {e.code} for /api/chat: {body}\n"
            "(A 500 here is often out-of-memory loading a model on low-RAM hosts.)"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host()} ({e}). "
            "Is the daemon running?  Try:  ollama serve"
        ) from e

    return "".join(parts).strip()


def with_feedback(prompt: str, feedback: str | None) -> str:
    """Append reviewer feedback to a prompt when a gate revision is requested."""
    if not feedback:
        return prompt
    return (
        prompt
        + "\n\nREVISION REQUEST FROM REVIEWER:\n"
        + feedback
        + "\n\nRevise your output to address the above. Maintain the required JSON format exactly."
    )


def safe_json(raw: str):
    """Best-effort JSON parse tolerant of code fences and chatty wrappers.

    Small local models sometimes wrap JSON in ```fences``` or stray prose.
    Returns a sentinel dict with the raw text if parsing fails, so callers can
    degrade gracefully instead of crashing the pipeline.
    """
    s = raw.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"[\{\[].*[\}\]]", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"_raw": raw, "_parse_error": True}
