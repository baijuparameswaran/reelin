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
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"

# Hardware-derived cap: keep prompts within the num_ctx budget of small local models.
MAX_CHARS = 12_000


@dataclass
class Profile:
    name: str
    model: str
    fallbacks: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    think: bool | None = None  # per-profile override; None = use runtime.think global


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def host() -> str:
    return config().get("ollama_host", "http://localhost:11434")


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


def get_profile(name: str) -> Profile:
    profiles = config()["profiles"]
    if name not in profiles:
        raise KeyError(f"Unknown profile {name!r}; available: {list(profiles)}")
    p = profiles[name]
    think = p.get("think")  # explicit bool in yaml → per-profile override
    return Profile(name, p["model"], p.get("fallbacks", []), p.get("options", {}),
                   think=think)


def agent_profile(agent: str) -> str:
    """The default profile name configured for a given agent."""
    return config().get("agent_profiles", {}).get(agent, "fast")


def resolve_model(profile: Profile) -> str:
    """Preferred model if installed, else the first installed fallback."""
    have = installed_models()

    def present(tag: str) -> bool:
        return any(m == tag or m.startswith(tag) for m in have)

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
_PROFILE_TIERS: tuple[str, ...] = ("fast", "quality", "thinking", "quality_high")


def next_profile(name: str) -> str | None:
    """Return the next higher quality profile, or None if already at the top."""
    try:
        idx = _PROFILE_TIERS.index(name)
        nxt = idx + 1
        return _PROFILE_TIERS[nxt] if nxt < len(_PROFILE_TIERS) else None
    except ValueError:
        return None


def generate(
    prompt: str,
    *,
    profile: str = "fast",
    system: str | None = None,
    as_json: bool = False,
    steer: bool = True,
    images: list | None = None,
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
    """
    p = get_profile(profile)
    model = resolve_model(p)
    steer_text = direction() if steer else None
    sys_msg = "\n\n".join(s for s in (steer_text, system) if s) or None
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
