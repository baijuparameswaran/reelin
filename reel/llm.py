"""Model-agnostic Ollama client shared by every agent.

Design goals:
  * Zero hard-coded model names in agent code — agents pick a *profile*.
  * Always-runnable: if a profile's preferred model isn't pulled, fall back to
    the first installed model in its `fallbacks` list.
  * Zero non-stdlib deps for the transport (urllib), so there's little to break.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


@dataclass
class Profile:
    name: str
    model: str
    fallbacks: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def host() -> str:
    return config().get("ollama_host", "http://localhost:11434")


def _api(path: str, payload: dict | None = None, method: str = "POST") -> dict:
    url = host() + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise RuntimeError(
            f"Ollama returned HTTP {e.code} for {path}: {body}\n"
            "(A 500 here is often out-of-memory loading a model on low-RAM hosts.)"
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
    return Profile(name, p["model"], p.get("fallbacks", []), p.get("options", {}))


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


def generate(
    prompt: str,
    *,
    profile: str = "fast",
    system: str | None = None,
    as_json: bool = False,
) -> str:
    """Single-turn generation against a local model selected by `profile`."""
    p = get_profile(profile)
    model = resolve_model(p)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": p.options,
    }
    if as_json:
        payload["format"] = "json"

    resp = _api("/api/chat", payload)
    return resp["message"]["content"].strip()


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
