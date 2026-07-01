"""Pluggable text-to-image backend for character casting renders.

Text-to-image is a different modality than the Ollama text pipeline (Ollama
generates text, not images), so this is a separate, model-agnostic client driven
by the `image` block in config/models.yaml. Backends:

  * "diffusers" — in-process generation via the optional `diffusers` + `torch`
    stack (lazy-imported, so they stay optional deps). Recommended model for a
    CPU host is `stabilityai/sd-turbo` (distilled, few-step, fast). On a GPU box
    swap in SDXL or FLUX.1-schnell by editing config only.
  * "auto1111" — HTTP POST to an AUTOMATIC1111 / stable-diffusion.cpp compatible
    `/sdapi/v1/txt2img` endpoint. Zero extra Python deps (stdlib urllib+base64);
    point it at a running image server.
  * "none" / enabled:false — skip image rendering.

Best-effort by design: any failure logs a warning and returns False, so the
casting/pipeline stages degrade gracefully — the text `visual_prompt` is always
kept regardless of whether a picture got rendered.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

from . import gemini
from . import llm


def _cfg() -> dict:
    return llm.config().get("image", {})


def enabled() -> bool:
    c = _cfg()
    return bool(c.get("enabled", False)) and backend() != "none"


def backend() -> str:
    """Resolved image backend. `auto` → Gemini when a key exists (policy: prefer
    Gemini for image generation), else the configured open backend."""
    b = _cfg().get("backend", "none")
    if b == "auto":
        return "gemini" if gemini.available() else _cfg().get("open_backend", "diffusers")
    return b


def _log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


# ── availability check (cheap; no model load) ────────────────────────────────

def available() -> bool:
    """True if the configured backend can actually run right now."""
    if not enabled():
        return False
    b = backend()
    if b == "gemini":
        return gemini.available()
    if b == "diffusers":
        return all(importlib.util.find_spec(m) for m in ("torch", "diffusers"))
    if b == "auto1111":
        host = _cfg().get("host", "http://localhost:7860")
        try:
            with urllib.request.urlopen(host + "/sdapi/v1/sd-models", timeout=3):
                return True
        except Exception:
            return False
    return False


def unavailable_hint() -> str:
    b = backend()
    if b == "gemini":
        return gemini.key_hint()
    if b == "diffusers":
        return ("diffusers/torch not installed — run "
                "`pip install -r requirements-image.txt` to enable casting renders")
    if b == "auto1111":
        return (f"no image server at {_cfg().get('host')} — start AUTOMATIC1111 "
                "(or set image.backend: diffusers)")
    return "image generation disabled (config 'image')"


# ── prompt shaping ───────────────────────────────────────────────────────────

def _dims() -> tuple[int, int]:
    """(width, height) for the render — portrait aspect if width/height are set,
    else the square `size` fallback. A taller-than-wide frame lets a full
    head-to-toe figure fit instead of being cropped to a bust."""
    c = _cfg()
    size = int(c.get("size", 512))
    return int(c.get("width", size)), int(c.get("height", size))


def _full_prompt(prompt: str) -> str:
    suffix = _cfg().get(
        "style_suffix",
        "character concept portrait, full figure, plain neutral background, soft studio light",
    )
    return f"{prompt.strip()}, {suffix}" if suffix else prompt.strip()


# ── backends ─────────────────────────────────────────────────────────────────

_PIPE = None   # text->image pipeline, cached across characters (load is expensive)


def _gen_diffusers(prompt: str, out_path: Path) -> bool:
    global _PIPE
    c = _cfg()
    model = c.get("model", "stabilityai/sd-turbo")
    if _PIPE is None:
        from diffusers import AutoPipelineForText2Image
        import torch
        _log(f"      loading image model {model} (first use, CPU) …")
        _PIPE = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float32)
        _PIPE.to("cpu")
        _PIPE.set_progress_bar_config(disable=True)
    width, height = _dims()
    img = _PIPE(
        _full_prompt(prompt),
        num_inference_steps=int(c.get("steps", 4)),
        guidance_scale=float(c.get("guidance_scale", 0.0)),
        height=height, width=width,
    ).images[0]
    img.save(out_path)
    return True


def _gen_auto1111(prompt: str, out_path: Path) -> bool:
    c = _cfg()
    host = c.get("host", "http://localhost:7860")
    width, height = _dims()
    payload = {
        "prompt": _full_prompt(prompt),
        "negative_prompt": c.get("negative_prompt", ""),
        "steps": int(c.get("steps", 4)),
        "width": width, "height": height,
        "cfg_scale": float(c.get("guidance_scale", 1.0)),
        "sampler_name": c.get("sampler", "Euler a"),
    }
    req = urllib.request.Request(
        host + "/sdapi/v1/txt2img",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    timeout = c.get("timeout_seconds", 600) or None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    images = data.get("images") or []
    if not images:
        raise RuntimeError("image server returned no images")
    out_path.write_bytes(base64.b64decode(images[0].split(",", 1)[-1]))
    return True


def _gen_gemini(prompt: str, out_path: Path) -> bool:
    c = _cfg()
    return gemini.generate_image(
        prompt, Path(out_path),
        model=c.get("model", "gemini-2.5-flash-image"),
        aspect_ratio=c.get("aspect_ratio"),
        image_size=c.get("image_size"),
        timeout=c.get("timeout_seconds", 300) or 300,
    )


# ── public entry point ───────────────────────────────────────────────────────

def generate_image(prompt: str, out_path: Path) -> bool:
    """Render `prompt` to `out_path` (PNG). Returns success; never raises fatally."""
    if not prompt:
        return False
    try:
        b = backend()
        if b == "gemini":
            return _gen_gemini(prompt, out_path)
        if b == "diffusers":
            return _gen_diffusers(prompt, out_path)
        if b == "auto1111":
            return _gen_auto1111(prompt, out_path)
        return False
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        _log(f"      image backend error ({e}); kept visual_prompt only")
        return False
    except Exception as e:  # model load / OOM / decode — stay non-fatal
        _log(f"      image render failed ({type(e).__name__}: {e}); kept visual_prompt only")
        return False


