"""Google Gemini API helpers — image generation + Veo video — over stdlib urllib.

Used by `reel.imagegen` (character image generation) and `reel.i2v` (image-to-
video). Both are REST calls to generativelanguage.googleapis.com; no extra deps.

Docs:
  * Images: https://ai.google.dev/gemini-api/docs/image-generation
  * Video:  https://ai.google.dev/gemini-api/docs/video

The API key is read from the environment — `GEMINIAPIKEY` (this project's name),
falling back to `GEMINI_API_KEY` / `GOOGLE_API_KEY`. Everything is best-effort:
callers catch failures and degrade (keep the text prompt) so a missing key or a
network hiccup never breaks the pipeline.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com"
_KEY_ENV = ("GEMINIAPIKEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def api_key() -> str | None:
    for name in _KEY_ENV:
        v = os.environ.get(name)
        if v:
            return v.strip()
    return None


def available() -> bool:
    """True if an API key is present (cheap; does not call the network)."""
    return bool(api_key())


def key_hint() -> str:
    return (f"set a Gemini API key — export {_KEY_ENV[0]}=… "
            f"(or {_KEY_ENV[1]}/{_KEY_ENV[2]})")


def _headers(json_body: bool = True) -> dict:
    h = {"x-goog-api-key": api_key() or ""}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 method="POST", headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get_bytes(url: str, timeout: float) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers=_headers(json_body=False))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("Content-Type", "")


def _inline(image_path: Path) -> dict:
    return {"inline_data": {"mime_type": "image/png",
                            "data": base64.b64encode(Path(image_path).read_bytes()).decode()}}


# ── image generation ─────────────────────────────────────────────────────────

def generate_image(prompt: str, out_path: Path, *,
                   model: str = "gemini-3.1-flash-image",
                   refs: list | None = None,
                   aspect_ratio: str | None = None,
                   image_size: str | None = None,
                   timeout: float = 300) -> bool:
    """Generate an image for `prompt` (optionally conditioned on `refs` reference
    images) and write it to `out_path`. Returns True on success; raises on a hard
    API error (callers catch)."""
    parts: list[dict] = [{"text": prompt}]
    for rp in (refs or []):
        if rp and Path(rp).exists():
            parts.append(_inline(Path(rp)))
    gen: dict = {"responseModalities": ["TEXT", "IMAGE"]}
    img_fmt = {}
    if aspect_ratio:
        img_fmt["aspectRatio"] = aspect_ratio
    if image_size:
        img_fmt["imageSize"] = image_size
    if img_fmt:
        gen["responseFormat"] = {"image": img_fmt}
    body = {"contents": [{"parts": parts}], "generationConfig": gen}
    data = _post(f"{BASE}/v1/models/{model}:generateContent", body, timeout)
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                Path(out_path).write_bytes(base64.b64decode(inline["data"]))
                return True
    raise RuntimeError("Gemini returned no image (check model/quota/prompt)")


# ── Veo video generation ─────────────────────────────────────────────────────

def generate_video(prompt: str, out_path: Path, *,
                   image_path: Path | None = None,
                   model: str = "veo-3.1-fast-generate-preview",
                   aspect_ratio: str = "16:9",
                   resolution: str = "720p",
                   poll_seconds: float = 10,
                   timeout_seconds: float = 1200) -> bool:
    """Generate a clip for `prompt`, optionally seeded by `image_path` (image-to-
    video). Submits a long-running op, polls until done, downloads the mp4 to
    `out_path`. Returns True on success; raises on a hard API error."""
    instance: dict = {"prompt": prompt}
    if image_path and Path(image_path).exists():
        inline = _inline(Path(image_path))["inline_data"]
        instance["image"] = {"inlineData": {"mimeType": inline["mime_type"],
                                            "data": inline["data"]}}
    body = {"instances": [instance],
            "parameters": {"aspectRatio": aspect_ratio, "resolution": resolution}}
    op = _post(f"{BASE}/v1beta/models/{model}:predictLongRunning", body, 120)
    name = op.get("name")
    if not name:
        raise RuntimeError("Veo did not return an operation name")

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        raw, _ = _get_bytes(f"{BASE}/v1beta/{name}", 60)
        status = json.loads(raw.decode())
        if status.get("done"):
            if status.get("error"):
                raise RuntimeError(f"Veo failed: {status['error']}")
            samples = (status.get("response", {})
                       .get("generateVideoResponse", {})
                       .get("generatedSamples", []))
            uri = samples[0].get("video", {}).get("uri") if samples else None
            if not uri:
                raise RuntimeError("Veo finished but returned no video URI")
            video, _ = _get_bytes(uri, 300)
            Path(out_path).write_bytes(video)
            return True
        time.sleep(poll_seconds)
    raise TimeoutError(f"Veo operation timed out after {timeout_seconds}s")
