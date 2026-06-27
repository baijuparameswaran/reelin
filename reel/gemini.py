"""Google Gemini API helpers — image generation + Veo video.

Uses the official `google-genai` SDK when available (preferred); falls back to
raw urllib REST calls so the text pipeline keeps working without the SDK.

Docs:
  * SDK:    https://ai.google.dev/gemini-api/docs/sdks
  * Images: https://ai.google.dev/gemini-api/docs/image-generation
  * Video:  https://ai.google.dev/gemini-api/docs/video

API key: stored at ~/.config/reel/gemini_key (chmod 600). Managed via
`python -m reel.secrets set/get/delete/status`. Read once per process and cached.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com"

# File-based key store: ~/.config/reel/gemini_key  (chmod 600, owner-only)
_KEY_FILE = Path.home() / ".config" / "reel" / "gemini_key"

_CACHED_KEY: str | None = None
_KEY_CHECKED: bool = False


def _file_get() -> str | None:
    if not _KEY_FILE.exists():
        return None
    try:
        key = _KEY_FILE.read_text(encoding="utf-8").strip()
        return key or None
    except Exception as e:
        print(f"[reel] ⚠ Gemini key file found but unreadable ({_KEY_FILE}): {e}", flush=True)
        return None


def api_key() -> str | None:
    """Return the Gemini API key (cached after first read)."""
    global _CACHED_KEY, _KEY_CHECKED
    if not _KEY_CHECKED:
        _CACHED_KEY = _file_get()
        _KEY_CHECKED = True
    return _CACHED_KEY


def _invalidate_key_cache() -> None:
    """Force the next api_key() call to re-read the file (used by secrets.set)."""
    global _CACHED_KEY, _KEY_CHECKED
    _CACHED_KEY = None
    _KEY_CHECKED = False


def available() -> bool:
    return bool(api_key())


def key_hint() -> str:
    return "set a Gemini API key — run `python -m reel.secrets set`"


# ── SDK availability ──────────────────────────────────────────────────────────

def _sdk() -> tuple | None:
    """Return (genai, types) from the google-genai SDK, or None if not installed."""
    try:
        from google import genai
        from google.genai import types
        return genai, types
    except ImportError:
        return None


def _sdk_client():
    """Return a google-genai Client, or raise if the SDK is unavailable."""
    sdk = _sdk()
    if sdk is None:
        raise ImportError("google-genai not installed — run: pip install google-genai")
    genai, _ = sdk
    return genai.Client(api_key=api_key())


# ── urllib helpers (fallback) ─────────────────────────────────────────────────

def _headers(json_body: bool = True) -> dict:
    h = {"x-goog-api-key": api_key() or ""}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _retry_after(err: urllib.error.HTTPError, attempt: int, base: float) -> float:
    hdr = err.headers.get("Retry-After") if getattr(err, "headers", None) else None
    if hdr:
        try:
            return float(hdr)
        except ValueError:
            pass
    return min(base * (2 ** attempt), 60.0)


def _post(url: str, body: dict, timeout: float, *, retries: int = 5, backoff: float = 5.0) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST", headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries:
                wait = _retry_after(e, attempt, backoff)
                print(f"[reel] Gemini {e.code} — backing off {wait:.0f}s "
                      f"(attempt {attempt + 1}/{retries})", flush=True)
                time.sleep(wait)
                continue
            raise


def _get_bytes(url: str, timeout: float, *, retries: int = 5, backoff: float = 5.0) -> tuple[bytes, str]:
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=_headers(json_body=False))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries:
                time.sleep(_retry_after(e, attempt, backoff))
                continue
            raise


def _inline(image_path: Path) -> dict:
    return {"inline_data": {"mime_type": "image/png",
                            "data": base64.b64encode(Path(image_path).read_bytes()).decode()}}


# ── image generation ──────────────────────────────────────────────────────────

def generate_image(prompt: str, out_path: Path, *,
                   model: str = "gemini-3.1-flash-image",
                   refs: list | None = None,
                   aspect_ratio: str | None = None,
                   image_size: str | None = None,
                   timeout: float = 300) -> bool:
    """Generate an image and write it to out_path. SDK-first, urllib fallback.

    aspect_ratio / image_size are appended as composition hints to the prompt
    (the generate_content path has no dedicated API parameter for these; Imagen
    models do, but this function targets the flash-image generate_content path).
    """
    full_prompt = _image_prompt(prompt, aspect_ratio=aspect_ratio, image_size=image_size)
    sdk = _sdk()
    if sdk:
        return _generate_image_sdk(full_prompt, out_path, model=model, refs=refs, timeout=timeout)
    return _generate_image_urllib(full_prompt, out_path, model=model, refs=refs, timeout=timeout)


def _image_prompt(prompt: str, *, aspect_ratio: str | None, image_size: str | None) -> str:
    """Append layout hints to the prompt when the caller specifies aspect ratio / size."""
    hints = []
    if aspect_ratio:
        mapping = {"16:9": "landscape widescreen", "9:16": "portrait vertical",
                   "1:1": "square", "4:3": "landscape", "3:4": "portrait"}
        hints.append(mapping.get(aspect_ratio, aspect_ratio) + " composition")
    if image_size:
        hints.append(f"{image_size} resolution")
    if hints:
        return f"{prompt.rstrip('.')}. {', '.join(hints)}."
    return prompt


def _generate_image_sdk(prompt: str, out_path: Path, *, model: str,
                         refs: list | None, timeout: float) -> bool:
    genai, types = _sdk()
    client = genai.Client(api_key=api_key())
    parts = [types.Part.from_text(text=prompt)]
    for rp in (refs or []):
        if rp and Path(rp).exists():
            parts.append(types.Part.from_bytes(data=Path(rp).read_bytes(), mime_type="image/png"))
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    for candidate in (response.candidates or []):
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline:
                raw = getattr(inline, "data", None)
                if raw:
                    Path(out_path).write_bytes(
                        base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
                    )
                    return True
    raise RuntimeError("Gemini returned no image")


def _generate_image_urllib(prompt: str, out_path: Path, *, model: str,
                            refs: list | None, timeout: float) -> bool:
    parts: list[dict] = [{"text": prompt}]
    for rp in (refs or []):
        if rp and Path(rp).exists():
            parts.append(_inline(Path(rp)))
    body: dict = {"contents": [{"parts": parts}]}
    data = _post(f"{BASE}/v1/models/{model}:generateContent", body, timeout)
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                Path(out_path).write_bytes(base64.b64decode(inline["data"]))
                return True
    raise RuntimeError("Gemini returned no image (check model/quota/prompt)")


# NB: no text generation here by design — per project policy Gemini is used ONLY
# for image + video; all text/LLM stages run on the local open models (Ollama).


# ── Veo video generation ──────────────────────────────────────────────────────

# Veo transient operation error codes worth retrying.
_VEO_TRANSIENT = {8, 13, 14}


def generate_video(prompt: str, out_path: Path, *,
                   image_path: Path | None = None,
                   model: str = "veo-3.0-generate-preview",
                   aspect_ratio: str = "16:9",
                   resolution: str = "720p",
                   duration_seconds: int = 8,
                   poll_seconds: float = 10,
                   timeout_seconds: float = 1200,
                   op_retries: int = 3) -> bool:
    """Generate a video clip via Veo. SDK-first, urllib fallback.

    Submits a long-running generation, polls until done, writes the mp4 to
    out_path. image_path seeds image-to-video continuity (the prior clip's
    tail frame or a character reference). Returns True on success.
    """
    sdk = _sdk()
    if sdk:
        try:
            return _generate_video_sdk(prompt, out_path, image_path=image_path,
                                        model=model, aspect_ratio=aspect_ratio,
                                        duration_seconds=duration_seconds,
                                        poll_seconds=poll_seconds,
                                        timeout_seconds=timeout_seconds)
        except ImportError:
            pass  # SDK available but API call failed for non-transient reason → fall through
        except Exception as e:
            print(f"[reel] SDK video failed ({type(e).__name__}: {e}) — falling back to urllib",
                  flush=True)
    return _generate_video_urllib(prompt, out_path, image_path=image_path,
                                   model=model, aspect_ratio=aspect_ratio,
                                   resolution=resolution, poll_seconds=poll_seconds,
                                   timeout_seconds=timeout_seconds,
                                   op_retries=op_retries)


def _generate_video_sdk(prompt: str, out_path: Path, *,
                         image_path: Path | None,
                         model: str,
                         aspect_ratio: str,
                         duration_seconds: int,
                         poll_seconds: float,
                         timeout_seconds: float) -> bool:
    """Veo via the google-genai SDK. Handles image-to-video and text-to-video."""
    genai, types = _sdk()
    client = genai.Client(api_key=api_key())

    cfg = types.GenerateVideoConfig(
        aspect_ratio=aspect_ratio,
        number_of_videos=1,
        duration_seconds=duration_seconds,
        person_generation="allow_adult",
        enhance_prompt=False,   # use the prompt exactly as written
    )

    if image_path and Path(image_path).exists():
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        image = types.Image(
            image_bytes=Path(image_path).read_bytes(),
            mime_type=mime,
        )
        operation = client.models.generate_video(
            model=model,
            image=image,
            config=types.GenerateVideoConfig(
                prompt=prompt,
                aspect_ratio=cfg.aspect_ratio,
                number_of_videos=cfg.number_of_videos,
                duration_seconds=cfg.duration_seconds,
                person_generation=cfg.person_generation,
                enhance_prompt=cfg.enhance_prompt,
            ),
        )
    else:
        operation = client.models.generate_video(
            model=model,
            prompt=prompt,
            config=cfg,
        )

    deadline = time.time() + timeout_seconds
    while not operation.done:
        if time.time() > deadline:
            raise TimeoutError(f"Veo operation timed out after {timeout_seconds}s")
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)

    for gen_video in (operation.result.generated_videos or []):
        video_bytes = client.files.download(gen_video.video)
        video_data = bytes(video_bytes) if not isinstance(video_bytes, (bytes, bytearray)) else video_bytes
        Path(out_path).write_bytes(video_data)
        return True
    raise RuntimeError("Veo SDK returned no video")


def _generate_video_urllib(prompt: str, out_path: Path, *,
                            image_path: Path | None,
                            model: str,
                            aspect_ratio: str,
                            resolution: str,
                            poll_seconds: float,
                            timeout_seconds: float,
                            op_retries: int) -> bool:
    """Veo via raw REST (urllib) — the original implementation, kept as fallback."""
    instance: dict = {"prompt": prompt}
    if image_path and Path(image_path).exists():
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        instance["image"] = {"bytesBase64Encoded": b64, "mimeType": "image/png"}
    body = {"instances": [instance],
            "parameters": {"aspectRatio": aspect_ratio, "resolution": resolution}}
    for attempt in range(op_retries + 1):
        try:
            return _run_veo_op_urllib(model, body, out_path, poll_seconds, timeout_seconds)
        except RuntimeError as e:
            if getattr(e, "veo_code", None) in _VEO_TRANSIENT and attempt < op_retries:
                wait = min(15.0 * (2 ** attempt), 90.0)
                print(f"[reel] Veo transient error (code {e.veo_code}) — resubmitting in "
                      f"{wait:.0f}s (attempt {attempt + 1}/{op_retries})", flush=True)
                time.sleep(wait)
                continue
            raise
    return False


def _run_veo_op_urllib(model: str, body: dict, out_path: Path,
                        poll_seconds: float, timeout_seconds: float) -> bool:
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
                err = RuntimeError(f"Veo failed: {status['error']}")
                err.veo_code = (status["error"].get("code")
                                if isinstance(status["error"], dict) else None)
                raise err
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
