"""Google Gemini API helpers — image generation + Veo video — over stdlib urllib.

Used by `reel.imagegen` (character image generation) and `reel.i2v` (image-to-
video). Both are REST calls to generativelanguage.googleapis.com; no extra deps.

Docs:
  * Images: https://ai.google.dev/gemini-api/docs/image-generation
  * Video:  https://ai.google.dev/gemini-api/docs/video

The API key is stored in a permissions-protected file at ~/.config/reel/gemini_key
(chmod 600) — outside the project directory and never in env vars. Managed via
`python -m reel.secrets set/get/delete/status`. The key is read once per process
and cached; no interactive prompts are needed at runtime (avoids the
keyrings.cryptfile TTY requirement that caused silent render skips on --resume).
"""
from __future__ import annotations

import base64
import json
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com"

# File-based key store: ~/.config/reel/gemini_key  (chmod 600, owner-only)
_KEY_FILE = Path.home() / ".config" / "reel" / "gemini_key"

# In-process cache — read the file once per pipeline run, not on every API call.
_CACHED_KEY: str | None = None
_KEY_CHECKED: bool = False  # False = not yet read; True = already attempted


def _file_get() -> str | None:
    """Read the Gemini API key from the protected credentials file.

    Returns None (silently) when the file doesn't exist. Logs a clear warning
    when the file exists but can't be read (permissions, corruption, etc.) so
    the user knows to re-run `python -m reel.secrets set`.
    """
    if not _KEY_FILE.exists():
        return None
    try:
        key = _KEY_FILE.read_text(encoding="utf-8").strip()
        return key or None
    except Exception as e:
        print(f"[reel] ⚠ Gemini key file found but unreadable ({_KEY_FILE}): {e}",
              flush=True)
        return None


def api_key() -> str | None:
    """Resolve the Gemini API key. Reads the credentials file once per process
    and caches the result — no interactive prompts, no per-call file I/O.

    Store or update the key with:  python -m reel.secrets set
    """
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
    """True if an API key is present (cheap; does not call the network)."""
    return bool(api_key())


def key_hint() -> str:
    return "set a Gemini API key — run `python -m reel.secrets set`"


def _headers(json_body: bool = True) -> dict:
    h = {"x-goog-api-key": api_key() or ""}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _retry_after(err: "urllib.error.HTTPError", attempt: int, base: float) -> float:
    """Backoff seconds — honor a Retry-After header if present, else exponential."""
    hdr = err.headers.get("Retry-After") if getattr(err, "headers", None) else None
    if hdr:
        try:
            return float(hdr)
        except ValueError:
            pass
    return min(base * (2 ** attempt), 60.0)


def _post(url: str, body: dict, timeout: float, *, retries: int = 5, backoff: float = 5.0) -> dict:
    """POST JSON, retrying on transient errors (429 rate limit, 5xx). The Veo
    preview tier rate-limits aggressively, so a batch of submissions needs spacing."""
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


# ── image generation ─────────────────────────────────────────────────────────

def generate_image(prompt: str, out_path: Path, *,
                   model: str = "gemini-3.1-flash-image",
                   refs: list | None = None,
                   aspect_ratio: str | None = None,
                   image_size: str | None = None,
                   timeout: float = 300) -> bool:
    """Generate an image for `prompt` (optionally conditioned on `refs` reference
    images) and write it to `out_path`. Mirrors the official text-to-image sample
    (ai.google.dev/gemini-api/docs/image-generation): a minimal contents/parts
    body, no generationConfig. Returns True on success; raises on a hard API error.

    aspect_ratio / image_size are accepted for forward-compatibility but NOT sent —
    the v1 `:generateContent` endpoint rejects extra generationConfig fields for
    these image models, so we keep the body byte-for-byte with the documented
    minimal request (aspect ratio can be steered via the prompt text).
    """
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
# for image + video generation; all text/LLM stages run on the local open models
# (Ollama via reel.llm / reel.models.text).


# ── Veo video generation ─────────────────────────────────────────────────────

# Veo operation-error codes worth a retry — the op submits fine (HTTP 200) but the
# generation itself fails transiently. 13=INTERNAL, 14=UNAVAILABLE, 8=RESOURCE_EXHAUSTED.
_VEO_TRANSIENT = {8, 13, 14}


def generate_video(prompt: str, out_path: Path, *,
                   image_path: Path | None = None,
                   model: str = "veo-3.1-fast-generate-preview",
                   aspect_ratio: str = "16:9",
                   resolution: str = "720p",
                   poll_seconds: float = 10,
                   timeout_seconds: float = 1200,
                   op_retries: int = 3) -> bool:
    """Generate a clip for `prompt`, optionally seeded by `image_path` (image-to-
    video). Submits a long-running op, polls until done, downloads the mp4 to
    `out_path`. Returns True on success; raises on a hard API error.

    Transient *operation* failures (Veo internal/unavailable errors that surface
    only after the op completes) are retried up to `op_retries` times with backoff —
    distinct from the HTTP-level 429/5xx that `_post`/`_get_bytes` retry."""
    instance: dict = {"prompt": prompt}
    if image_path and Path(image_path).exists():
        # Veo wants the seed image as bytesBase64Encoded (NOT inlineData).
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        instance["image"] = {"bytesBase64Encoded": b64, "mimeType": "image/png"}
    body = {"instances": [instance],
            "parameters": {"aspectRatio": aspect_ratio, "resolution": resolution}}
    for attempt in range(op_retries + 1):
        try:
            return _run_veo_op(model, body, out_path, poll_seconds, timeout_seconds)
        except RuntimeError as e:
            if getattr(e, "veo_code", None) in _VEO_TRANSIENT and attempt < op_retries:
                wait = min(15.0 * (2 ** attempt), 90.0)
                print(f"[reel] Veo transient error (code {e.veo_code}) — resubmitting in "
                      f"{wait:.0f}s (attempt {attempt + 1}/{op_retries})", flush=True)
                time.sleep(wait)
                continue
            raise
    return False  # unreachable


def _run_veo_op(model: str, body: dict, out_path: Path,
                poll_seconds: float, timeout_seconds: float) -> bool:
    """One Veo submit + poll + download cycle. Raises RuntimeError (with `.veo_code`
    on operation failures) so the caller can retry transients."""
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
                err.veo_code = status["error"].get("code") if isinstance(status["error"], dict) else None
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
