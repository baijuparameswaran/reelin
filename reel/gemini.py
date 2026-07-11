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
from datetime import datetime, timezone
from pathlib import Path

from . import session as _session

BASE = "https://generativelanguage.googleapis.com"

# ── API invocation log ─────────────────────────────────────────────────────
# Set once per run (pipeline.run / stages.run_stage / the standalone `render`
# and `gen-video` CLI commands) so every actual Gemini/Veo call — not just the
# prompt text — gets a persistent audit line: timestamp, call kind, model,
# which transport served it (sdk vs. urllib fallback), and the outcome.
_LOG_DIR: Path | None = None


def set_log_dir(out: Path | str) -> None:
    """Point subsequent API-invocation log lines at `<out>/logs/gemini_api.log`."""
    global _LOG_DIR
    _LOG_DIR = Path(out)


def _log_call(kind: str, *, model: str = "", backend: str = "", outcome: str = "",
              path: Path | str | None = None, note: str = "") -> None:
    """Append one line to <out>/logs/gemini_api.log. No-ops if set_log_dir was
    never called (e.g. ad-hoc/unit-test use of this module) rather than raising."""
    if _LOG_DIR is None:
        return
    logs_dir = _LOG_DIR / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sid = _session.current(_LOG_DIR) or "-"
        bits = [ts, f"session={sid}", kind, f"model={model}", f"backend={backend}",
                f"outcome={outcome}"]
        if path is not None:
            bits.append(f"path={path}")
        if note:
            bits.append(note)
        with open(logs_dir / "gemini_api.log", "a", encoding="utf-8") as f:
            f.write("  ".join(bits) + "\n")
    except Exception:
        pass  # logging must never break an actual API call

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
    backend = "sdk" if sdk else "urllib"
    try:
        ok = (_generate_image_sdk(full_prompt, out_path, model=model, refs=refs, timeout=timeout)
              if sdk else
              _generate_image_urllib(full_prompt, out_path, model=model, refs=refs, timeout=timeout))
        _log_call("IMAGE", model=model, backend=backend,
                  outcome="success" if ok else "failed", path=out_path)
        return ok
    except Exception as e:
        _log_call("IMAGE", model=model, backend=backend,
                  outcome=f"error({type(e).__name__})", path=out_path)
        raise


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
                   model: str = "veo-3.1-generate-preview",
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
        for attempt in range(op_retries + 1):
            try:
                ok = _generate_video_sdk(prompt, out_path, image_path=image_path,
                                          model=model, aspect_ratio=aspect_ratio,
                                          resolution=resolution,
                                          duration_seconds=duration_seconds,
                                          poll_seconds=poll_seconds,
                                          timeout_seconds=timeout_seconds)
                _log_call("VIDEO", model=model, backend="sdk", outcome="success", path=out_path)
                return ok
            except ImportError:
                break  # SDK available but API call failed for non-transient reason → fall through
            except Exception as e:
                code = getattr(e, "veo_code", None)
                if code in _VEO_TRANSIENT and attempt < op_retries:
                    wait = min(15.0 * (2 ** attempt), 90.0)
                    print(f"[reel] Veo transient error (code {code}) — resubmitting in "
                          f"{wait:.0f}s (attempt {attempt + 1}/{op_retries})", flush=True)
                    _log_call("VIDEO", model=model, backend="sdk",
                              outcome=f"retry(code={code})", path=out_path,
                              note=f"attempt={attempt + 1}/{op_retries}")
                    time.sleep(wait)
                    continue
                print(f"[reel] SDK video failed ({type(e).__name__}: {e}) — falling back to urllib",
                      flush=True)
                _log_call("VIDEO", model=model, backend="sdk",
                          outcome=f"error({type(e).__name__}, code={code})", path=out_path,
                          note="falling back to urllib")
                break
    ok = _generate_video_urllib(prompt, out_path, image_path=image_path,
                                model=model, aspect_ratio=aspect_ratio,
                                resolution=resolution, poll_seconds=poll_seconds,
                                timeout_seconds=timeout_seconds,
                                op_retries=op_retries)
    _log_call("VIDEO", model=model, backend="urllib",
              outcome="success" if ok else "failed", path=out_path,
              note="fallback from sdk" if sdk else "")
    return ok


def _generate_video_sdk(prompt: str, out_path: Path, *,
                         image_path: Path | None,
                         model: str,
                         aspect_ratio: str,
                         resolution: str,
                         duration_seconds: int,
                         poll_seconds: float,
                         timeout_seconds: float) -> bool:
    """Veo via the google-genai SDK. Handles image-to-video and text-to-video."""
    genai, types = _sdk()
    client = genai.Client(api_key=api_key())

    # enhance_prompt is omitted outright — confirmed absent from the official
    # Veo API parameter table entirely (not a documented field for any Veo 3.1
    # variant), which is why it 400'd rather than just being ignored.
    #
    # person_generation's allowed value is fixed by generation MODE, not by
    # tier/model (same official table): text-to-video only accepts "allow_all";
    # image-to-video only accepts "allow_adult". Sending the wrong one for the
    # mode is what actually 400'd earlier ("allow_adult ... not supported" was
    # tested against text-to-video, the wrong mode for that value).
    has_image = bool(image_path and Path(image_path).exists())
    person_generation = "allow_adult" if has_image else "allow_all"

    if has_image:
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        image = types.Image(
            image_bytes=Path(image_path).read_bytes(),
            mime_type=mime,
        )
        # prompt is a top-level generate_videos() kwarg, not a GenerateVideosConfig
        # field (config rejects it with a pydantic "extra_forbidden" error).
        operation = client.models.generate_videos(
            model=model,
            image=image,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                number_of_videos=1,
                duration_seconds=duration_seconds,
                person_generation=person_generation,
            ),
        )
    else:
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                number_of_videos=1,
                duration_seconds=duration_seconds,
                person_generation=person_generation,
            ),
        )

    deadline = time.time() + timeout_seconds
    while not operation.done:
        if time.time() > deadline:
            raise TimeoutError(f"Veo operation timed out after {timeout_seconds}s")
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)

    for gen_video in (operation.result.generated_videos or []):
        video_bytes = client.files.download(file=gen_video.video)
        video_data = bytes(video_bytes) if not isinstance(video_bytes, (bytes, bytearray)) else video_bytes
        Path(out_path).write_bytes(video_data)
        return True

    # Operation completed with no video — same shape the urllib path already
    # parses (`operation.error` mirrors the raw REST `status["error"]`). Surface
    # the real code via `.veo_code` so the retry loop above can tell a transient
    # failure (8/13/14) from a hard one instead of always falling through to
    # urllib on a generic "no video" message.
    if operation.error:
        err = RuntimeError(f"Veo failed: {operation.error}")
        err.veo_code = (operation.error.get("code")
                        if isinstance(operation.error, dict) else None)
        raise err
    raise RuntimeError("Veo SDK returned no video")


def extend_video(prev_video_path: Path, prompt: str, out_path: Path, *,
                 model: str = "veo-3.1-generate-preview",
                 aspect_ratio: str = "16:9",
                 resolution: str = "720p",
                 poll_seconds: float = 10,
                 timeout_seconds: float = 1200,
                 op_retries: int = 3) -> bool:
    """Extend a previously Veo-generated clip via native video-to-video scene
    extension (SDK only — the Gemini Developer API's extend feature isn't on the
    raw predictLongRunning REST surface `_generate_video_urllib` uses).

    Unlike image-seeding (`generate_video(image_path=...)`, which only carries
    the last still frame forward), this feeds the whole prior CLIP back in, so
    Veo continues its ambient/music audio bed as well as the visual across the
    cut — the fix for audio that resets or drifts between clips of one scene.

    Requires: `prev_video_path` must itself be Veo-generated output (a Veo 3.1
    API constraint). Both `veo-3.1-generate-preview` and `-fast-` are reported
    to support extend; `-lite-` does not. The extend feature itself is fixed
    at 720p regardless of what the general video config specifies — this
    function trusts `resolution` as given rather than silently coercing it
    (that would violate the caller's configured output resolution for the
    clip); the caller (`i2v._gen_gemini`) is responsible for only invoking
    extend when resolution == "720p" and choosing the seed-continuity path
    otherwise. Retries transient Veo operation errors (8/13/14) with the same
    exponential backoff as `generate_video`, since a one-shot attempt would
    otherwise let a routine transient hiccup permanently downgrade this frame
    to the seed-continuity fallback instead of just retrying the extend call
    itself. Best-effort — raises on any non-transient failure; the caller
    catches it and falls back to the proven image-seed path.
    """
    for attempt in range(op_retries + 1):
        try:
            ok = _extend_video_once(prev_video_path, prompt, out_path, model=model,
                                    aspect_ratio=aspect_ratio, resolution=resolution,
                                    poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
            _log_call("VIDEO_EXTEND", model=model, backend="sdk", outcome="success", path=out_path)
            return ok
        except Exception as e:
            code = getattr(e, "veo_code", None)
            if code in _VEO_TRANSIENT and attempt < op_retries:
                wait = min(15.0 * (2 ** attempt), 90.0)
                print(f"[reel] Veo extend: transient error (code {code}) — resubmitting in "
                      f"{wait:.0f}s (attempt {attempt + 1}/{op_retries})", flush=True)
                _log_call("VIDEO_EXTEND", model=model, backend="sdk",
                          outcome=f"retry(code={code})", path=out_path,
                          note=f"attempt={attempt + 1}/{op_retries}")
                time.sleep(wait)
                continue
            _log_call("VIDEO_EXTEND", model=model, backend="sdk",
                      outcome=f"error({type(e).__name__}, code={code})", path=out_path)
            raise
    return False


def _extend_video_once(prev_video_path: Path, prompt: str, out_path: Path, *,
                       model: str, aspect_ratio: str, resolution: str,
                       poll_seconds: float, timeout_seconds: float) -> bool:
    """Single extend attempt — no retry, no logging (extend_video wraps both)."""
    genai, types = _sdk() or (None, None)
    if genai is None:
        raise ImportError("google-genai not installed — run: pip install google-genai")
    client = genai.Client(api_key=api_key())

    # Per the official parameter table, "Extension" is grouped with
    # text-to-video for person_generation: "allow_all" only (not "allow_adult",
    # which is for image-to-video/interpolation/reference-images instead).
    prev_video = types.Video.from_file(location=str(prev_video_path))
    operation = client.models.generate_videos(
        model=model,
        video=prev_video,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            number_of_videos=1,
            person_generation="allow_all",
        ),
    )
    deadline = time.time() + timeout_seconds
    while not operation.done:
        if time.time() > deadline:
            raise TimeoutError(f"Veo extend operation timed out after {timeout_seconds}s")
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)

    for gen_video in (operation.result.generated_videos or []):
        video_bytes = client.files.download(file=gen_video.video)
        video_data = bytes(video_bytes) if not isinstance(video_bytes, (bytes, bytearray)) else video_bytes
        Path(out_path).write_bytes(video_data)
        return True

    # Operation completed with no video — same shape the urllib/SDK video path
    # already parses (`operation.error` mirrors the raw REST `status["error"]`).
    # Surface the real code via `.veo_code` so the retry loop above can tell a
    # transient failure from a hard one instead of always falling through.
    if operation.error:
        err = RuntimeError(f"Veo extend failed: {operation.error}")
        err.veo_code = (operation.error.get("code")
                        if isinstance(operation.error, dict) else None)
        raise err
    raise RuntimeError("Veo extend returned no video")


def generate_video_with_references(prompt: str, out_path: Path,
                                   reference_image_paths: list[Path], *,
                                   model: str = "veo-3.1-generate-preview",
                                   aspect_ratio: str = "16:9",
                                   resolution: str = "720p",
                                   poll_seconds: float = 10,
                                   timeout_seconds: float = 1200,
                                   op_retries: int = 3) -> bool:
    """Generate a video clip anchored to up to 3 character/object identity
    reference images — Veo 3.1's "Ingredients to Video" `reference_images`
    config field, each typed ASSET (STYLE is Veo-2-only, rejected on 3.1;
    confirmed via live SDK introspection — see project memory
    `veo-character-consistency`). Used at a "shot boundary" where MORE THAN
    ONE character is in frame: unlike a single `image=` seed (which can
    only carry ONE identity forward, whichever character
    `pipeline._frame_char_anchor` happened to pick first), every in-frame
    character gets its own locked identity reference.

    SDK-only — the raw REST `predictLongRunning` surface
    `_generate_video_urllib` uses has not been verified to accept this
    field (unlike image-seeding/extend, which have both been exercised over
    both transports); raises `RuntimeError` immediately if the SDK isn't
    available, which the caller (`i2v._gen_gemini`) treats as any other
    reference-call failure and falls back to the proven single-image-seed
    path.

    MUTUALLY EXCLUSIVE with image-seed/last-frame continuity in the same
    call — a genuine Veo API constraint (confirmed via a live ad-hoc SDK
    smoke test, see the same memory note), not a limitation of this
    wrapper: a reference-image call can't also chain from a previous
    clip's tail frame, so this one clip trades frame-to-frame visual
    continuity for multi-character identity-lock. Always requests Veo's
    only valid duration for this mode, 8s (see
    `i2v._veo_nearest_valid_duration`'s `force_max` cases), and
    `person_generation="allow_adult"` (the official parameter table groups
    reference-images with image-to-video/interpolation for this field, not
    with text-to-video/extension's "allow_all" — see the comment in
    `_extend_video_once` above). Retries transient Veo operation errors
    (8/13/14) with the same backoff as `generate_video`/`extend_video`.
    Best-effort — raises on any non-transient failure; the caller catches
    it and falls back to the single-image-seed path."""
    for attempt in range(op_retries + 1):
        try:
            ok = _generate_video_with_references_once(
                prompt, out_path, reference_image_paths, model=model,
                aspect_ratio=aspect_ratio, resolution=resolution,
                poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
            _log_call("VIDEO_REFS", model=model, backend="sdk", outcome="success", path=out_path)
            return ok
        except Exception as e:
            code = getattr(e, "veo_code", None)
            if code in _VEO_TRANSIENT and attempt < op_retries:
                wait = min(15.0 * (2 ** attempt), 90.0)
                print(f"[reel] Veo reference-image call: transient error (code {code}) — "
                      f"resubmitting in {wait:.0f}s (attempt {attempt + 1}/{op_retries})", flush=True)
                _log_call("VIDEO_REFS", model=model, backend="sdk",
                          outcome=f"retry(code={code})", path=out_path,
                          note=f"attempt={attempt + 1}/{op_retries}")
                time.sleep(wait)
                continue
            _log_call("VIDEO_REFS", model=model, backend="sdk",
                      outcome=f"error({type(e).__name__}, code={code})", path=out_path)
            raise
    return False


def _generate_video_with_references_once(prompt: str, out_path: Path,
                                          reference_image_paths: list[Path], *,
                                          model: str, aspect_ratio: str, resolution: str,
                                          poll_seconds: float, timeout_seconds: float) -> bool:
    """Single reference-image attempt — no retry, no logging (the caller
    wraps both, mirroring `_extend_video_once`)."""
    genai, types = _sdk() or (None, None)
    if genai is None:
        raise ImportError("google-genai not installed — run: pip install google-genai")
    client = genai.Client(api_key=api_key())

    refs = []
    for p in reference_image_paths[:3]:   # Veo 3.1 accepts at most 3 reference images
        p = Path(p)
        mime = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
        refs.append(types.VideoGenerationReferenceImage(
            image=types.Image(image_bytes=p.read_bytes(), mime_type=mime),
            reference_type=types.VideoGenerationReferenceType.ASSET,
        ))

    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            number_of_videos=1,
            duration_seconds=8,             # only valid value for a reference-image call
            person_generation="allow_adult",
            reference_images=refs,
        ),
    )
    deadline = time.time() + timeout_seconds
    while not operation.done:
        if time.time() > deadline:
            raise TimeoutError(f"Veo reference-image operation timed out after {timeout_seconds}s")
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)

    for gen_video in (operation.result.generated_videos or []):
        video_bytes = client.files.download(file=gen_video.video)
        video_data = bytes(video_bytes) if not isinstance(video_bytes, (bytes, bytearray)) else video_bytes
        Path(out_path).write_bytes(video_data)
        return True

    if operation.error:
        err = RuntimeError(f"Veo reference-image call failed: {operation.error}")
        err.veo_code = (operation.error.get("code")
                        if isinstance(operation.error, dict) else None)
        raise err
    raise RuntimeError("Veo reference-image call returned no video")


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
            code = getattr(e, "veo_code", None)
            if code in _VEO_TRANSIENT and attempt < op_retries:
                wait = min(15.0 * (2 ** attempt), 90.0)
                print(f"[reel] Veo transient error (code {code}) — resubmitting in "
                      f"{wait:.0f}s (attempt {attempt + 1}/{op_retries})", flush=True)
                _log_call("VIDEO", model=model, backend="urllib",
                          outcome=f"retry(code={code})", path=out_path,
                          note=f"attempt={attempt + 1}/{op_retries}")
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
