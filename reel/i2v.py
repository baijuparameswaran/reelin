"""Pluggable image-to-video backend — render storyboard frames into motion.

This is the *next phase* after storyboard + screenplay: turn each rendered frame
(a still) into a short clip, and chain clips for continuity. Video generation is a
heavier modality than the sd-turbo casting stills — the efficient open models
(LTX-Video / LTX-2, Wan 2.x, CogVideoX-I2V, HunyuanVideo) all want a GPU
(~12 GB+ VRAM), so on a CPU-only host this is a no-op that degrades gracefully.
Model-agnostic by design (like `reel.llm`): pick a backend + model in config.

Backends (config `video.backend`):
  * "diffusers" — in-process image-to-video via 🤗 diffusers on a **GPU** host.
    Model-agnostic through `video.pipeline_class` (e.g. `LTXImageToVideoPipeline`,
    `WanImageToVideoPipeline`, `CogVideoXImageToVideoPipeline`) + `video.model`.
    Lazy-imported so torch/diffusers stay optional; requires CUDA (CPU is
    impractical for video).
  * "comfyui" / "http" — POST the conditioning image(s) + prompt to a remote
    endpoint that runs the heavy model (ComfyUI / a GPU box / cloud), and save the
    returned video. Zero extra Python deps (stdlib urllib+base64). This is the
    recommended route from this CPU-only host: render stills locally, generate
    motion remotely.
  * "none" — disabled (default). The pipeline keeps each frame's still + prompt.

Best-effort: any failure logs a warning and returns False; the scene-render stage
keeps the per-frame stills regardless of whether clips got produced.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from . import gemini
from . import llm


def _cfg() -> dict:
    return llm.config().get("video", {})


def enabled() -> bool:
    c = _cfg()
    return bool(c.get("enabled", False)) and backend() != "none"


def backend() -> str:
    """Resolved video backend. `auto` → Gemini Veo when a key exists (policy:
    prefer Gemini for video generation), else the configured open backend."""
    b = _cfg().get("backend", "none")
    if b == "auto":
        return "gemini" if gemini.available() else _cfg().get("open_backend", "diffusers")
    return b


def _log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


# ── availability (cheap; no model load) ──────────────────────────────────────

def available() -> bool:
    """True if the configured backend can actually render right now."""
    if not enabled():
        return False
    b = backend()
    if b in ("gemini", "veo"):
        return gemini.available()
    if b == "diffusers":
        if not all(importlib.util.find_spec(m) for m in ("torch", "diffusers")):
            return False
        try:  # video on CPU is impractical — require a GPU
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    if b in ("comfyui", "http"):
        host = _cfg().get("host", "")
        if not host:
            return False
        try:
            with urllib.request.urlopen(host, timeout=4):
                return True
        except Exception:
            return False
    return False


def unavailable_hint() -> str:
    b = backend()
    if b in ("gemini", "veo"):
        return gemini.key_hint()
    if b == "none":
        return ("no video backend — set video.backend (ltx/wan via diffusers on a "
                "GPU host, or comfyui/http to a remote GPU endpoint)")
    if b == "diffusers":
        return ("diffusers/torch+CUDA not available — video needs a GPU; install "
                "the model's deps or use video.backend: comfyui to offload remotely")
    return f"no video endpoint reachable at {_cfg().get('host', '(unset)')}"


# ── prompt / params ──────────────────────────────────────────────────────────

def _full_prompt(prompt: str) -> str:
    suffix = _cfg().get("style_suffix", "cinematic, smooth natural motion, consistent character")
    return f"{prompt.strip()}, {suffix}" if suffix and prompt else (prompt or suffix)


def _frames(c: dict) -> int:
    return int(round(float(c.get("seconds", 4)) * float(c.get("fps", 24))))


# ── backends ─────────────────────────────────────────────────────────────────

def _gen_gemini(images: list[Path], prompt: str, out_path: Path) -> bool:
    """Veo image-to-video via the Gemini API. Seeds from the last keyframe (the
    reference image produced by the image stage); text-to-video if none given."""
    c = _cfg()
    return gemini.generate_video(
        _full_prompt(prompt), Path(out_path),
        image_path=images[-1] if images else None,
        #model=c.get("model", "veo-3.1-fast-generate-preview"),
        model=c.get("model", "veo-3.1-lite-generate-preview"),
        aspect_ratio=c.get("aspect_ratio", "16:9"),
        resolution=c.get("resolution", "720p"),
        poll_seconds=c.get("poll_seconds", 10),
        timeout_seconds=c.get("timeout_seconds", 1200) or 1200,
    )


_PIPE = None  # diffusers i2v pipeline, cached (load is very expensive)


def _gen_diffusers(images: list[Path], prompt: str, out_path: Path) -> bool:
    """Local image-to-video on a GPU host. Model-agnostic via `video.pipeline_class`."""
    global _PIPE
    c = _cfg()
    model = c.get("model", "Lightricks/LTX-Video")
    cls_name = c.get("pipeline_class", "LTXImageToVideoPipeline")
    if _PIPE is None:
        import diffusers
        import torch
        from diffusers.utils import export_to_video  # noqa: F401 (validate availability)
        _log(f"      loading video model {model} via {cls_name} (GPU) …")
        cls = getattr(diffusers, cls_name)
        _PIPE = cls.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    from PIL import Image
    from diffusers.utils import export_to_video
    init = Image.open(images[-1]).convert("RGB")  # last keyframe drives the start
    size = int(c.get("size", 768))
    kwargs = dict(prompt=_full_prompt(prompt), image=init,
                  num_frames=_frames(c), width=size, height=size)
    # first+last-frame conditioning when two keyframes are supplied and supported
    if len(images) >= 2 and "last_image" in getattr(_PIPE, "__call__").__doc__ or "":
        kwargs["last_image"] = Image.open(images[0]).convert("RGB")
    result = _PIPE(**kwargs).frames[0]
    export_to_video(result, str(out_path), fps=int(c.get("fps", 24)))
    return True


def _gen_http(images: list[Path], prompt: str, out_path: Path) -> bool:
    """Offload to a remote GPU endpoint (ComfyUI / custom). Sends prompt + base64
    keyframe image(s); expects JSON {"video": "<base64 mp4>"} or raw video bytes."""
    c = _cfg()
    host = c.get("host", "")
    payload = {
        "prompt": _full_prompt(prompt),
        "images": [base64.b64encode(Path(p).read_bytes()).decode() for p in images],
        "num_frames": _frames(c),
        "fps": int(c.get("fps", 24)),
        "size": int(c.get("size", 768)),
        "params": c.get("params", {}),
    }
    endpoint = c.get("endpoint", host.rstrip("/") + "/i2v")
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    timeout = c.get("timeout_seconds", 1200) or None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    if "application/json" in ctype:
        data = json.loads(body.decode())
        vid = data.get("video") or (data.get("videos") or [None])[0]
        if not vid:
            raise RuntimeError("endpoint returned no video")
        out_path.write_bytes(base64.b64decode(vid.split(",", 1)[-1]))
    else:
        out_path.write_bytes(body)  # raw video bytes
    return True


# ── public API ───────────────────────────────────────────────────────────────

def generate_clip(images, prompt: str, out_path: Path) -> bool:
    """Render a clip to `out_path` (mp4) conditioned on one or more keyframe
    `images` (a Path or list — last is the start frame; a leading second image is
    used as the prior/last-frame anchor for continuity when the model supports it).
    Returns success; never raises fatally."""
    imgs = [Path(p) for p in ([images] if isinstance(images, (str, Path)) else images) if p]
    imgs = [p for p in imgs if p.exists()]
    b = backend()
    # Veo can do text-to-video; the other backends require a seed image.
    if not imgs and b not in ("gemini", "veo"):
        return False
    try:
        if b in ("gemini", "veo"):
            return _gen_gemini(imgs, prompt, out_path)
        if b == "diffusers":
            return _gen_diffusers(imgs, prompt, out_path)
        if b in ("comfyui", "http"):
            return _gen_http(imgs, prompt, out_path)
        return False
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        _log(f"      video backend error ({e}); kept frame still only")
        return False
    except Exception as e:  # model load / OOM / decode — stay non-fatal
        _log(f"      clip render failed ({type(e).__name__}: {e}); kept frame still only")
        return False


def last_frame(video_path: Path, out_image: Path) -> Path | None:
    """Extract the final frame of a clip as an image (for cross-clip continuity).
    Best-effort via imageio, then ffmpeg; None if neither is available."""
    try:
        import imageio.v3 as iio
        frames = iio.imread(str(video_path), index=None)  # (T,H,W,C)
        from PIL import Image
        Image.fromarray(frames[-1]).save(out_image)
        return out_image
    except Exception:
        pass
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-1", "-i", str(video_path),
             "-update", "1", "-q:v", "2", str(out_image)],
            check=True, capture_output=True, timeout=60,
        )
        return out_image if out_image.exists() else None
    except Exception:
        return None
