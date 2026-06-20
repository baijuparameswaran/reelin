"""Free stock-photo lookup for casting *actor* identities.

The casting layer separates the actor (a real, consistent performer) from the
character (that actor aged/costumed into the role). This module sources the actor
as a real **Creative-Commons stock photo** so the identity is a genuine human face
that can anchor — via img2img — every character/scene render of that role.

Backend is **Openverse** (https://openverse.org): a CC-licensed image search over
Flickr/Wikimedia/etc., **no API key required**. We filter to licenses that permit
*modification* (so deriving the character render is allowed) and keep full
attribution, since CC-BY/-SA require crediting the creator.

Best-effort: any failure returns None and the pipeline falls back to generating
the actor from text — nothing breaks if the network or API is unavailable.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import llm

API = "https://api.openverse.org/v1/images/"
UA = "reel/0.1 (casting actor lookup; https://github.com/local/reel)"


def _cfg() -> dict:
    return llm.config().get("image", {}).get("stock", {})


def enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def _log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search(query: str, n: int = 8) -> list[dict]:
    """CC image results for `query`, restricted to modification-allowed licenses."""
    c = _cfg()
    params = urllib.parse.urlencode({
        "q": query,
        "page_size": n,
        # only photos we're allowed to modify (the character render is a derivative)
        "license_type": c.get("license_filter", "modification"),
        "mature": "false",
    })
    data = json.loads(_get(API + "?" + params).decode())
    return data.get("results", []) or []


def _attribution(r: dict) -> dict:
    lic = r.get("license", "")
    ver = r.get("license_version", "")
    return {
        "title": r.get("title"),
        "creator": r.get("creator"),
        "license": f"CC {lic.upper()} {ver}".strip(),
        "license_url": r.get("license_url"),
        "source_page": r.get("foreign_landing_url") or r.get("url"),
        "provider": r.get("provider"),
    }


def _normalize(raw: bytes, out_path: Path, size: tuple[int, int]) -> bool:
    """Validate bytes are an image; center-crop to the target aspect and save PNG.
    Returns False if the bytes aren't a usable image."""
    try:
        from PIL import Image
    except Exception:
        out_path.write_bytes(raw)  # no PIL: save as-is, best effort
        return True
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return False
    tw, th = size
    w, h = img.size
    # cover-fit: scale so the image fills the frame, then center-crop
    scale = max(tw / w, th / h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    w, h = img.size
    left, top = (w - tw) // 2, (h - th) // 2
    img.crop((left, top, left + tw, top + th)).save(out_path)
    return True


def _broaden(query: str) -> list[str]:
    """Progressively shorter queries — long specific phrases return no stock hits,
    so fall back from the full query down to the last word + 'portrait'."""
    words = query.split()
    cands = [query]
    if len(words) > 2:
        cands.append(" ".join(words[-2:]))          # e.g. 'woman portrait'
    if "portrait" not in words[-1:]:
        cands.append(f"{words[-1]} portrait")
    cands.append("person portrait")
    seen, out = set(), []
    for q in cands:                                  # dedup, keep order
        if q and q not in seen:
            seen.add(q); out.append(q)
    return out


def fetch_actor(query: str, out_path: Path, size: tuple[int, int]) -> dict | None:
    """Find a CC portrait matching `query` (broadening if needed), save it (cropped
    to `size`) to out_path, and return its attribution. None on any failure."""
    if not enabled():
        return None
    results = []
    for q in _broaden(query):
        try:
            results = search(q, n=int(_cfg().get("candidates", 8)))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            _log(f"      stock search failed ({e}); will generate actor instead")
            return None
        except Exception as e:  # noqa: BLE001 — API shape / decode
            _log(f"      stock search error ({type(e).__name__}); generating actor instead")
            return None
        if results:
            break
    for r in results:
        url = r.get("url")
        if not url:
            continue
        try:
            raw = _get(url)
        except Exception:
            continue
        if _normalize(raw, out_path, size):
            return _attribution(r)
    return None
