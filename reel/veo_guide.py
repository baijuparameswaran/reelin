"""Veo prompt guide cache — snapshot, staleness check, and prompt verifier.

Caches the Veo prompting guide from ai.google.dev so the verifier can check
prompts against the authoritative vocabulary without a network round-trip on
every video generation call. When content changes are detected the operator is
shown exactly which source files to review.

URL discovery: the primary URL is tried first; if it redirects or returns a
non-guide page, a ranked list of alternates is tried in order. The working URL
is saved in the snapshot so the next fetch goes straight there.

Usage (CLI):
  python -m reel.cli veo-sync           # manual sync (always fetches)
  python -m reel.cli veo-sync --status  # print cache status only

Automatic refresh is intentionally disabled (was noise on every pipeline run).
Run veo-sync manually when preparing a release or after a long gap.

verify_prompt(prompt) — call before submitting any Veo API request; returns
a report dict with issues (blocking), warnings (advisory), and valid flag.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Cache lives alongside the project config so it travels with the repo.
_CACHE_FILE = Path(__file__).parent.parent / "config" / "veo_guide_snapshot.json"

STALE_DAYS = 14

# Candidate URLs in priority order.  The guide path has changed before; if the
# primary 404s or returns a non-Veo page, the next URL is tried.
_CANDIDATE_URLS: list[str] = [
    "https://ai.google.dev/gemini-api/docs/veo",
    "https://ai.google.dev/gemini-api/docs/video",
    "https://ai.google.dev/gemini-api/docs/generate-video",
    "https://ai.google.dev/api/generate-video",
]

_UA = "Mozilla/5.0 (compatible; reel-veo-sync/1.0; +https://github.com/reel)"
_FETCH_TIMEOUT = 30.0

# Vocabulary the guide defines — used both to validate that we fetched the right
# page and to detect when the guide's recommended terminology changes.
# Terms are matched against the page text after Unicode normalization (smart quotes
# → straight quotes) so apostrophe encoding differences don't cause false misses.
_EXPECTED_TERMS: dict[str, list[str]] = {
    "style_keywords": [
        "cinematic", "film noir", "stop-motion", "3d animated",
        "cartoon", "photorealistic", "sci-fi", "horror film",
    ],
    "focus_types": [
        "shallow focus", "deep focus", "soft focus", "macro lens", "wide-angle lens",
    ],
    "camera_movements": [
        "aerial view", "dolly shot", "tracking", "panning",
        "worms eye", "eye-level", "pov shot",
    ],
    "shot_types": [
        "wide shot", "close-up", "extreme close-up",
        "pov shot", "two-shot", "eye-level", "top-down",
    ],
    "audio_cues": [
        "dialogue", "sound effects", "ambient noise", "soundscape",
    ],
}

# Code locations that implement the guide — shown in the change notice.
_CODE_LOCATIONS = [
    ("reel/pipeline.py",           "_VEO_FOCUS dict — focus/lens terms by shot type"),
    ("reel/fountain.py",           "_VEO_FOCUS_FOUNTAIN dict — same for standalone render"),
    ("reel/agents/storyboard.py",  "image_prompt instructions — 5-element structure + focus hints"),
    ("config/models.yaml",         "video.style_suffix — Veo style keyword fallback"),
]


# ── cache I/O ─────────────────────────────────────────────────────────────────

def load() -> dict:
    """Load the cached snapshot. Returns {} when absent or unreadable."""
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8")) if _CACHE_FILE.exists() else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    except Exception as e:
        _log(f"⚠ could not write snapshot: {e}")


def _log(msg: str) -> None:
    print(f"[reel/veo-guide] {msg}", flush=True)


# ── staleness ─────────────────────────────────────────────────────────────────

def is_stale(cache: dict | None = None) -> bool:
    """True when the cache is absent or older than STALE_DAYS."""
    c = cache if cache is not None else load()
    ts = c.get("fetched_at")
    if not ts:
        return True
    try:
        fetched = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).days >= STALE_DAYS
    except Exception:
        return True


def age_days(cache: dict | None = None) -> int | None:
    """Days since last fetch, or None if the cache is absent."""
    c = cache if cache is not None else load()
    ts = c.get("fetched_at")
    if not ts:
        return None
    try:
        fetched = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).days
    except Exception:
        return None


# ── network ───────────────────────────────────────────────────────────────────

def _fetch(url: str) -> tuple[str, str]:
    """GET `url`, following redirects. Returns (final_url, body_text).
    Raises urllib.error.URLError / urllib.error.HTTPError on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
        return resp.url, resp.read().decode("utf-8", errors="replace")


def _is_veo_guide(content: str) -> bool:
    """Heuristic: does this page look like the Veo prompting guide?"""
    norm = _normalize(content)
    return ("veo" in norm and "prompt" in norm
            and any(t in norm for t in ("subject", "action", "style", "cinematic")))


def _discover() -> tuple[str, str] | None:
    """Try candidate URLs in order; return (final_url, content) for the first
    that looks like the Veo guide, following HTTP redirects automatically."""
    # Also try the URL saved in the previous cache (it may be more current).
    cached_url = load().get("url")
    urls = ([cached_url] if cached_url and cached_url not in _CANDIDATE_URLS else []) + _CANDIDATE_URLS
    for url in urls:
        try:
            final_url, content = _fetch(url)
            if _is_veo_guide(content):
                return final_url, content
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            continue
    return None


# ── content parsing ───────────────────────────────────────────────────────────

def _section_hash(content: str) -> str:
    """SHA-256 of the section of the page most likely to be the prompt guide.
    Uses a ~3 kB window around the first 'subject' occurrence (the first of
    Veo's five prompt elements) so nav/date changes don't trigger false positives.
    Normalized before hashing so apostrophe/quote encoding doesn't matter."""
    norm = _normalize(content)
    idx = norm.find("subject")
    if idx < 0:
        idx = 0
    window = norm[max(0, idx - 200): idx + 3000]
    return hashlib.sha256(window.encode()).hexdigest()[:20]


def _normalize(text: str) -> str:
    """Normalize Unicode punctuation so smart quotes / dashes don't break matching."""
    return (text
            .replace("‘", "'").replace("’", "'")   # curly single quotes → straight
            .replace("“", '"').replace("”", '"')   # curly double quotes → straight
            .replace("–", "-").replace("—", "-")   # en/em dash → hyphen
            .lower())


def _extract_terms(content: str) -> dict[str, list[str]]:
    """Return which expected vocabulary terms are present in the fetched content."""
    norm = _normalize(content)
    found: dict[str, list[str]] = {}
    for category, terms in _EXPECTED_TERMS.items():
        found[category] = [t for t in terms if t in norm]
    return found


def _missing_terms(found: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return terms that are expected but absent in the fetched content."""
    missing: dict[str, list[str]] = {}
    for category, terms in _EXPECTED_TERMS.items():
        gone = [t for t in terms if t not in found.get(category, [])]
        if gone:
            missing[category] = gone
    return missing


# ── public sync API ───────────────────────────────────────────────────────────

def sync(*, force: bool = False, quiet: bool = False) -> dict:
    """Fetch + cache the guide if stale (or force=True). Returns the cache dict.

    Prints a change notice (with code-review checklist) when content shifts.
    Never raises — network errors are logged and the old cache is kept.
    """
    cache = load()
    if not force and not is_stale(cache):
        if not quiet:
            days = age_days(cache)
            remaining = STALE_DAYS - (days or 0)
            _log(f"up to date (fetched {days}d ago; next check in ~{remaining}d)")
        return cache

    _log("checking for Veo prompt guide updates …")
    result = _discover()
    if result is None:
        _log("⚠ could not reach any candidate URL — keeping existing snapshot")
        return cache

    url, content = result
    new_hash = _section_hash(content)
    old_hash = cache.get("section_hash", "")
    found_terms = _extract_terms(content)
    missing = _missing_terms(found_terms)

    new_cache: dict = {
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "url": url,
        "section_hash": new_hash,
        "terms_found": found_terms,
        "terms_missing": missing,
    }
    _save(new_cache)

    changed = new_hash != old_hash and old_hash != ""

    if changed:
        _log(f"⚠  GUIDE CONTENT CHANGED  (hash {old_hash} → {new_hash})")
        _log(f"   source: {url}")
        if missing:
            for cat, terms in missing.items():
                _log(f"   terms no longer found [{cat}]: {', '.join(terms)}")
        _log("   Review these files against the updated guide:")
        for path, desc in _CODE_LOCATIONS:
            _log(f"     {path}  —  {desc}")
        _log("   Run `python -m reel.cli veo-sync` again after updating.")
    elif old_hash:
        _log(f"guide unchanged (hash {new_hash}, url: {url})")
    else:
        _log(f"snapshot saved (hash {new_hash}, url: {url})")

    if missing and not quiet:
        _log("⚠  some expected terms not found in fetched content:")
        for cat, terms in missing.items():
            _log(f"   [{cat}] missing: {', '.join(terms)}")
        _log("   This may indicate a page-structure change or a fetch of the wrong page.")

    return new_cache



# ── Veo prompt verifier ───────────────────────────────────────────────────────

# Style keywords Veo understands — at least one must be present per guide.
_VEO_STYLE_KWS = frozenset([
    "cinematic", "film noir", "photorealistic", "sci-fi", "horror film",
    "stop-motion", "3d animated", "cartoon", "surreal", "vintage", "futuristic",
    "animation", "documentary", "animated",
])

# Composition/shot-type terms the guide lists.
_VEO_SHOT_KWS = frozenset([
    "wide shot", "close-up", "medium shot", "extreme close-up", "pov shot",
    "two-shot", "single-shot", "over-the-shoulder", "insert shot", "establishing",
    "aerial view", "top-down", "eye-level",
])

# Focus/lens terms the guide defines.
_VEO_FOCUS_KWS = frozenset([
    "shallow focus", "deep focus", "soft focus", "macro lens",
    "wide-angle lens", "portrait", "telephoto lens",
])


def verify_prompt(prompt: str) -> dict:
    """Check a Veo video prompt against the prompting guide.

    Returns a report dict:
      valid   — True when all required elements are present
      issues  — list of missing/malformed element descriptions
      warnings — list of advisory notes (non-blocking)
      prompt  — the original prompt (unchanged; caller decides whether to use it)

    This function is intentionally read-only and never raises. It is called
    by the Gemini Veo backend immediately before submitting to the API so
    operators see exactly what the guide flags without blocking generation.
    """
    norm = prompt.lower()
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Subject — must be something concrete (person, animal, object, scenery).
    #    Hard to detect reliably; skip automated check, flag if the prompt is very short.
    if len(prompt.split()) < 5:
        issues.append("Subject: prompt is too short to contain a meaningful subject description")

    # 2. Action — look for a verb indicating motion or activity.
    action_verbs = ("walk", "run", "stand", "move", "turn", "look", "enter", "exit",
                    "drive", "fly", "fall", "rise", "sit", "open", "close", "reach",
                    "hold", "carry", "speak", "say", "whisper", "shout", "gesture",
                    "step", "approach", "cross", "climb", "descend", "float", "swim")
    if not any(v in norm for v in action_verbs):
        warnings.append("Action: no clear action verb detected — add what the subject does")

    # 3. Style — at least one Veo style keyword required.
    if not any(kw in norm for kw in _VEO_STYLE_KWS):
        issues.append(
            f"Style: no Veo style keyword found — add one of: "
            + ", ".join(sorted(_VEO_STYLE_KWS)[:8])
        )

    # 4. Camera & Composition — at least one shot-type or camera-position term.
    has_shot = any(kw in norm for kw in _VEO_SHOT_KWS)
    has_cam_motion = any(t in norm for t in (
        "dolly", "tracking", "panning", "pan", "tilt", "handheld",
        "steadicam", "crane", "zoom", "static", "aerial",
    ))
    if not has_shot and not has_cam_motion:
        issues.append(
            "Camera & Composition: no shot type or camera movement found — "
            "add e.g. 'wide shot', 'close-up', 'dolly in', 'eye-level'"
        )

    # 5. Focus & Ambiance — advisory if neither focus nor lighting/color is present.
    has_focus = any(kw in norm for kw in _VEO_FOCUS_KWS)
    has_ambiance = any(t in norm for t in (
        "tones", "light", "shadow", "warm", "cool", "blue", "golden",
        "dark", "bright", "glow", "neon", "sunrise", "sunset", "night",
        "dusk", "dawn", "overcast", "muted",
    ))
    if not has_focus and not has_ambiance:
        warnings.append(
            "Focus & Ambiance: no focus term or lighting/color mood detected — "
            "consider adding e.g. 'shallow focus', 'warm tones', 'deep focus'"
        )

    # Audio — dialogue should use quotation marks per guide.
    if "says" in norm or "whisper" in norm or "shout" in norm or "murmur" in norm:
        if '"' not in prompt and "'" not in prompt:
            issues.append(
                "Audio/Dialogue: speech verbs present but no quoted dialogue — "
                "wrap spoken lines in quotation marks (Veo guide requirement)"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "prompt": prompt,
    }


def status() -> str:
    """One-line status string for display."""
    cache = load()
    if not cache:
        return "no snapshot — run `python -m reel.cli veo-sync` to fetch"
    ts = cache.get("fetched_at", "?")[:10]
    url = cache.get("url", "?")
    days = age_days(cache)
    stale_tag = f"  ⚠ STALE ({days}d — refresh with veo-sync)" if is_stale(cache) else f"  (fetched {days}d ago)"
    missing = cache.get("terms_missing") or {}
    warn = f"  ⚠ {sum(len(v) for v in missing.values())} term(s) missing" if missing else ""
    return f"{ts}  {url}{stale_tag}{warn}"
