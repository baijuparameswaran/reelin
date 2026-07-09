"""Session identity for one story-to-video pipeline run.

A "session" groups everything a single story produces — ingest through the
final rendered movie — under one stable id, so `output/logs/gemini_api.log`
and the per-scene Veo prompt logs (`output/logs/scene_NN_veo_prompts.txt`)
stay attributable to a run even when the same `--out` directory is reused
across `--resume`d continuations or repeated standalone `stage`/`render`
invocations. The id is minted once per fresh `pipeline.run()` and persisted
to `<out>/session.json`; every later call against the same `--out` (resume,
or a standalone `stage`/`render` invocation) reattaches to it instead of
minting a new one.

No dependency on any other `reel` module, so it's safe to import from
`gemini.py`/`pipeline.py`/`stages.py` without risk of a cycle.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

_FILE = "session.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _path(out: Path | str) -> Path:
    return Path(out) / _FILE


def start(out: Path | str, *, source: str | None = None, fresh: bool = False) -> str:
    """Return the session id for `out`, creating or reattaching to one.

    `fresh=True` (a non-resumed `pipeline.run()`) always mints a NEW session,
    overwriting any prior `session.json` — matches the existing fresh-run
    semantics elsewhere in the pipeline (checkpoints in `out` get overwritten
    too). Otherwise (resume, or any standalone `stage`/`render` invocation)
    reattach to the existing session if `out` already has one; only mint a
    new one if it doesn't (e.g. an older output dir, or the very first stage
    of a brand-new run started via a standalone `stage` command).
    """
    out = Path(out)
    p = _path(out)
    if not fresh and p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data.setdefault("resumes", []).append(_now())
            data["status"] = "running"
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data["session_id"]
        except Exception:
            pass  # corrupt/unreadable session.json — fall through and mint a new one
    sid = _new_id()
    data = {
        "session_id": sid,
        "source": source,
        "started_at": _now(),
        "status": "running",
        "resumes": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return sid


def finish(out: Path | str, status: str = "complete") -> None:
    """Mark the current session's terminal status: 'complete' | 'paused' | 'failed'."""
    p = _path(out)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data["status"] = status
        data["ended_at"] = _now()
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def current(out: Path | str) -> str | None:
    """Best-effort read of the session id already active for `out`, or None."""
    p = _path(out)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("session_id")
    except Exception:
        return None
