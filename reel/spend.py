"""Estimate real-money spend from `gemini_api.log` — the persistent record
`gemini.py`'s `_log_call` already writes for every actual Gemini/Veo API
call (see that module's docstring). The log already has everything needed
to reconstruct cost (model, outcome, and the full request `params` —
including `duration_seconds` for video); nothing aggregated it into an
actual dollar figure until now, despite this project's own prompting
explicitly steering toward FEWER scenes specifically to control the cost
those calls represent (`scenes.py`'s "MINIMIZE SCENE COUNT" rule) — a
run's operator had no way to see the dollar consequence of that tradeoff.

Pricing below is a snapshot (fetched live from ai.google.dev/gemini-api/docs/pricing,
2026-07-14) of each model's STANDARD tier at this project's configured
resolution (720p for video — see config/models.yaml's `video.resolution`
comment on why 720p is the recommended setting) — NOT an exact bill: real
billing can vary by resolution/tier/region, and this module has no visibility
into anything beyond what `gemini_api.log` already recorded. Treat every
figure here as an ESTIMATE for budgeting, not a reconciled invoice. A model
not in the price table below is counted but flagged `unpriced` rather than
silently omitted or guessed at.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# $ per image, STANDARD tier, at this project's typical (well under 1K/2K)
# render size — see this module's docstring for the pricing snapshot date
# and its "estimate, not an exact bill" caveat.
IMAGE_PRICE_USD: dict[str, float] = {
    "gemini-2.5-flash-image": 0.039,
    "gemini-3.1-flash-image": 0.045,
    "gemini-3.1-flash-lite-image": 0.0336,
    "gemini-3-pro-image": 0.134,
}

# $ per SECOND of generated video, STANDARD tier, 720p (this project's
# configured default resolution — see config/models.yaml's `video.resolution`).
VIDEO_PRICE_USD_PER_SECOND_720P: dict[str, float] = {
    "veo-3.1-generate-preview": 0.40,
    "veo-3.1-fast-generate-preview": 0.10,
    "veo-3.1-lite-generate-preview": 0.05,
}

# Header fields are joined by exactly TWO spaces (`_log_call`'s own
# `"  ".join(bits)`); `outcome` itself can legitimately contain a SINGLE
# embedded space (e.g. `error(TimeoutError, code=8)`), so this stops at the
# first double-space rather than the first space of any kind — a plain
# `\S+` would truncate an outcome like that at the comma.
_HEADER_RE = re.compile(
    r"^(?P<ts>\S+)\s{2}session=(?P<session>\S*)\s{2}(?P<kind>\S+)\s{2}"
    r"model=(?P<model>\S*)\s{2}backend=(?P<backend>\S*)\s{2}outcome=(?P<outcome>.+?)"
    r"(?:\s{2}|$)"
)
_JSON_DECODER = json.JSONDecoder()


def parse_log_lines(text: str) -> list[dict]:
    """Parse `gemini_api.log`'s lines (`gemini._log_call`'s exact format)
    into dicts. The header fields (timestamp/session/kind/model/backend/
    outcome) are extracted via `_HEADER_RE`; the trailing `params={...}`
    JSON blob (if present) is located by its `params=` marker and parsed
    with `json.JSONDecoder.raw_decode` rather than a regex — a regex
    matching "up to the next double-space" would incorrectly truncate a
    params JSON that happens to contain a literal double-space in a string
    value (e.g. a Veo prompt with two consecutive spaces in it), which
    `raw_decode` handles correctly since it parses by JSON syntax, not by
    the log's own field-delimiter convention. Lines that don't match the
    expected header shape (a stray blank line, manual editing, a future
    format change) are silently skipped rather than raising — this is a
    best-effort audit read of an append-only log, not a strict parser
    something else depends on for correctness."""
    entries = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if not m:
            continue
        entry = m.groupdict()
        entry["params"] = {}
        marker = "params="
        idx = line.find(marker)
        if idx != -1:
            try:
                parsed, _end = _JSON_DECODER.raw_decode(line, idx + len(marker))
                entry["params"] = parsed
            except (json.JSONDecodeError, ValueError):
                pass
        entries.append(entry)
    return entries


def _entry_cost(entry: dict) -> tuple[float | None, str]:
    """(estimated $ for this one call, reason) — `None` cost means "not
    counted" (a skipped/failed call spent no money) or "can't price it"
    (an unrecognized model — reason explains which)."""
    if entry.get("outcome") != "success":
        return None, "not billed (outcome != success — skipped or failed)"
    kind = entry.get("kind", "")
    model = entry.get("model", "")
    if kind == "IMAGE":
        price = IMAGE_PRICE_USD.get(model)
        if price is None:
            return None, f"no price on file for image model '{model}'"
        return price, "1 image"
    if kind in ("VIDEO", "VIDEO_EXTEND", "VIDEO_REFS"):
        rate = VIDEO_PRICE_USD_PER_SECOND_720P.get(model)
        if rate is None:
            return None, f"no price on file for video model '{model}'"
        params = entry.get("params") or {}
        seconds = params.get("duration_seconds")
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            seconds = 8  # Veo's own SDK default when no duration was requested
        return rate * seconds, f"{seconds}s video"
    return None, f"unrecognized call kind '{kind}'"


def estimate_cost(entries: list[dict]) -> dict:
    """Aggregate parsed log entries into a spend summary:
    `total_usd`, `priced_calls`, `unpriced_calls` (successful calls with no
    matching price entry — NOT included in `total_usd`, since guessing
    would be worse than omitting), `by_model` (per-model $ + call count),
    and `unpriced_models` (the distinct model names that couldn't be
    priced, for the caller to report clearly rather than silently
    understating spend)."""
    total = 0.0
    priced_calls = 0
    unpriced_calls = 0
    unpriced_models: set[str] = set()
    by_model: dict[str, dict] = {}

    for entry in entries:
        cost, _reason = _entry_cost(entry)
        if cost is None:
            if entry.get("outcome") == "success":
                unpriced_calls += 1
                unpriced_models.add(entry.get("model", "") or "(unknown model)")
            continue
        total += cost
        priced_calls += 1
        model = entry.get("model", "") or "(unknown model)"
        bucket = by_model.setdefault(model, {"usd": 0.0, "calls": 0})
        bucket["usd"] += cost
        bucket["calls"] += 1

    return {
        "total_usd": round(total, 4),
        "priced_calls": priced_calls,
        "unpriced_calls": unpriced_calls,
        "unpriced_models": sorted(unpriced_models),
        "by_model": {m: {"usd": round(v["usd"], 4), "calls": v["calls"]}
                    for m, v in by_model.items()},
    }


def estimate_run_cost(out: Path | str) -> dict:
    """Read `<out>/logs/gemini_api.log` (if present) and return its spend
    estimate — `{"total_usd": 0.0, "priced_calls": 0, ...}` (all zeros,
    never an error) when the log doesn't exist yet, e.g. a run that hasn't
    made any real API calls (a `--no-render` run, or one still on its
    text-only stages)."""
    log_path = Path(out) / "logs" / "gemini_api.log"
    if not log_path.exists():
        return estimate_cost([])
    text = log_path.read_text(encoding="utf-8")
    return estimate_cost(parse_log_lines(text))


def format_summary(result: dict) -> str:
    """One-line human-readable summary, e.g.:
    '~$1.2340 (12 call(s), Gemini/Veo) — 2 call(s) unpriced (model X)'
    An all-zero result (nothing billed yet) reads as a plain "no billed API
    calls yet" line rather than "~$0.0000", which would misleadingly imply
    a precise measurement of nothing."""
    if result["priced_calls"] == 0 and result["unpriced_calls"] == 0:
        return "no billed Gemini/Veo API calls yet"
    bits = [f"~${result['total_usd']:.4f} estimated spend "
           f"({result['priced_calls']} priced call(s))"]
    if result["unpriced_calls"]:
        models = ", ".join(result["unpriced_models"])
        bits.append(f"— {result['unpriced_calls']} call(s) not priced (model(s): {models})")
    return " ".join(bits)
