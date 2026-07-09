"""Human-in-the-loop review gate.

After each pipeline stage the gate shows a summary of the output and waits
for the operator to approve or supply feedback. If feedback is given the
calling stage re-runs with it appended to its prompt. Parallel stages present
for approval sequentially (one terminal, one interactive prompt at a time).

Typing 'view' opens the stage's FULL output (not just the summary) in
$EDITOR/$VISUAL (default vim) — read it, or edit it and save. Closing the
editor unchanged just reprompts the same gate with the same result. Saving a
real, valid change applies it as a new candidate, which the caller
(`pipeline._gated`) re-checks against fidelity/genre before showing this same
gate again (approve / feedback / 'view' again / 'stop') — an edit is never
auto-approved on its own.

The auto-approve timeout only ever counts down while waiting for your first
keystroke at the prompt; it is not running at all while you're inside the
editor (see `_read`), and restarts fresh once the gate redisplays afterward.

Config knobs in config/models.yaml under `hitl`:
  enabled         — false skips all gates (fully automated)
  timeout_seconds — auto-approve after N seconds idle (0 = wait forever)

Non-interactive runs (cron, CI, the model-update smoke test): with no TTY,
`input()` raises EOFError, which the gate treats as approval — so an enabled
gate never hangs a headless run; it passes straight through.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_BAR = "─" * 64


def edit_in_editor(label: str, data: dict) -> dict | None:
    """Open `data` (any stage's full JSON output, or e.g. a raw source-text
    wrapper) in $EDITOR/$VISUAL — for reading, or for real editing; the same
    file serves both. `label` is used only for the temp-file prefix and log
    messages (e.g. a stage name, or "source").

    Returns the parsed edited dict when the operator saved a real, valid
    change; None in every other case (no change made — including a pure read
    with no save — editor failed to launch, or the saved content isn't valid
    JSON) — in all of those the original `data` is left completely untouched
    and an explanatory message is printed instead of raising, so the caller
    (the live gate, or the standalone `revise` CLI loop) can simply reprompt.

    Module-level (not a `Gate` method) since it needs nothing from a live
    gate's state — both `Gate._edit_in_editor` (the per-run HITL loop) and
    the standalone `revise` command call this directly."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vim"
    try:
        original_text = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"  ⚠ could not serialize output for editing: {exc}", flush=True)
        return None

    fd, path = tempfile.mkstemp(prefix=f"reel_gate_{label}_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(original_text)

    try:
        print(f"  opening {label} output in {editor} for editing "
             "(save + quit to apply; quit without saving to cancel) …",
             flush=True)
        subprocess.call([editor, path])
    except OSError as exc:
        print(f"  ⚠ could not open editor {editor!r}: {exc} "
             f"— set $EDITOR/$VISUAL, or edit it yourself: {path}", flush=True)
        return None

    try:
        edited_text = Path(path).read_text(encoding="utf-8")
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    if edited_text.strip() == original_text.strip():
        print("  no changes made — output unchanged\n", flush=True)
        return None

    try:
        edited = json.loads(edited_text)
    except Exception as exc:
        print(f"  ⚠ edited content is not valid JSON ({exc}) — "
             "change discarded, original output kept\n", flush=True)
        return None

    if not isinstance(edited, dict):
        print("  ⚠ edited content must be a JSON object — "
             "change discarded, original output kept\n", flush=True)
        return None

    return edited


@dataclass
class Decision:
    approved: bool
    feedback: str = field(default="")
    stop: bool = False
    edited: dict | None = None
    # ^ set when the operator made a real, valid edit in-editor. The caller
    # (pipeline._gated) should replace its working result with this and loop
    # back through fidelity/genre re-checking + the gate again — an edit is
    # a new candidate, not an approval.


# Typing any of these at a gate pauses the pipeline (completed stages stay saved).
STOP_WORDS = {"stop", "pause", "quit", "q", "exit"}

# Typing any of these opens the full stage output in $EDITOR — read it, or edit
# and save it. Unchanged/invalid content just reprompts the same gate; a real,
# valid edit becomes a new candidate result (see Decision.edited above).
VIEW_WORDS = {"view", "v", "edit", "vim"}


class Gate:
    def __init__(self, enabled: bool = True, timeout_seconds: int = 120):
        self.enabled = enabled
        self.timeout = timeout_seconds

    @classmethod
    def from_config(cls, cfg: dict) -> "Gate":
        h = cfg.get("hitl", {})
        return cls(
            enabled=h.get("enabled", True),
            timeout_seconds=h.get("timeout_seconds", 120),
        )

    def review(self, stage: str, result: dict, summarize) -> Decision:
        """Show summary, collect approval or feedback. Returns a Decision."""
        if not self.enabled:
            return Decision(approved=True)

        while True:
            print(f"\n{_BAR}", flush=True)
            print(f"[gate] {stage}", flush=True)
            print(_BAR, flush=True)
            try:
                print(summarize(result), flush=True)
            except Exception as exc:
                print(f"  (summary error: {exc})", flush=True)
            print(_BAR, flush=True)

            timeout_hint = (
                f"  auto-approve in {self.timeout}s — " if self.timeout > 0 else "  "
            )
            print(
                f"\n{timeout_hint}press Enter to approve, type feedback, "
                "'view' to open the full output in an editor (read it, or edit "
                "and save to submit a new candidate), or 'stop' to pause:\n",
                flush=True,
            )

            first = self._read("> ")
            if first is None:
                print(f"[gate] {stage} — auto-approved (timeout)\n", flush=True)
                return Decision(approved=True)

            first = first.strip()
            if not first or first.lower() in ("a", "approve", "y", "yes", "ok", ""):
                print(f"[gate] {stage} — approved\n", flush=True)
                return Decision(approved=True)

            if first.lower() in STOP_WORDS:
                print(f"[gate] {stage} — stopping (this stage not saved)\n", flush=True)
                return Decision(approved=False, stop=True)

            if first.lower() in VIEW_WORDS:
                edited = self._edit_in_editor(stage, result)
                if edited is None:
                    continue  # unchanged / parse failed / editor failed — reprompt as-is
                print(f"[gate] {stage} — edit applied, re-checking alignment …\n", flush=True)
                return Decision(approved=False, edited=edited)

            # Collect multi-line feedback: keep reading until blank line
            lines = [first]
            print("  (continue feedback; blank line to submit)\n", flush=True)
            while True:
                more = input("  > ").strip()
                if not more:
                    break
                lines.append(more)

            feedback = "\n".join(lines)
            print(f"[gate] {stage} — re-running with feedback …\n", flush=True)
            return Decision(approved=False, feedback=feedback)

    def _edit_in_editor(self, stage: str, result: dict) -> dict | None:
        return edit_in_editor(stage, result)

    def _read(self, prompt: str) -> str | None:
        """Read one line with optional SIGALRM timeout. Returns None on timeout."""
        if self.timeout <= 0:
            try:
                return input(prompt)
            except EOFError:
                return ""

        try:
            import signal

            timed_out: list[bool] = [False]

            def _handler(signum, frame):
                timed_out[0] = True
                raise TimeoutError()

            old = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.timeout)
            try:
                line = input(prompt)
                signal.alarm(0)
                return line
            except TimeoutError:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return None
            except EOFError:
                signal.alarm(0)
                return ""
            finally:
                signal.signal(signal.SIGALRM, old)
                signal.alarm(0)

        except (ImportError, AttributeError):
            # signal.SIGALRM unavailable (Windows non-WSL): fall back to no timeout
            try:
                return input(prompt)
            except EOFError:
                return ""
