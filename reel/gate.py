"""Human-in-the-loop review gate.

After each pipeline stage the gate shows a summary of the output and waits
for the operator to approve or supply feedback. If feedback is given the
calling stage re-runs with it appended to its prompt. Parallel stages present
for approval sequentially (one terminal, one interactive prompt at a time).
Typing 'view' opens the stage's FULL output (not just the summary) in
$EDITOR/$VISUAL (default vim) to read before deciding — the gate reprompts
once you close the editor.

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


@dataclass
class Decision:
    approved: bool
    feedback: str = field(default="")
    stop: bool = False


# Typing any of these at a gate pauses the pipeline (completed stages stay saved).
STOP_WORDS = {"stop", "pause", "quit", "q", "exit"}

# Typing any of these opens the full stage output in an editor to read (not just
# the (possibly truncated) summary), then reprompts the same gate.
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
                "'view' to read the full output in an editor, "
                "or 'stop' to pause:\n",
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
                self._view_in_editor(stage, result)
                continue  # redisplay the gate and reprompt

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

    def _view_in_editor(self, stage: str, result: dict) -> None:
        """Write the stage's full output to a temp file and open it in
        $EDITOR/$VISUAL (default vim) — read-only in spirit: any edits made are
        NOT read back or applied, this is purely for reading past what the
        (possibly truncated) gate summary shows. Never raises; a missing/broken
        editor just prints a hint and returns to the gate prompt."""
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vim"
        try:
            text = json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"  ⚠ could not serialize output for viewing: {exc}", flush=True)
            return

        fd, path = tempfile.mkstemp(prefix=f"reel_gate_{stage}_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)

        try:
            print(f"  opening {stage} output in {editor} (edits are not saved back) …",
                 flush=True)
            subprocess.call([editor, path])
        except OSError as exc:
            # Editor failed to launch (e.g. not on PATH) — leave the file in
            # place so the hint below still points somewhere real.
            print(f"  ⚠ could not open editor {editor!r}: {exc} "
                 f"— set $EDITOR/$VISUAL, or read it yourself: {path}", flush=True)
            return

        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

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
