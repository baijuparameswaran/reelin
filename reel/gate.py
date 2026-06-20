"""Human-in-the-loop review gate.

After each pipeline stage the gate shows a summary of the output and waits
for the operator to approve or supply feedback. If feedback is given the
calling stage re-runs with it appended to its prompt. Parallel stages present
for approval sequentially (one terminal, one interactive prompt at a time).

Config knobs in config/models.yaml under `hitl`:
  enabled         — false skips all gates (fully automated)
  timeout_seconds — auto-approve after N seconds idle (0 = wait forever)

Non-interactive runs (cron, CI, the model-update smoke test): with no TTY,
`input()` raises EOFError, which the gate treats as approval — so an enabled
gate never hangs a headless run; it passes straight through.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

_BAR = "─" * 64


@dataclass
class Decision:
    approved: bool
    feedback: str = field(default="")
    stop: bool = False


# Typing any of these at a gate pauses the pipeline (completed stages stay saved).
STOP_WORDS = {"stop", "pause", "quit", "q", "exit"}


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
