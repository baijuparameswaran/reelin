"""Validates the CLI-level wiring for the two 2026-07-14 structural
additions: `spend` (an on-demand subcommand plus post-run printouts) and
`_print_config_warnings` (a startup sanity check for config/models.yaml).
Both are best-effort/non-blocking by design — neither should ever be able
to fail a command that would otherwise have succeeded.

Offline — no live Ollama/Gemini/Veo calls; `cli.main`'s config-warning
check reads the REAL config/models.yaml (expected to validate clean, same
guarantee `tests.test_config_validation` already covers directly).

Run with: python -m unittest tests.test_cli_spend_and_config_warnings -v
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import cli


def _write_log(out: Path, lines: list[str]) -> None:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "gemini_api.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCmdSpend(unittest.TestCase):
    def test_spend_subcommand_reports_estimated_total(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            _write_log(out, [
                "2026-07-14T10:00:00+00:00  session=abc  IMAGE  "
                "model=gemini-3.1-flash-image  backend=sdk  outcome=success  "
                "path=x.png  params=" + json.dumps({"prompt": "p"}),
            ])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["spend", "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertIn("0.0450", buf.getvalue())

    def test_spend_subcommand_with_no_log_reports_zero(self):
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["spend", "--out", td])
            self.assertEqual(rc, 0)
            self.assertIn("no billed", buf.getvalue())


class TestPrintSpendSummary(unittest.TestCase):
    def test_prints_nothing_when_no_calls_were_ever_made(self):
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._print_spend_summary(td)
            self.assertEqual(buf.getvalue(), "")

    def test_prints_a_line_when_calls_were_made(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            _write_log(out, [
                "2026-07-14T10:00:00+00:00  session=abc  VIDEO  "
                "model=veo-3.1-fast-generate-preview  backend=sdk  outcome=success  "
                "path=x.mp4  params=" + json.dumps({"duration_seconds": 8}),
            ])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._print_spend_summary(out)
            self.assertIn("estimated spend", buf.getvalue())

    def test_never_raises_even_if_estimate_run_cost_blows_up(self):
        with mock.patch("reel.spend.estimate_run_cost", side_effect=RuntimeError("boom")):
            cli._print_spend_summary("/nonexistent/path")  # must not raise


class TestPrintConfigWarnings(unittest.TestCase):
    def test_real_config_prints_nothing(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._print_config_warnings()
        self.assertEqual(buf.getvalue(), "")

    def test_prints_warnings_when_present(self):
        with mock.patch("reel.llm.validate_config", return_value=["unknown key 'foo'"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._print_config_warnings()
            self.assertIn("foo", buf.getvalue())

    def test_never_raises_even_if_validate_config_blows_up(self):
        with mock.patch("reel.llm.validate_config", side_effect=RuntimeError("boom")):
            cli._print_config_warnings()  # must not raise


if __name__ == "__main__":
    unittest.main()
