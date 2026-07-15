"""Validates `spend.py` — estimating real-money Gemini/Veo API spend from
`gemini_api.log`, the persistent record `gemini._log_call` already writes
for every actual call. Nothing previously aggregated that log into an
actual dollar figure, despite the pipeline's own prompting explicitly
steering toward fewer scenes specifically to control the cost those calls
represent (`scenes.py`'s "MINIMIZE SCENE COUNT" rule).

Pure logic — no live Gemini/Veo API calls, no network access (pricing is a
hardcoded snapshot, not fetched at runtime).

Run with: python -m unittest tests.test_spend -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reel import spend


def _line(*, ts="2026-07-14T10:00:00+00:00", session="abc", kind="IMAGE",
         model="gemini-3.1-flash-image", backend="sdk", outcome="success",
         path=None, params=None) -> str:
    bits = [ts, f"session={session}", kind, f"model={model}", f"backend={backend}",
           f"outcome={outcome}"]
    if path is not None:
        bits.append(f"path={path}")
    if params is not None:
        bits.append("params=" + json.dumps(params))
    return "  ".join(bits)


class TestParseLogLines(unittest.TestCase):
    def test_parses_a_well_formed_line(self):
        line = _line(params={"prompt": "hi", "duration_seconds": 6})
        entries = spend.parse_log_lines(line)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["kind"], "IMAGE")
        self.assertEqual(e["model"], "gemini-3.1-flash-image")
        self.assertEqual(e["outcome"], "success")
        self.assertEqual(e["params"]["duration_seconds"], 6)

    def test_outcome_with_embedded_single_space_parsed_in_full(self):
        line = _line(kind="VIDEO", outcome="error(TimeoutError, code=8)")
        entries = spend.parse_log_lines(line)
        self.assertEqual(entries[0]["outcome"], "error(TimeoutError, code=8)")

    def test_params_json_with_embedded_double_space_parsed_correctly(self):
        # A prompt with a literal double-space must not truncate the JSON
        # parse — this is exactly why raw_decode is used instead of a
        # double-space-delimited regex.
        line = _line(params={"prompt": "a  scene with  double spaces", "duration_seconds": 6})
        entries = spend.parse_log_lines(line)
        self.assertEqual(entries[0]["params"]["prompt"], "a  scene with  double spaces")
        self.assertEqual(entries[0]["params"]["duration_seconds"], 6)

    def test_malformed_line_skipped_not_raised(self):
        text = "garbage\n" + _line() + "\n\n"
        entries = spend.parse_log_lines(text)
        self.assertEqual(len(entries), 1)

    def test_line_with_no_params_gets_empty_dict(self):
        line = _line(params=None)
        entries = spend.parse_log_lines(line)
        self.assertEqual(entries[0]["params"], {})


class TestEntryCost(unittest.TestCase):
    def test_image_success_priced(self):
        entry = {"kind": "IMAGE", "model": "gemini-3.1-flash-image", "outcome": "success", "params": {}}
        cost, _ = spend._entry_cost(entry)
        self.assertAlmostEqual(cost, 0.045)

    def test_video_success_uses_requested_duration(self):
        entry = {"kind": "VIDEO", "model": "veo-3.1-fast-generate-preview",
                 "outcome": "success", "params": {"duration_seconds": 6}}
        cost, _ = spend._entry_cost(entry)
        self.assertAlmostEqual(cost, 0.10 * 6)

    def test_video_success_no_duration_falls_back_to_veo_default_8s(self):
        entry = {"kind": "VIDEO", "model": "veo-3.1-fast-generate-preview",
                 "outcome": "success", "params": {}}
        cost, _ = spend._entry_cost(entry)
        self.assertAlmostEqual(cost, 0.10 * 8)

    def test_non_success_outcome_never_priced(self):
        entry = {"kind": "IMAGE", "model": "gemini-3.1-flash-image",
                 "outcome": "skipped(no-render)", "params": {}}
        cost, reason = spend._entry_cost(entry)
        self.assertIsNone(cost)

    def test_unrecognized_model_returns_none_not_a_guess(self):
        entry = {"kind": "IMAGE", "model": "some-new-model", "outcome": "success", "params": {}}
        cost, reason = spend._entry_cost(entry)
        self.assertIsNone(cost)
        self.assertIn("no price on file", reason)

    def test_video_extend_and_refs_kinds_also_priced(self):
        for kind in ("VIDEO_EXTEND", "VIDEO_REFS"):
            entry = {"kind": kind, "model": "veo-3.1-generate-preview",
                     "outcome": "success", "params": {"duration_seconds": 8}}
            cost, _ = spend._entry_cost(entry)
            self.assertAlmostEqual(cost, 0.40 * 8, msg=kind)


class TestEstimateCost(unittest.TestCase):
    def test_aggregates_across_multiple_entries(self):
        entries = [
            {"kind": "IMAGE", "model": "gemini-3.1-flash-image", "outcome": "success", "params": {}},
            {"kind": "IMAGE", "model": "gemini-3.1-flash-image", "outcome": "success", "params": {}},
            {"kind": "VIDEO", "model": "veo-3.1-fast-generate-preview", "outcome": "success",
             "params": {"duration_seconds": 8}},
        ]
        result = spend.estimate_cost(entries)
        self.assertAlmostEqual(result["total_usd"], 0.045 * 2 + 0.10 * 8)
        self.assertEqual(result["priced_calls"], 3)
        self.assertEqual(result["unpriced_calls"], 0)

    def test_unpriced_calls_excluded_from_total_and_tracked_separately(self):
        entries = [
            {"kind": "IMAGE", "model": "gemini-3.1-flash-image", "outcome": "success", "params": {}},
            {"kind": "IMAGE", "model": "mystery-model", "outcome": "success", "params": {}},
        ]
        result = spend.estimate_cost(entries)
        self.assertAlmostEqual(result["total_usd"], 0.045)
        self.assertEqual(result["unpriced_calls"], 1)
        self.assertEqual(result["unpriced_models"], ["mystery-model"])

    def test_empty_entries_yields_all_zero(self):
        result = spend.estimate_cost([])
        self.assertEqual(result["total_usd"], 0.0)
        self.assertEqual(result["priced_calls"], 0)
        self.assertEqual(result["unpriced_calls"], 0)

    def test_by_model_breakdown(self):
        entries = [
            {"kind": "IMAGE", "model": "gemini-3.1-flash-image", "outcome": "success", "params": {}},
            {"kind": "VIDEO", "model": "veo-3.1-fast-generate-preview", "outcome": "success",
             "params": {"duration_seconds": 4}},
        ]
        result = spend.estimate_cost(entries)
        self.assertIn("gemini-3.1-flash-image", result["by_model"])
        self.assertIn("veo-3.1-fast-generate-preview", result["by_model"])
        self.assertEqual(result["by_model"]["gemini-3.1-flash-image"]["calls"], 1)


class TestEstimateRunCost(unittest.TestCase):
    def test_missing_log_returns_all_zero_not_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            result = spend.estimate_run_cost(Path(td))
            self.assertEqual(result["total_usd"], 0.0)
            self.assertEqual(result["priced_calls"], 0)

    def test_reads_real_log_file_from_disk(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            logs = out / "logs"
            logs.mkdir()
            (logs / "gemini_api.log").write_text(
                _line(params={}) + "\n", encoding="utf-8")
            result = spend.estimate_run_cost(out)
            self.assertEqual(result["priced_calls"], 1)


class TestFormatSummary(unittest.TestCase):
    def test_zero_result_reads_as_no_billed_calls(self):
        result = spend.estimate_cost([])
        summary = spend.format_summary(result)
        self.assertEqual(summary, "no billed Gemini/Veo API calls yet")

    def test_nonzero_result_includes_dollar_figure(self):
        entries = [{"kind": "IMAGE", "model": "gemini-3.1-flash-image",
                   "outcome": "success", "params": {}}]
        summary = spend.format_summary(spend.estimate_cost(entries))
        self.assertIn("$0.0450", summary)

    def test_unpriced_calls_mentioned_in_summary(self):
        entries = [{"kind": "IMAGE", "model": "mystery-model", "outcome": "success", "params": {}}]
        summary = spend.format_summary(spend.estimate_cost(entries))
        self.assertIn("mystery-model", summary)
        self.assertIn("not priced", summary)


if __name__ == "__main__":
    unittest.main()
