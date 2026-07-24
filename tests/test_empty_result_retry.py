"""Validates the empty-result safety net: any stage that comes back with
nothing meaningful (a JSON-parse failure, or its defining content key
missing/empty) is automatically rerun ONCE — even when the empty result
came from what's already an iteration/retry, not just the first attempt —
and if that single retry is ALSO empty, the run fails via
`StageEmptyResultError` rather than let empty data propagate to every
downstream stage. Direct instruction: "if at any stage, the outcome is
empty rerun the stage (even if it is iteration)... if still empty fail the
pipeline run."

Covers `stages.is_stage_result_empty`, `pipeline._ensure_nonempty_result`,
`pipeline._gated`'s three call sites (initial compute, self-critique
refine, gate-loop feedback rerun), and `stages.run_stage`'s standalone
retry. Pure logic + mocked I/O — no live Ollama/Gemini/Veo calls anywhere.

Run with: python -m unittest tests.test_empty_result_retry -v
"""
from __future__ import annotations

import unittest
from unittest import mock

from reel import pipeline, stages
from reel.gate import Decision, Gate
from reel.stages import StageEmptyResultError, is_stage_result_empty


class TestIsStageResultEmpty(unittest.TestCase):
    def test_empty_dict_is_empty(self):
        self.assertTrue(is_stage_result_empty("scenes", {}))

    def test_none_is_empty(self):
        self.assertTrue(is_stage_result_empty("scenes", None))

    def test_non_dict_is_empty(self):
        self.assertTrue(is_stage_result_empty("scenes", ["not", "a", "dict"]))

    def test_parse_error_sentinel_is_empty_regardless_of_stage(self):
        self.assertTrue(is_stage_result_empty(
            "scenes", {"_raw": "garbage", "_parse_error": True}))

    def test_missing_content_key_is_empty(self):
        self.assertTrue(is_stage_result_empty("scenes", {"dropped_scenes": []}))

    def test_empty_list_under_content_key_is_empty(self):
        self.assertTrue(is_stage_result_empty("characters", {"characters": []}))

    def test_non_empty_content_key_is_not_empty(self):
        self.assertFalse(is_stage_result_empty(
            "characters", {"characters": [{"name": "Alice"}]}))

    def test_every_registered_content_stage_honors_its_own_key(self):
        cases = {
            "structure": ("three_act", {}),
            "moodboard": ("overall_aesthetic", ""),
            "characters": ("characters", []),
            "scenes": ("scenes", []),
            "casting": ("casting", []),
            "soundscape": ("soundscapes", []),
            "visuals": ("scenes", []),
            "cinematography": ("scenes", []),
            "screenplay": ("scenes", []),
            "storyboard": ("storyboard", []),
        }
        for name, (key, empty_val) in cases.items():
            with self.subTest(name=name):
                self.assertTrue(is_stage_result_empty(name, {key: empty_val}))
                filled = {"structure": {"one": "beat"}}.get(name, [{"x": 1}]) \
                    if key != "overall_aesthetic" else "a real vision statement"
                self.assertFalse(is_stage_result_empty(name, {key: filled}))

    def test_unregistered_stage_name_skips_the_specific_key_check(self):
        # A stage outside `_STAGE_CONTENT_KEYS` (render/grader stages, or a
        # synthetic test-only name) has no defining key to check against —
        # this function is deliberately scoped to creative content
        # generation only, so ANY non-empty dict passes for it, even one
        # that would fail a real stage's key requirement.
        self.assertFalse(is_stage_result_empty("teststage", {"result": "initial"}))
        self.assertFalse(is_stage_result_empty("fidelity", {"fidelity_score": 90}))

    def test_unregistered_stage_name_still_fails_the_universal_empty_dict_check(self):
        # The "is this dict empty/parse-failed at all" check applies
        # unconditionally, regardless of whether the stage has a specific
        # content key registered — only the KEY-specific check is skipped.
        self.assertTrue(is_stage_result_empty("teststage", {}))
        self.assertTrue(is_stage_result_empty("fidelity", None))


class TestEnsureNonemptyResult(unittest.TestCase):
    def test_non_empty_result_returned_unchanged_rerun_never_called(self):
        rerun_calls = []
        def rerun(fb, p=None):
            rerun_calls.append(fb)
            return {"scenes": [{"number": 99}]}

        result = pipeline._ensure_nonempty_result(
            "scenes", {"scenes": [{"number": 1}]}, rerun, "fast")
        self.assertEqual(result, {"scenes": [{"number": 1}]})
        self.assertEqual(rerun_calls, [])

    def test_empty_result_retries_once_and_succeeds(self):
        rerun_calls = []
        def rerun(fb, p=None):
            rerun_calls.append((fb, p))
            return {"scenes": [{"number": 1}]}

        result = pipeline._ensure_nonempty_result("scenes", {"scenes": []}, rerun, "fast")
        self.assertEqual(result, {"scenes": [{"number": 1}]})
        self.assertEqual(len(rerun_calls), 1)
        self.assertEqual(rerun_calls[0][1], "fast")
        self.assertIn("empty", rerun_calls[0][0].lower())

    def test_empty_result_retry_still_empty_raises(self):
        def rerun(fb, p=None):
            return {"scenes": []}

        with self.assertRaises(StageEmptyResultError) as ctx:
            pipeline._ensure_nonempty_result("scenes", {"scenes": []}, rerun, "fast")
        self.assertEqual(ctx.exception.stage, "scenes")

    def test_retry_raising_an_exception_also_raises_stage_empty_result_error(self):
        def rerun(fb, p=None):
            raise RuntimeError("model connection failed")

        with self.assertRaises(StageEmptyResultError):
            pipeline._ensure_nonempty_result("scenes", {}, rerun, "fast")


class TestGatedEmptyResultOnInitialCompute(unittest.TestCase):
    """The most common case: the very first compute() for a stage comes
    back empty, before critique/fidelity/the gate ever see it."""

    def setUp(self):
        self.gate = Gate(enabled=False)   # auto-approves — isolates the retry logic
        self.summarize = lambda r: f"summary: {r}"

    def test_empty_initial_result_retried_once_then_gate_sees_the_retry(self):
        rerun_calls = []
        def rerun(fb, p=None):
            rerun_calls.append(fb)
            return {"scenes": [{"number": 1}]}

        result, _, _ = pipeline._gated(
            self.gate, "scenes", {"scenes": []}, self.summarize, rerun,
            agent_module=None, critique_enabled=False)

        self.assertEqual(len(rerun_calls), 1)
        self.assertEqual(result, {"scenes": [{"number": 1}]})

    def test_empty_initial_result_still_empty_after_retry_fails_the_run(self):
        def rerun(fb, p=None):
            return {"scenes": []}

        with self.assertRaises(StageEmptyResultError):
            pipeline._gated(
                self.gate, "scenes", {"scenes": []}, self.summarize, rerun,
                agent_module=None, critique_enabled=False)

    def test_non_empty_initial_result_never_triggers_a_retry(self):
        rerun_calls = []
        def rerun(fb, p=None):
            rerun_calls.append(fb)
            return {"scenes": [{"number": 99}]}

        result, _, _ = pipeline._gated(
            self.gate, "scenes", {"scenes": [{"number": 1}]}, self.summarize, rerun,
            agent_module=None, critique_enabled=False)

        self.assertEqual(rerun_calls, [])
        self.assertEqual(result, {"scenes": [{"number": 1}]})


class TestGatedEmptyResultFiresOnLaterIteration(unittest.TestCase):
    """Direct requirement: the safety net must fire even when the empty
    result came from what's already an iteration/retry (a human-feedback
    gate-loop rerun), not just the very first attempt."""

    def test_a_later_gate_loop_rerun_coming_back_empty_is_itself_retried(self):
        # Sequence: initial result is non-empty (gate shows it) -> operator
        # gives feedback -> the feedback-driven rerun comes back EMPTY ->
        # the safety net's own one-shot retry kicks in and succeeds ->
        # gate shows the recovered result -> operator approves.
        decisions = [
            Decision(approved=False, feedback="make it better"),  # 1st gate: reject w/ feedback
            Decision(approved=True),                              # 2nd gate: approve the recovered result
        ]
        gate = Gate(enabled=True)
        rerun_sequence = [
            {"scenes": []},                    # the gate-loop's feedback rerun: EMPTY
            {"scenes": [{"number": 1}]},        # the safety net's own retry: recovers
        ]
        rerun_calls = []
        def rerun(fb, p=None):
            rerun_calls.append(fb)
            return rerun_sequence.pop(0)

        with mock.patch.object(Gate, "review", side_effect=decisions):
            result, _, _ = pipeline._gated(
                gate, "scenes", {"scenes": [{"number": 0}]},
                lambda r: f"summary: {r}", rerun,
                agent_module=None, critique_enabled=False)

        self.assertEqual(result, {"scenes": [{"number": 1}]})
        self.assertEqual(len(rerun_calls), 2)

    def test_a_later_gate_loop_rerun_still_empty_after_its_retry_fails_the_run(self):
        decisions = [Decision(approved=False, feedback="make it better")]
        gate = Gate(enabled=True)
        def rerun(fb, p=None):
            return {"scenes": []}   # every call comes back empty

        with mock.patch.object(Gate, "review", side_effect=decisions):
            with self.assertRaises(StageEmptyResultError):
                pipeline._gated(
                    gate, "scenes", {"scenes": [{"number": 0}]},
                    lambda r: f"summary: {r}", rerun,
                    agent_module=None, critique_enabled=False)


class TestRunStageEmptyResultRetry(unittest.TestCase):
    """`run_stage`'s own copy of the same safety net, for the standalone
    `stage NAME` CLI path (which doesn't go through `_gated` at all)."""

    def test_empty_result_retried_once_then_saved(self):
        calls = {"n": 0}
        def fake_run(ctx, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"characters": []}
            return {"characters": [{"name": "Alice"}]}

        fake_stage = stages.Stage("characters", ["source"], fake_run)
        with mock.patch.dict(stages.REGISTRY, {"characters": fake_stage}), \
             mock.patch.object(stages, "_resolve_input", return_value={"text": "a story"}), \
             mock.patch.object(stages, "_save_artifact") as mock_save, \
             mock.patch("reel.gemini.set_log_dir"), \
             mock.patch("reel.session.start"):
            result = stages.run_stage("characters", out="/tmp/whatever", input_path="story.txt")

        self.assertEqual(calls["n"], 2)
        self.assertEqual(result, {"characters": [{"name": "Alice"}]})
        mock_save.assert_called_once()

    def test_empty_result_still_empty_after_retry_raises(self):
        def fake_run(ctx, **kw):
            return {"characters": []}

        fake_stage = stages.Stage("characters", ["source"], fake_run)
        with mock.patch.dict(stages.REGISTRY, {"characters": fake_stage}), \
             mock.patch.object(stages, "_resolve_input", return_value={"text": "a story"}), \
             mock.patch.object(stages, "_save_artifact") as mock_save, \
             mock.patch("reel.gemini.set_log_dir"), \
             mock.patch("reel.session.start"):
            with self.assertRaises(StageEmptyResultError):
                stages.run_stage("characters", out="/tmp/whatever", input_path="story.txt")
        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
