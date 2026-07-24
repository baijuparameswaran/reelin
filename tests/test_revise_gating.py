"""Validates `cli._gate_stage_result` — the mechanism that lets a `revise`
round show the SAME per-stage review gate a fresh `pipeline.run()` uses
(fidelity/genre score, feedback-driven re-run, view/edit-in-editor,
auto-escalation, auto-approve timeout) for each stage the revision cascade
regenerates, per direct request ("can the revise do gating like normal
pipeline run"). Reuses `pipeline._gated` (the exact same function a fresh
run's gate loop calls) rather than a second implementation.

`gate=None` (every pre-existing call site's default) must remain a complete
no-op — every test in this file that isn't specifically about the new
gating passes no `gate` at all, so this file is purely about the new,
opt-in behavior.

Pure logic checks with mocked I/O — no LLM/API calls (fidelity/genre config
is forced off via a mocked `llm.config()`, so `fidelity.check_stage`/
`genre.enforce_stage` are never actually reached), no real $EDITOR, no
Ollama. Run with:
    python -m unittest tests.test_revise_gating -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import cli
from reel.gate import Decision, Gate
from reel.pipeline import PipelineStopped

_NO_SCORING_CFG = {"fidelity": {"per_stage": False}, "genre": {"enforce": False},
                   "runtime": {}}


class TestGateNoneIsANoOp(unittest.TestCase):
    def test_gate_none_returns_result_unchanged(self):
        result = {"scenes": [{"number": 1}]}
        out = cli._gate_stage_result(
            None, "soundscape", result, out=Path("/tmp/whatever"), profile="fast",
            existing=None, revise_keys={1}, max_scenes=None, duration_kwargs={})
        self.assertIs(out, result)


class TestGateSkipsUnsummarizableStages(unittest.TestCase):
    def test_stage_with_no_summarizer_is_untouched(self):
        gate = Gate(enabled=True, timeout_seconds=0)
        result = {"casting": []}
        with mock.patch.object(Gate, "review") as mocked_review:
            out = cli._gate_stage_result(
                gate, "casting_images", result, out=Path("/tmp/whatever"), profile="fast",
                existing=None, revise_keys=None, max_scenes=None, duration_kwargs={})
        self.assertIs(out, result)
        mocked_review.assert_not_called()


class TestGateApprovalAndPersistence(unittest.TestCase):
    def _write(self, out: Path, name: str, data: dict) -> None:
        (out / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")

    def test_immediate_approval_saves_the_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._write(out, "soundscape", {"soundscapes": [{"scene_number": 1}]})
            gate = Gate(enabled=True, timeout_seconds=0)
            result = {"soundscapes": [{"scene_number": 1, "ambient_bed": "wind"}]}

            with mock.patch.object(Gate, "review", return_value=Decision(approved=True)), \
                 mock.patch("reel.llm.config", return_value=_NO_SCORING_CFG):
                approved = cli._gate_stage_result(
                    gate, "soundscape", result, out=out, profile="fast",
                    existing=None, revise_keys=None, max_scenes=None, duration_kwargs={})

            self.assertEqual(approved, result)
            saved = json.loads((out / "soundscape.json").read_text())
            self.assertEqual(saved, result)

    def test_feedback_triggers_a_rerun_scoped_to_the_same_existing_and_revise_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            gate = Gate(enabled=True, timeout_seconds=0)
            initial = {"soundscapes": [{"scene_number": 1, "ambient_bed": "wind"}]}
            reran = {"soundscapes": [{"scene_number": 1, "ambient_bed": "storm"}]}
            existing = {"soundscapes": [{"scene_number": 1, "ambient_bed": "old"}]}

            run_stage_calls = []

            def fake_run_stage(name, *, out, profile=None, feedback=None,
                               existing=None, revise_keys=None, max_scenes=None, **kw):
                run_stage_calls.append(dict(name=name, feedback=feedback,
                                            existing=existing, revise_keys=revise_keys))
                return reran

            with mock.patch.object(Gate, "review", side_effect=[
                    Decision(approved=False, feedback="make it stormier"),
                    Decision(approved=True)]), \
                 mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.llm.config", return_value=_NO_SCORING_CFG):
                approved = cli._gate_stage_result(
                    gate, "soundscape", initial, out=out, profile="fast",
                    existing=existing, revise_keys={1}, max_scenes=None, duration_kwargs={})

            self.assertEqual(approved, reran)
            self.assertEqual(len(run_stage_calls), 1)
            call = run_stage_calls[0]
            self.assertEqual(call["feedback"], "make it stormier")
            # The rerun stayed scoped to the SAME existing/revise_keys this
            # round already committed to — feedback refines, never widens.
            self.assertEqual(call["existing"], existing)
            self.assertEqual(call["revise_keys"], {1})
            saved = json.loads((out / "soundscape.json").read_text())
            self.assertEqual(saved, reran)

    def test_extra_kwargs_survive_a_feedback_rerun(self):
        # A drastic "scenes" regen carries prior_scene_count (feeds the
        # revision reminder note) — a feedback-driven rerun must not
        # silently drop it.
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            gate = Gate(enabled=True, timeout_seconds=0)
            initial = {"scenes": [{"number": 1}]}
            reran = {"scenes": [{"number": 1}, {"number": 2}]}

            captured = {}

            def fake_run_stage(name, *, out, profile=None, feedback=None,
                               existing=None, revise_keys=None, max_scenes=None, **kw):
                captured.update(kw)
                return reran

            with mock.patch.object(Gate, "review", side_effect=[
                    Decision(approved=False, feedback="add the missing scene"),
                    Decision(approved=True)]), \
                 mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.llm.config", return_value=_NO_SCORING_CFG):
                cli._gate_stage_result(
                    gate, "scenes", initial, out=out, profile="fast",
                    existing=None, revise_keys=None, max_scenes=None, duration_kwargs={},
                    extra_kwargs={"prior_scene_count": 1})

            self.assertEqual(captured.get("prior_scene_count"), 1)

    def test_manual_edit_is_saved_even_with_no_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            gate = Gate(enabled=True, timeout_seconds=0)
            initial = {"soundscapes": [{"scene_number": 1, "ambient_bed": "wind"}]}
            edited = {"soundscapes": [{"scene_number": 1, "ambient_bed": "hand-edited"}]}

            with mock.patch.object(Gate, "review", side_effect=[
                    Decision(approved=False, edited=edited),
                    Decision(approved=True)]), \
                 mock.patch("reel.stages.run_stage") as mocked_run_stage, \
                 mock.patch("reel.llm.config", return_value=_NO_SCORING_CFG):
                approved = cli._gate_stage_result(
                    gate, "soundscape", initial, out=out, profile="fast",
                    existing=None, revise_keys=None, max_scenes=None, duration_kwargs={})

            mocked_run_stage.assert_not_called()   # a manual edit never calls run_stage
            self.assertEqual(approved, edited)
            saved = json.loads((out / "soundscape.json").read_text())
            self.assertEqual(saved, edited)

    def test_screenplay_approval_regenerates_the_fountain_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._write(out, "source", {"title": "T", "text": "story"})
            self._write(out, "structure", {"logline": "a story"})
            gate = Gate(enabled=True, timeout_seconds=0)
            result = {"scenes": [{"number": 1, "scene_number": 1,
                                 "fountain": "INT. BAR - DAY\n\nAction."}]}

            with mock.patch.object(Gate, "review", return_value=Decision(approved=True)), \
                 mock.patch("reel.llm.config", return_value=_NO_SCORING_CFG):
                cli._gate_stage_result(
                    gate, "screenplay", result, out=out, profile="fast",
                    existing=None, revise_keys=None, max_scenes=None, duration_kwargs={})

            fountain_text = (out / "screenplay.fountain").read_text()
            self.assertIn("Action.", fountain_text)

    def test_stop_raises_pipeline_stopped(self):
        gate = Gate(enabled=True, timeout_seconds=0)
        with mock.patch.object(Gate, "review", return_value=Decision(approved=False, stop=True)), \
             mock.patch("reel.llm.config", return_value=_NO_SCORING_CFG):
            with self.assertRaises(PipelineStopped):
                cli._gate_stage_result(
                    gate, "soundscape",
                    {"soundscapes": [{"scene_number": 1, "ambient_bed": "quiet room tone"}]},
                    out=Path("/tmp/whatever"),
                    profile="fast", existing=None, revise_keys=None,
                    max_scenes=None, duration_kwargs={})


if __name__ == "__main__":
    unittest.main()
