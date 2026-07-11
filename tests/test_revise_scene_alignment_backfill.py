"""Regression test for a reported bug: `revise` could leave screenplay.json/
storyboard.json (and any other scene-keyed downstream artifact) permanently
stuck covering only a subset of scenes — e.g. "only scene 1" — no matter how
many later revision rounds ran, because there was no reason it should have
been out of sync with scenes.json in the first place. `revision_merge.
merge_by_key` (what every scoped agent call uses) only ever ADDS or REPLACES
the specific keys a given round's `revise_keys` names, carrying every other
existing entry over byte-identical — so if a downstream artifact's existing
data was ever missing a scene (e.g. drafted before that scene existed, or an
earlier revision round's cascade never reached it), no LATER, unrelated
scoped revision would ever notice or backfill it merely by being in that
edit's own `stages.downstream_of` closure.

Fixed with a new, unconditional invariant: `cli._align_scene_keyed_stages`
is called once after EVERY `_revise_one`/`_revise_source` round completes
— regardless of which stage was actually edited, and regardless of whether
that edit's own downstream cascade happened to reach a given scene-keyed
stage. It re-checks every scene-keyed artifact already on disk (soundscape/
visuals/cinematography/screenplay/storyboard) against the CURRENT
scenes.json via `fidelity.check_scene_alignment`/`strip_orphan_scenes` (the
exact same deterministic primitives a fresh `pipeline.run()` already
self-heals with inside `run_group`) and reiterates anything found missing.

Pure logic checks with mocked I/O — no LLM/API calls, no real $EDITOR, no
Ollama. Run with:
    python -m unittest tests.test_revise_scene_alignment_backfill -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import cli


class TestAlignSceneKeyedStages(unittest.TestCase):
    def test_missing_scene_is_backfilled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "scenes.json").write_text(json.dumps({"scenes": [
                {"number": 1, "location": "Bar"},
                {"number": 2, "location": "Bar"},
                {"number": 3, "location": "Bar"},
            ]}), encoding="utf-8")
            # screenplay.json only ever covered scene 1 — stale/incomplete,
            # e.g. from before scenes 2/3 existed, or a skipped cascade.
            (out / "screenplay.json").write_text(json.dumps({
                "drafted_count": 1, "total_scenes": 3,
                "scenes": [{"number": 1, "scene_number": 1, "shots": []}],
            }), encoding="utf-8")

            calls = []

            def fake_run_stage(name, *, out, existing=None, revise_keys=None, **kwargs):
                calls.append((name, revise_keys))
                return {"drafted_count": 0, "scenes": []}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage):
                cli._align_scene_keyed_stages(out, profile=None, max_scenes=None,
                                              target_duration=None)

            self.assertIn(("screenplay", {2, 3}), calls)

    def test_no_op_when_everything_already_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "scenes.json").write_text(json.dumps({"scenes": [
                {"number": 1}, {"number": 2},
            ]}), encoding="utf-8")
            (out / "screenplay.json").write_text(json.dumps({
                "scenes": [{"number": 1, "scene_number": 1}, {"number": 2, "scene_number": 2}],
            }), encoding="utf-8")

            calls = []
            with mock.patch("reel.stages.run_stage",
                           side_effect=lambda *a, **k: calls.append(a) or {}):
                cli._align_scene_keyed_stages(out, profile=None, max_scenes=None,
                                              target_duration=None)

            self.assertEqual(calls, [])

    def test_stage_never_generated_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "scenes.json").write_text(json.dumps({"scenes": [{"number": 1}]}),
                                             encoding="utf-8")
            # storyboard.json doesn't exist at all yet — nothing to align.
            calls = []
            with mock.patch("reel.stages.run_stage",
                           side_effect=lambda *a, **k: calls.append(a) or {}):
                cli._align_scene_keyed_stages(out, profile=None, max_scenes=None,
                                              target_duration=None)
            self.assertEqual(calls, [])

    def test_orphan_scene_is_stripped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "scenes.json").write_text(json.dumps({"scenes": [{"number": 1}]}),
                                             encoding="utf-8")
            (out / "soundscape.json").write_text(json.dumps({
                "soundscapes": [{"scene_number": 1}, {"scene_number": 99}],
            }), encoding="utf-8")
            with mock.patch("reel.stages.run_stage", side_effect=AssertionError(
                    "no missing scenes here — only an orphan to strip, run_stage "
                    "shouldn't be called")):
                cli._align_scene_keyed_stages(out, profile=None, max_scenes=None,
                                              target_duration=None)
            saved = json.loads((out / "soundscape.json").read_text())
            self.assertEqual([s["scene_number"] for s in saved["soundscapes"]], [1])

    def test_fires_regardless_of_which_stage_was_edited(self):
        # The whole point of this guarantee: it's not conditional on the
        # edited stage's own `downstream_of` closure reaching screenplay —
        # `_revise_one` calls `_align_scene_keyed_stages` unconditionally
        # after every round, so a screenplay that's already out of sync
        # gets caught even when the edit that round was to an unrelated
        # stage whose cascade doesn't happen to include it.
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "scenes.json").write_text(json.dumps({"scenes": [
                {"number": 1}, {"number": 2},
            ]}), encoding="utf-8")
            (out / "characters.json").write_text(json.dumps({"characters": [
                {"name": "Alice", "role": "lead"},
            ]}), encoding="utf-8")
            (out / "screenplay.json").write_text(json.dumps({
                "scenes": [{"number": 1, "scene_number": 1}],
            }), encoding="utf-8")

            run_stage_calls = []

            def fake_run_stage(name, *, out, existing=None, revise_keys=None, **kwargs):
                run_stage_calls.append((name, revise_keys))
                return existing if existing is not None else {}

            edited_characters = {"characters": [{"name": "Alice", "role": "protagonist"}]}
            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.stages.downstream_of", return_value=[]):
                applied = cli._revise_one("characters", out, edited_override=edited_characters,
                                          auto_confirm=True, render=False)

            self.assertTrue(applied)
            self.assertIn(("screenplay", {2}), run_stage_calls)


if __name__ == "__main__":
    unittest.main()
