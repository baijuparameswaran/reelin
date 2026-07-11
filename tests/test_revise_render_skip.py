"""Validates that `revise` skips casting regeneration and all image/video
rendering by default, with an explicit opt-in to include them.

`revise` is the one place in this codebase that can trigger PAID API calls
(Gemini image generation, Veo video) purely as a side effect of editing a
cheap text stage — e.g. editing `characters` cascades to `casting`, which
cascades to `casting_images`. Iterating on text content shouldn't have to
pay that cost every round, so `RENDER_SKIP_STAGES` (casting +
casting_images/moodboard_tiles/scene_render) is skipped unless the caller
explicitly opts in via `render=True` (`--render` on the standalone command,
or 'render on'/'render off' inside the interactive loop).

Pure logic checks with mocked I/O — no LLM/API calls, no real $EDITOR, no
Ollama. Run with: python -m unittest tests.test_revise_render_skip -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import cli


class TestRenderSkipStages(unittest.TestCase):
    def test_render_skip_stages_are_exactly_casting_and_rendering(self):
        self.assertEqual(
            cli.RENDER_SKIP_STAGES,
            {"casting", "casting_images", "moodboard_tiles", "scene_render"},
        )


class TestReviseOneRenderSkip(unittest.TestCase):
    """Edits `casting` directly (a scenario already proven to reach
    downstream = {screenplay, storyboard, casting_images, scene_render,
    fidelity} — see test_revise_inheritance.py) and checks which of those
    actually run under render=False vs render=True."""

    def _setup_run(self, tmp: Path):
        (tmp / "scenes.json").write_text(json.dumps({"scenes": [
            {"number": 1, "location": "Bar", "characters": ["Marcel"],
             "source_line": "x", "summary": "s", "purpose": "p"},
        ]}))
        (tmp / "casting.json").write_text(json.dumps({"casting": [
            {"name": "Marcel", "kind": "person",
             "character": {"physical_form": "old"}},
        ]}))
        cli._save_run_params(tmp, max_scenes=None, profile="fast",
                             genre=None, target_duration=30)

    def test_render_false_skips_casting_images_and_scene_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            edited = json.loads((out / "casting.json").read_text())
            edited["casting"][0]["character"]["physical_form"] = "young"

            run_stage_calls = []
            scene_render_calls = []

            def fake_run_stage(name, *, out, **kwargs):
                run_stage_calls.append(name)
                return {}

            def fake_apply_scene_render_revision(*a, **kw):
                scene_render_calls.append((a, kw))

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch.object(cli, "_apply_scene_render_revision",
                                   side_effect=fake_apply_scene_render_revision):
                applied = cli._revise_one("casting", out, edited_override=edited,
                                          auto_confirm=True, profile="fast",
                                          max_scenes=None, target_duration=30,
                                          render=False)

            self.assertTrue(applied)
            self.assertNotIn("casting_images", run_stage_calls)
            self.assertEqual(scene_render_calls, [],
                             "_apply_scene_render_revision must not run when render=False")
            # Something downstream still ran (e.g. screenplay/storyboard/fidelity) —
            # confirms this is a SELECTIVE skip, not "nothing downstream runs at all".
            self.assertTrue(run_stage_calls,
                            "non-rendering downstream stages should still regenerate")

    def test_render_true_includes_casting_images_and_scene_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            edited = json.loads((out / "casting.json").read_text())
            edited["casting"][0]["character"]["physical_form"] = "young"

            run_stage_calls = []
            scene_render_calls = []

            def fake_run_stage(name, *, out, **kwargs):
                run_stage_calls.append(name)
                return {}

            def fake_apply_scene_render_revision(*a, **kw):
                scene_render_calls.append((a, kw))

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch.object(cli, "_apply_scene_render_revision",
                                   side_effect=fake_apply_scene_render_revision):
                applied = cli._revise_one("casting", out, edited_override=edited,
                                          auto_confirm=True, profile="fast",
                                          max_scenes=None, target_duration=30,
                                          render=True)

            self.assertTrue(applied)
            self.assertIn("casting_images", run_stage_calls)
            self.assertEqual(len(scene_render_calls), 1,
                             "_apply_scene_render_revision should run once when render=True")


class TestReviseSourceRenderSkip(unittest.TestCase):
    """A full source-text regen (always drastic) replays every STAGES
    entry — confirms the same skip applies there, since it's a completely
    separate code path (registry replay loop, not downstream_of())."""

    def _setup_run(self, tmp: Path):
        (tmp / "source.json").write_text(json.dumps(
            {"title": "T", "text": "original text", "chunks": []}))
        cli._save_run_params(tmp, max_scenes=1, profile="fast",
                             genre=None, target_duration=30)

    def test_render_false_skips_all_render_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            edited_source = {"title": "T", "text": "a genuinely different story text", "chunks": []}

            run_stage_calls = []

            def fake_run_stage(name, *, out, **kwargs):
                run_stage_calls.append(name)
                return {}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage):
                applied = cli._revise_source(out, edited_override=edited_source,
                                             auto_confirm=True, profile="fast",
                                             max_scenes=1, target_duration=30,
                                             render=False)

            self.assertTrue(applied)
            for skipped in cli.RENDER_SKIP_STAGES:
                self.assertNotIn(skipped, run_stage_calls)
            self.assertIn("structure", run_stage_calls)
            self.assertIn("screenplay", run_stage_calls)

    def test_render_true_includes_all_render_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            edited_source = {"title": "T", "text": "a genuinely different story text", "chunks": []}

            run_stage_calls = []

            def fake_run_stage(name, *, out, **kwargs):
                run_stage_calls.append(name)
                return {}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage):
                applied = cli._revise_source(out, edited_override=edited_source,
                                             auto_confirm=True, profile="fast",
                                             max_scenes=1, target_duration=30,
                                             render=True)

            self.assertTrue(applied)
            for included in cli.RENDER_SKIP_STAGES:
                self.assertIn(included, run_stage_calls)


if __name__ == "__main__":
    unittest.main()
