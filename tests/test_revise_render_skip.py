"""Validates that `revise` skips casting regeneration and all image/video
rendering by default, with image rendering and video rendering each
independently enable-able.

`revise` is the one place in this codebase that can trigger PAID API calls
(Gemini image generation, Veo video) purely as a side effect of editing a
cheap text stage — e.g. editing `characters` cascades to `casting`, which
cascades to `casting_images`. Iterating on text content shouldn't have to
pay that cost every round, so `IMAGE_RENDER_STAGES` (casting/casting_images/
moodboard_tiles — Gemini spend) and `VIDEO_RENDER_STAGES` (scene_render —
Veo spend) are each skipped unless the caller explicitly opts in via
`render_images=True`/`render_video=True` (`--render-images`/`--render-video`
on the standalone command, or 'render images on/off'/'render video on/off'
inside the interactive loop) — independently of each other, so an operator
can e.g. re-cast a character's look and see the portrait update without
paying for a full video re-render, or vice versa.

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
from reel.gate import Gate


class TestRenderStageCategories(unittest.TestCase):
    def test_image_and_video_render_stages_partition_render_skip_stages(self):
        self.assertEqual(cli.IMAGE_RENDER_STAGES,
                         {"casting", "casting_images", "moodboard_tiles"})
        self.assertEqual(cli.VIDEO_RENDER_STAGES, {"scene_render"})
        self.assertEqual(cli.RENDER_SKIP_STAGES,
                         cli.IMAGE_RENDER_STAGES | cli.VIDEO_RENDER_STAGES)
        self.assertEqual(cli.IMAGE_RENDER_STAGES & cli.VIDEO_RENDER_STAGES, set())

    def test_render_stage_skipped_checks_the_right_category(self):
        # image category
        self.assertTrue(cli._render_stage_skipped("casting", False, True))
        self.assertFalse(cli._render_stage_skipped("casting", True, False))
        self.assertTrue(cli._render_stage_skipped("casting_images", False, False))
        self.assertFalse(cli._render_stage_skipped("moodboard_tiles", True, False))
        # video category — independent of the image flag
        self.assertTrue(cli._render_stage_skipped("scene_render", True, False))
        self.assertFalse(cli._render_stage_skipped("scene_render", False, True))
        # a non-render stage is never skipped by this check
        self.assertFalse(cli._render_stage_skipped("screenplay", False, False))


class TestReviseOneRenderSkip(unittest.TestCase):
    """Edits `casting` directly (a scenario already proven to reach
    downstream = {screenplay, storyboard, casting_images, scene_render,
    fidelity} — see test_revise_inheritance.py) and checks which of those
    actually run under every render_images/render_video combination."""

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

    def _revise_casting(self, render_images: bool, render_video: bool):
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
                                          render_images=render_images,
                                          render_video=render_video)
            self.assertTrue(applied)
            return run_stage_calls, scene_render_calls

    def test_both_off_skips_casting_images_and_scene_render(self):
        run_stage_calls, scene_render_calls = self._revise_casting(False, False)
        self.assertNotIn("casting_images", run_stage_calls)
        self.assertEqual(scene_render_calls, [])
        self.assertTrue(run_stage_calls,
                        "non-rendering downstream stages should still regenerate")

    def test_both_on_includes_casting_images_and_scene_render(self):
        run_stage_calls, scene_render_calls = self._revise_casting(True, True)
        self.assertIn("casting_images", run_stage_calls)
        self.assertEqual(len(scene_render_calls), 1)

    def test_images_only_includes_casting_images_but_not_scene_render(self):
        run_stage_calls, scene_render_calls = self._revise_casting(True, False)
        self.assertIn("casting_images", run_stage_calls)
        self.assertEqual(scene_render_calls, [],
                         "video rendering must stay skipped when only images are enabled")

    def test_video_only_includes_scene_render_but_not_casting_images(self):
        run_stage_calls, scene_render_calls = self._revise_casting(False, True)
        self.assertNotIn("casting_images", run_stage_calls,
                         "image rendering must stay skipped when only video is enabled")
        self.assertEqual(len(scene_render_calls), 1)


class TestReviseSourceRenderSkip(unittest.TestCase):
    """A full source-text regen (always drastic) replays every STAGES
    entry — confirms the same independent skip applies there, since it's a
    completely separate code path (registry replay loop, not
    downstream_of())."""

    def _setup_run(self, tmp: Path):
        (tmp / "source.json").write_text(json.dumps(
            {"title": "T", "text": "original text", "chunks": []}))
        cli._save_run_params(tmp, max_scenes=1, profile="fast",
                             genre=None, target_duration=30)

    def _revise_source(self, render_images: bool, render_video: bool):
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
                                             render_images=render_images,
                                             render_video=render_video)
            self.assertTrue(applied)
            return run_stage_calls

    def test_both_off_skips_all_render_stages(self):
        run_stage_calls = self._revise_source(False, False)
        for skipped in cli.RENDER_SKIP_STAGES:
            self.assertNotIn(skipped, run_stage_calls)
        self.assertIn("structure", run_stage_calls)
        self.assertIn("screenplay", run_stage_calls)

    def test_both_on_includes_all_render_stages(self):
        run_stage_calls = self._revise_source(True, True)
        for included in cli.RENDER_SKIP_STAGES:
            self.assertIn(included, run_stage_calls)

    def test_images_only_includes_image_stages_not_video(self):
        run_stage_calls = self._revise_source(True, False)
        for included in cli.IMAGE_RENDER_STAGES:
            self.assertIn(included, run_stage_calls)
        for skipped in cli.VIDEO_RENDER_STAGES:
            self.assertNotIn(skipped, run_stage_calls)

    def test_video_only_includes_video_stages_not_images(self):
        run_stage_calls = self._revise_source(False, True)
        for included in cli.VIDEO_RENDER_STAGES:
            self.assertIn(included, run_stage_calls)
        for skipped in cli.IMAGE_RENDER_STAGES:
            self.assertNotIn(skipped, run_stage_calls)


class TestReviseLoopRenderToggles(unittest.TestCase):
    """Drives `_revise_loop`'s interactive menu (mocked `input()`) to confirm
    the four toggle command pairs ('render images on/off', 'render video
    on/off', and the 'render on/off' shorthand for both) each flip the
    right flag(s) before reaching `_revise_one`."""

    def _setup_run(self, tmp: Path):
        cli._save_run_params(tmp, max_scenes=None, profile=None,
                             genre=None, target_duration=None)

    def _run_loop(self, commands: list[str]):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            calls = []

            def fake_revise_one(choice, out, **kwargs):
                calls.append(dict(kwargs))
                return True

            inputs = iter(commands + ["quit"])
            with mock.patch("builtins.input", side_effect=lambda *_a: next(inputs)), \
                 mock.patch.object(cli, "_revise_one", side_effect=fake_revise_one), \
                 mock.patch.object(cli, "_restore_direction"), \
                 mock.patch.object(Gate, "from_config", return_value=Gate(enabled=False)):
                cli._revise_loop(out)
            return calls

    def test_render_images_on_then_a_stage_pick(self):
        calls = self._run_loop(["render images on", "characters"])
        self.assertTrue(calls)
        self.assertTrue(calls[0]["render_images"])
        self.assertFalse(calls[0]["render_video"])

    def test_render_video_on_then_a_stage_pick(self):
        calls = self._run_loop(["render video on", "characters"])
        self.assertTrue(calls)
        self.assertFalse(calls[0]["render_images"])
        self.assertTrue(calls[0]["render_video"])

    def test_render_on_enables_both(self):
        calls = self._run_loop(["render on", "characters"])
        self.assertTrue(calls[0]["render_images"])
        self.assertTrue(calls[0]["render_video"])

    def test_render_off_after_render_on_disables_both(self):
        calls = self._run_loop(["render on", "render off", "characters"])
        self.assertFalse(calls[0]["render_images"])
        self.assertFalse(calls[0]["render_video"])

    def test_images_and_video_can_be_toggled_independently(self):
        calls = self._run_loop(["render images on", "render video on",
                                "render images off", "characters"])
        self.assertFalse(calls[0]["render_images"])
        self.assertTrue(calls[0]["render_video"])


if __name__ == "__main__":
    unittest.main()
