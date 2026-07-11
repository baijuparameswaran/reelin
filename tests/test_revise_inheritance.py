"""Validates that the `revise` command (reel.cli's standalone `revise`
subcommand and the auto-offered post-run loop) inherits the ORIGINAL
pipeline run's attributes — profile, target_duration, and the genre/
moodboard creative-direction steering — rather than silently reverting to
bare defaults, and that `max_scenes` is the ONE attribute deliberately NOT
inherited (see `TestReviseLoopAlwaysUsesMaxScenesAll` below).

`revise` is a separate process invocation from `pipeline.run()`, so none of
these carried over automatically before an earlier fix: `llm.set_direction`'s
process-wide directive started unset (losing genre/moodboard steering on
every revised stage); `stages.run_stage`'s calls never passed `profile=`
(reverting to each stage's own default model tier); `max_scenes` wasn't
threaded through at all (silently re-capping a `--max-scenes all` run back
down to `run_stage`'s own default of 1); and `--target-duration`'s
scene/shot-count guidance had no pathway into `revise`'s calls whatsoever.
A LATER change went further for `max_scenes` specifically: rather than just
inheriting the original run's value, `_revise_loop` now always passes
`None` ("all") regardless of what that value was — a revision should never
re-cap rendering to a prototype-scale `--max-scenes N` just because that's
what the original full-pipeline run happened to use; the diff-based
`revise_keys` scoping (not `max_scenes`) is what actually controls which
scenes get regenerated. See PROGRESS.md's session log for both changes.

Pure logic checks with mocked I/O — no LLM/API calls, no real $EDITOR, no
Ollama. Run with: python -m unittest tests.test_revise_inheritance -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import cli, llm, pipeline


class TestComposeDirection(unittest.TestCase):
    """`pipeline.compose_direction` is the shared primitive both a fresh
    `pipeline.run()` and `cli.py`'s revise flow use to build the steering
    string — the foundation `_restore_direction` (tested below) depends on."""

    def test_composes_genre_and_moodboard_guidance(self):
        direction = pipeline.compose_direction(
            {"genre": "Noir", "tone": "cynical"},
            {"overall_aesthetic": "rain-slicked neon streets"},
        )
        self.assertIn("Noir", direction)
        self.assertIn("rain-slicked neon streets", direction)

    def test_empty_specs_yield_none(self):
        self.assertIsNone(pipeline.compose_direction({}, {}))
        self.assertIsNone(pipeline.compose_direction(None, None))

    def test_genre_only_when_moodboard_missing(self):
        direction = pipeline.compose_direction({"genre": "Comedy"}, {})
        self.assertIn("Comedy", direction)


class TestRestoreDirection(unittest.TestCase):
    """The real bug: `revise` is a fresh process, so `llm.set_direction`
    starts unset even when genre.json/moodboard.json exist on disk from the
    original run. `_restore_direction` must reload and re-apply them."""

    def setUp(self):
        llm.set_direction(None)   # start each test from a clean slate

    def tearDown(self):
        llm.set_direction(None)   # don't leak steering state into other tests

    def test_reloads_and_applies_genre_and_moodboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "genre.json").write_text(json.dumps({"genre": "Noir", "tone": "cynical"}))
            (out / "moodboard.json").write_text(json.dumps({"overall_aesthetic": "neon rain"}))
            self.assertIsNone(llm.direction())
            cli._restore_direction(out)
            self.assertIsNotNone(llm.direction())
            self.assertIn("Noir", llm.direction())
            self.assertIn("neon rain", llm.direction())

    def test_missing_genre_and_moodboard_files_degrade_to_none_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cli._restore_direction(out)   # neither file exists
            self.assertIsNone(llm.direction())


class TestEffectiveMaxScenes(unittest.TestCase):
    """`--max-scenes` only ever restricts actual media-rendering stages in a
    full pipeline run (casting-image generation, scene_render, moodboard
    tiles) — screenplay always drafts every scene regardless (see
    PROGRESS.md's "Current state"). `revise` must preserve that policy
    rather than reintroducing a cap via `run_stage`'s own non-None default."""

    def test_screenplay_always_uncapped_regardless_of_inherited_value(self):
        self.assertIsNone(cli._effective_max_scenes("screenplay", 3))
        self.assertIsNone(cli._effective_max_scenes("screenplay", 1))
        self.assertIsNone(cli._effective_max_scenes("screenplay", None))

    def test_other_stages_get_the_inherited_value_as_is(self):
        self.assertEqual(cli._effective_max_scenes("casting_images", 3), 3)
        self.assertEqual(cli._effective_max_scenes("scene_render", None), None)
        self.assertEqual(cli._effective_max_scenes("moodboard", 5), 5)


class TestDurationKwargs(unittest.TestCase):
    """`--target-duration`'s planning guidance only applies to the two
    stages that actually consume it (`scenes`/`cinematography`); every
    other stage gets nothing to compute (and would ignore it anyway via
    its own `**_` catch-all)."""

    def test_empty_for_unrelated_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for stage in ("casting", "soundscape", "visuals", "screenplay",
                         "storyboard", "characters", "structure"):
                with self.subTest(stage=stage):
                    self.assertEqual(cli._duration_kwargs(stage, out, 45), {})

    def test_scenes_gets_a_target_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            kwargs = cli._duration_kwargs("scenes", out, 10)
            self.assertIn("target", kwargs)
            self.assertIsInstance(kwargs["target"], str)
            self.assertIn("10s", kwargs["target"])

    def test_cinematography_uses_current_scene_count_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "scenes.json").write_text(json.dumps(
                {"scenes": [{"number": i} for i in range(1, 6)]}))
            kwargs = cli._duration_kwargs("cinematography", out, 45)
            self.assertIn("shots_guidance", kwargs)
            self.assertIn("5 scene(s)", kwargs["shots_guidance"])

    def test_falls_back_to_default_target_when_none_inherited(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            from reel import duration_budget
            kwargs = cli._duration_kwargs("scenes", out, None)
            self.assertIn(f"{duration_budget.DEFAULT_TARGET_SECONDS}s", kwargs["target"])


class TestReviseThreadsInheritedAttributes(unittest.TestCase):
    """End-to-end (mocked): drive `_revise_one`/`_revise_source` with a
    canned edit and a stubbed `run_stage`, and confirm the ORIGINAL run's
    profile/max_scenes actually reach every call — not just that the
    plumbing functions above work in isolation."""

    def _setup_run(self, tmp: Path, *, profile="fast", max_scenes=None):
        """A minimal completed run: scenes.json + casting.json on disk,
        plus the run_params.json a real `pipeline.run()` would have
        written (via `_save_run_params`)."""
        (tmp / "scenes.json").write_text(json.dumps({"scenes": [
            {"number": 1, "location": "Bar", "characters": ["Marcel"],
             "source_line": "x", "summary": "s", "purpose": "p"},
        ]}))
        (tmp / "casting.json").write_text(json.dumps({"casting": [
            {"name": "Marcel", "kind": "person",
             "character": {"physical_form": "old"}},
        ]}))
        cli._save_run_params(tmp, max_scenes=max_scenes, profile=profile,
                             genre=None, target_duration=30)

    def test_revise_one_passes_inherited_profile_and_max_scenes_downstream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out, profile="fast", max_scenes=None)
            edited = json.loads((out / "casting.json").read_text())
            edited["casting"][0]["character"]["physical_form"] = "young"

            calls = []

            def fake_run_stage(name, *, out, **kwargs):
                calls.append((name, kwargs))
                return {}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage):
                cli._revise_one("casting", out, edited_override=edited,
                                auto_confirm=True, profile="fast",
                                max_scenes=None, target_duration=30)

            self.assertTrue(calls, "expected at least one downstream run_stage call")
            for name, kwargs in calls:
                with self.subTest(stage=name):
                    self.assertEqual(kwargs.get("profile"), "fast",
                                     f"{name} didn't inherit the original run's profile")
                    if name == "screenplay":
                        self.assertIsNone(kwargs.get("max_scenes"),
                                          "screenplay must stay uncapped regardless "
                                          "of the inherited max_scenes")
                    else:
                        self.assertIsNone(kwargs.get("max_scenes"),
                                          f"{name} should inherit max_scenes=None (all)")

    def test_revise_source_passes_inherited_attributes_to_every_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out, profile="quality", max_scenes=2)
            (out / "source.json").write_text(json.dumps(
                {"title": "T", "text": "original text", "chunks": []}))
            edited_source = {"title": "T", "text": "revised text with a real change", "chunks": []}

            calls = []

            def fake_run_stage(name, *, out, **kwargs):
                calls.append((name, kwargs))
                return {}

            # This test is about profile/max_scenes threading through the
            # FULL-REGEN fallback path specifically, not the new scoped
            # source-text analysis — force drastic=True so it takes that
            # path deterministically, without a live LLM call.
            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.agents.revision.identify_source_text_changes",
                           return_value={"drastic": True, "reason": "forced for this test",
                                        "changed_scene_numbers": [], "summary": ""}):
                cli._revise_source(out, edited_override=edited_source, auto_confirm=True,
                                   profile="quality", max_scenes=2, target_duration=30)

            self.assertTrue(calls)
            for name, kwargs in calls:
                with self.subTest(stage=name):
                    self.assertEqual(kwargs.get("profile"), "quality")
                    if name == "screenplay":
                        self.assertIsNone(kwargs.get("max_scenes"))
                    else:
                        self.assertEqual(kwargs.get("max_scenes"), 2)


class TestReviseLoopAlwaysUsesMaxScenesAll(unittest.TestCase):
    """`_revise_loop` must always pass `max_scenes=None` ("all") to
    `_revise_one`, regardless of what the original run's `--max-scenes`
    (persisted to run_params.json) actually was — a revision should never
    silently re-cap rendering to a prototype-scale value. Drives the loop
    via mocked `input()` (pick one stage, then 'quit') and a mocked
    `cli._revise_one` call-recorder — no real $EDITOR, no LLM/API calls."""

    def _setup_run(self, tmp: Path, *, max_scenes) -> None:
        (tmp / "casting.json").write_text(json.dumps({"casting": []}))
        cli._save_run_params(tmp, max_scenes=max_scenes, profile="fast",
                             genre=None, target_duration=30)

    def test_a_small_inherited_max_scenes_does_not_reach_revise_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            # The original run explicitly used --max-scenes 1 (the
            # prototype default) — revise must NOT inherit that.
            self._setup_run(out, max_scenes=1)

            calls = []

            def fake_revise_one(stage_name, out, **kwargs):
                calls.append((stage_name, kwargs.get("max_scenes")))
                return True

            with mock.patch("reel.cli._revise_one", side_effect=fake_revise_one), \
                 mock.patch("reel.cli._restore_direction"), \
                 mock.patch("builtins.input", side_effect=["casting", "quit"]):
                cli._revise_loop(out)

            self.assertEqual(calls, [("casting", None)])

    def test_all_already_inherited_stays_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out, max_scenes=None)

            calls = []

            def fake_revise_one(stage_name, out, **kwargs):
                calls.append(kwargs.get("max_scenes"))
                return True

            with mock.patch("reel.cli._revise_one", side_effect=fake_revise_one), \
                 mock.patch("reel.cli._restore_direction"), \
                 mock.patch("builtins.input", side_effect=["casting", "quit"]):
                cli._revise_loop(out)

            self.assertEqual(calls, [None])

    def test_no_run_params_at_all_still_uses_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "casting.json").write_text(json.dumps({"casting": []}))
            # No run_params.json written at all this time.

            calls = []

            def fake_revise_one(stage_name, out, **kwargs):
                calls.append(kwargs.get("max_scenes"))
                return True

            with mock.patch("reel.cli._revise_one", side_effect=fake_revise_one), \
                 mock.patch("reel.cli._restore_direction"), \
                 mock.patch("builtins.input", side_effect=["casting", "quit"]):
                cli._revise_loop(out)

            self.assertEqual(calls, [None])


if __name__ == "__main__":
    unittest.main()
