"""Validates that the `revise` flow can DELETE and ADD scenes directly on
scenes.json — not just modify existing ones — propagating a deletion to
every other scene-keyed artifact (soundscape/visuals/cinematography/
screenplay/storyboard), screenplay.fountain, and the video manifest, since
none of those can delete a key on their own (`revision_merge.merge_by_key`
only ever adds/replaces). See PROGRESS.md's session log for the background.

Pure logic checks with mocked I/O — no LLM/API calls, no real $EDITOR, no
Ollama, no ffmpeg. Run with:
    python -m unittest tests.test_revise_scene_delete_add -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import artifact_diff, cli
from reel.agents import scenes as scenes_agent


class TestScenesShapeAllowsAddRemove(unittest.TestCase):
    def test_adding_a_scene_is_not_drastic(self):
        old = {"scenes": [{"number": 1, "summary": "a"}, {"number": 2, "summary": "b"}]}
        new = {"scenes": [{"number": 1, "summary": "a"}, {"number": 2, "summary": "b"},
                          {"number": 3, "summary": "c"}]}
        diff = artifact_diff.diff_artifact("scenes", old, new)
        self.assertFalse(diff.drastic)
        self.assertEqual(diff.added, [3])
        self.assertEqual(diff.removed, [])

    def test_removing_a_scene_is_not_drastic(self):
        old = {"scenes": [{"number": 1, "summary": "a"}, {"number": 2, "summary": "b"}]}
        new = {"scenes": [{"number": 1, "summary": "a"}]}
        diff = artifact_diff.diff_artifact("scenes", old, new)
        self.assertFalse(diff.drastic)
        self.assertEqual(diff.removed, [2])
        self.assertEqual(diff.added, [])

    def test_other_scene_keyed_artifacts_still_treat_removal_as_drastic(self):
        # soundscape/visuals/cinematography/screenplay/storyboard are kept in
        # sync with scenes.json by the cascade, not independently editable
        # to add/remove scenes — a direct edit that drops one is still
        # unrelated/drastic for THAT artifact's own diff.
        old = {"soundscapes": [{"scene_number": 1}, {"scene_number": 2}]}
        new = {"soundscapes": [{"scene_number": 1}]}
        diff = artifact_diff.diff_artifact("soundscape", old, new)
        self.assertTrue(diff.drastic)


class TestSegmentScenesMergeSortsNewScenes(unittest.TestCase):
    def test_a_genuinely_new_scene_number_lands_in_narrative_order(self):
        existing = {"scenes": [
            {"number": 1, "summary": "s1"},
            {"number": 3, "summary": "s3"},
        ]}
        # Model's fresh pass returns all 3 (full-story context), but the
        # caller only trusts number 2 (the newly-inserted one) via
        # revise_keys — merge_by_key appends it, then segment_scenes must
        # re-sort so it lands BETWEEN 1 and 3, not after them.
        fake_scenes = [
            {"number": 1, "summary": "s1 (model rewrote this, should be ignored)"},
            {"number": 2, "summary": "s2 (the new one)"},
            {"number": 3, "summary": "s3 (model rewrote this, should be ignored)"},
        ]
        with mock.patch.object(scenes_agent.llm, "generate",
                              return_value=json.dumps({"scenes": fake_scenes})), \
             mock.patch.object(scenes_agent, "_validate", side_effect=lambda s, t: (s, [])), \
             mock.patch.object(scenes_agent, "_attach_source_excerpts", side_effect=lambda s, t: s), \
             mock.patch.object(scenes_agent, "_map_chunks", side_effect=lambda s, src: s), \
             mock.patch.object(scenes_agent, "_reconcile_character_names", side_effect=lambda s, c: s):
            result = scenes_agent.segment_scenes(
                {"title": "T", "text": "..."}, {"three_act": {}},
                existing=existing, revise_keys={2})

        numbers = [s["number"] for s in result["scenes"]]
        self.assertEqual(numbers, [1, 2, 3])
        # untouched scenes came back byte-identical from `existing`, not the model's rewrite
        self.assertEqual(result["scenes"][0]["summary"], "s1")
        self.assertEqual(result["scenes"][2]["summary"], "s3")
        self.assertEqual(result["scenes"][1]["summary"], "s2 (the new one)")


class TestTranslateReviseKeysEmptyScope(unittest.TestCase):
    def test_empty_revise_keys_translates_to_empty_not_drastic(self):
        # A scenes.json edit that ONLY deleted scenes (nothing added/changed)
        # reaches downstream translation with revise_keys=set() — must stay
        # an empty (skip) scope, not None (which means "full regen").
        result = cli._translate_revise_keys("scenes", "casting", set(), {"scenes": []})
        self.assertEqual(result, set())

    def test_none_still_means_drastic(self):
        self.assertIsNone(cli._translate_revise_keys("scenes", "casting", None, {}))

    def test_nonempty_with_no_locations_found_still_falls_back(self):
        edited = {"scenes": [{"number": 1, "location": None}]}
        result = cli._translate_revise_keys("scenes", "casting", {1}, edited)
        self.assertIsNone(result)


class TestReviseOneScenesDeletion(unittest.TestCase):
    """End-to-end (mocked run_stage, real strip/merge logic, stubbed
    ffmpeg/i2v): deleting a scene from scenes.json must remove that scene's
    entry from every downstream scene-keyed artifact, regenerate
    screenplay.fountain, and drop it from the video manifest + reassembled
    movie — without calling any LLM for stages where nothing else changed."""

    def _write(self, out: Path, name: str, data: dict) -> None:
        (out / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")

    def _setup_run(self, out: Path):
        self._write(out, "source", {"title": "T", "text": "story text"})
        self._write(out, "structure", {"logline": "a story"})
        self._write(out, "scenes", {"scenes": [
            {"number": 1, "location": "Bar", "characters": ["Marcel"], "summary": "s1"},
            {"number": 2, "location": "Road", "characters": ["Marcel"], "summary": "s2"},
            {"number": 3, "location": "Home", "characters": ["Marcel"], "summary": "s3"},
        ]})
        self._write(out, "soundscape", {"soundscapes": [
            {"scene_number": 1, "score_direction": "tense"},
            {"scene_number": 2, "score_direction": "calm"},
            {"scene_number": 3, "score_direction": "resolved"},
        ]})
        self._write(out, "screenplay", {
            "scenes": [
                {"scene_number": 1, "fountain": "INT. BAR - DAY\n\nScene one."},
                {"scene_number": 2, "fountain": "EXT. ROAD - DAY\n\nScene two."},
                {"scene_number": 3, "fountain": "INT. HOME - DAY\n\nScene three."},
            ],
            "drafted_count": 3, "total_scenes": 3,
        })
        (out / "video").mkdir()
        (out / "video" / "manifest.json").write_text(json.dumps({
            "clips": 3, "failed": 0,
            "scenes": [
                {"scene_number": 1, "scene_video": "video/scene_01.mp4", "frames": []},
                {"scene_number": 2, "scene_video": "video/scene_02.mp4", "frames": []},
                {"scene_number": 3, "scene_video": "video/scene_03.mp4", "frames": []},
            ],
        }), encoding="utf-8")
        cli._save_run_params(out, max_scenes=None, profile="fast", genre=None, target_duration=30)

    def test_deleting_scene_2_removes_it_from_every_downstream_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            current_scenes = json.loads((out / "scenes.json").read_text())
            edited_scenes = {"scenes": [s for s in current_scenes["scenes"] if s["number"] != 2]}

            run_stage_calls = []

            def fake_run_stage(name, *, out, existing=None, revise_keys=None, **kwargs):
                run_stage_calls.append((name, revise_keys))
                return existing if existing is not None else {}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.pipeline._assemble_movie", return_value=None):
                applied = cli._revise_one("scenes", out, edited_override=edited_scenes,
                                          auto_confirm=True, render_images=True, render_video=True)

            self.assertTrue(applied)

            # scenes.json itself reflects the deletion
            saved_scenes = json.loads((out / "scenes.json").read_text())
            self.assertEqual([s["number"] for s in saved_scenes["scenes"]], [1, 3])

            # soundscape.json had scene 2's entry stripped, deterministically
            # (no run_stage call needed — nothing else changed)
            saved_sound = json.loads((out / "soundscape.json").read_text())
            self.assertEqual([s["scene_number"] for s in saved_sound["soundscapes"]], [1, 3])

            # screenplay.json stripped + counts corrected + .fountain regenerated
            saved_screenplay = json.loads((out / "screenplay.json").read_text())
            self.assertEqual([s["scene_number"] for s in saved_screenplay["scenes"]], [1, 3])
            self.assertEqual(saved_screenplay["drafted_count"], 2)
            self.assertEqual(saved_screenplay["total_scenes"], 2)
            fountain_text = (out / "screenplay.fountain").read_text()
            self.assertIn("Scene one.", fountain_text)
            self.assertIn("Scene three.", fountain_text)
            self.assertNotIn("Scene two.", fountain_text)

            # video manifest had scene 2 stripped; movie re-assembled (mocked)
            manifest = json.loads((out / "video" / "manifest.json").read_text())
            self.assertEqual([s["scene_number"] for s in manifest["scenes"]], [1, 3])

            # nothing else changed for soundscape (no add/change scenes) —
            # so no wasted LLM call for it; scenes itself did NOT get called
            # again (it's the artifact being edited, saved directly)
            called_names = [n for n, _ in run_stage_calls]
            self.assertNotIn("soundscape", called_names,
                             "soundscape had only a deletion (nothing to regenerate) — "
                             "run_stage should have been skipped entirely for it")

    def test_deletion_only_does_not_hit_the_no_changes_detected_bailout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            current_scenes = json.loads((out / "scenes.json").read_text())
            edited_scenes = {"scenes": [s for s in current_scenes["scenes"] if s["number"] != 3]}

            with mock.patch("reel.stages.run_stage", return_value={}), \
                 mock.patch("reel.pipeline._assemble_movie", return_value=None):
                applied = cli._revise_one("scenes", out, edited_override=edited_scenes,
                                          auto_confirm=True, render_images=True, render_video=True)
            self.assertTrue(applied)

    def test_adding_a_scene_scopes_the_new_number_downstream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            current_scenes = json.loads((out / "scenes.json").read_text())
            edited_scenes = {"scenes": current_scenes["scenes"] + [
                {"number": 4, "location": "Cafe", "characters": ["Marcel"], "summary": "s4"},
            ]}

            run_stage_calls = []

            def fake_run_stage(name, *, out, existing=None, revise_keys=None, **kwargs):
                run_stage_calls.append((name, revise_keys))
                return existing if existing is not None else {}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.pipeline._assemble_movie", return_value=None):
                applied = cli._revise_one("scenes", out, edited_override=edited_scenes,
                                          auto_confirm=True, render_images=True, render_video=True)

            self.assertTrue(applied)
            soundscape_call = next(rk for n, rk in run_stage_calls if n == "soundscape")
            self.assertEqual(soundscape_call, {4})
            # no deletions here — video manifest untouched
            manifest = json.loads((out / "video" / "manifest.json").read_text())
            self.assertEqual([s["scene_number"] for s in manifest["scenes"]], [1, 2, 3])

    def test_evaluation_and_per_stage_printouts(self):
        """The scene-scope evaluation must print as soon as the diff is
        computed, and each downstream stage must announce what it's about
        to do — both requested directly so an operator can see the plan
        without waiting for every stage to finish."""
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            current_scenes = json.loads((out / "scenes.json").read_text())
            edited_scenes = {"scenes": current_scenes["scenes"] + [
                {"number": 4, "location": "Cafe", "characters": ["Marcel"], "summary": "s4"},
            ]}

            buf = io.StringIO()
            with mock.patch("reel.stages.run_stage", return_value={}), \
                 mock.patch("reel.pipeline._assemble_movie", return_value=None), \
                 redirect_stdout(buf):
                applied = cli._revise_one("scenes", out, edited_override=edited_scenes,
                                          auto_confirm=True, render_images=True, render_video=True)
            self.assertTrue(applied)
            output = buf.getvalue()

            eval_line = next((ln for ln in output.splitlines()
                             if ln.startswith("[reel] evaluation")), None)
            self.assertIsNotNone(eval_line, "no evaluation line printed")
            self.assertIn("added=[4]", eval_line)
            self.assertIn("changed=[]", eval_line)
            self.assertIn("removed=[]", eval_line)
            # printed BEFORE the final "plan:" confirm line
            self.assertLess(output.index("evaluation"), output.index("[reel] plan:"))

            self.assertIn("[reel]   soundscape: regenerating [4]", output)
            self.assertIn("[reel]   casting_images: re-rendering (image provider)", output)


if __name__ == "__main__":
    unittest.main()
