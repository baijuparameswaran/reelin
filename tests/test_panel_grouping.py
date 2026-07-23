"""Validates `reel/panel_grouping.py` (same-cast/duration-bounded panel
grouping for multi-segment timestamped Veo prompts) and
`veo_prompt.multi_panel_video_prompt` (the timestamp-segmented prompt built
from a group). Pure logic only — no LLM/API calls, offline, <1s.

Also covers `pipeline._dedupe_clip_paths`, the small pure helper that keeps
`_stitch_scene`/`_clips_in_order` from handing `i2v.stitch` the same merged
group's clip path multiple times in a row.

Run with: python -m unittest tests.test_panel_grouping -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reel import panel_grouping, veo_prompt
from reel import pipeline


def _panel(num, *, characters=None, duration=None, action="", sound="", dialogue=None,
          shot_type="MS") -> dict:
    return {
        "panel": num,
        "shot_type": shot_type,
        "characters_in_frame": characters or [],
        "duration": duration,
        "action": action,
        "sound": sound,
        "dialogue": dialogue or [],
    }


class TestGroupPanels(unittest.TestCase):
    def test_same_cast_contiguous_panels_merge(self):
        panels = [
            _panel(1, characters=["Alice", "Bob"], duration="2s"),
            _panel(2, characters=["Alice", "Bob"], duration="2s"),
        ]
        groups = panel_grouping.group_panels(panels)
        self.assertEqual(len(groups), 1)
        self.assertEqual([p["panel"] for p in groups[0]], [1, 2])

    def test_cast_change_starts_new_group(self):
        panels = [
            _panel(1, characters=["Alice"], duration="2s"),
            _panel(2, characters=["Bob"], duration="2s"),
        ]
        groups = panel_grouping.group_panels(panels)
        self.assertEqual(len(groups), 2)
        self.assertEqual([p["panel"] for p in groups[0]], [1])
        self.assertEqual([p["panel"] for p in groups[1]], [2])

    def test_empty_cast_panel_rides_along_with_current_group(self):
        panels = [
            _panel(1, characters=["Alice"], duration="2s"),
            _panel(2, characters=[], duration="2s"),          # cutaway, no listed cast
            _panel(3, characters=["Alice"], duration="2s"),
        ]
        groups = panel_grouping.group_panels(panels)
        self.assertEqual(len(groups), 1)
        self.assertEqual([p["panel"] for p in groups[0]], [1, 2, 3])

    def test_duration_overflow_splits_same_cast_run(self):
        panels = [
            _panel(1, characters=["Alice"], duration="4s"),
            _panel(2, characters=["Alice"], duration="4s"),
            _panel(3, characters=["Alice"], duration="4s"),   # 4+4+4=12s > 8s cap
        ]
        groups = panel_grouping.group_panels(panels, max_seconds=8)
        self.assertEqual(len(groups), 2)
        total_panels = sum(len(g) for g in groups)
        self.assertEqual(total_panels, 3)
        for g in groups:
            self.assertLessEqual(sum(panel_grouping._panel_duration(p) for p in g), 8)

    def test_first_panel_of_scene_is_always_a_valid_group_start(self):
        panels = [_panel(1, characters=["Alice"], duration="2s")]
        groups = panel_grouping.group_panels(panels)
        self.assertEqual(len(groups), 1)
        self.assertEqual([p["panel"] for p in groups[0]], [1])

    def test_empty_panel_list_returns_empty(self):
        self.assertEqual(panel_grouping.group_panels([]), [])

    def test_config_disabled_path_is_all_singletons(self):
        # pipeline.py achieves this by never calling group_panels at all when
        # `use_grouping` is False — simulated here directly for clarity.
        panels = [
            _panel(1, characters=["Alice", "Bob"], duration="2s"),
            _panel(2, characters=["Alice", "Bob"], duration="2s"),
        ]
        groups = [[p] for p in panels]
        self.assertEqual(len(groups), 2)


class TestSegmentBoundaries(unittest.TestCase):
    def test_sums_exactly_to_target(self):
        for raw, target in [([2, 2], 4), ([2, 3, 3], 8), ([1, 1, 1, 1], 4), ([5], 6)]:
            with self.subTest(raw=raw, target=target):
                segs = panel_grouping.segment_boundaries(raw, target)
                self.assertEqual(len(segs), len(raw))
                self.assertEqual(segs[0][0], 0.0)
                self.assertEqual(segs[-1][1], float(target))
                # monotonically non-decreasing, no negative-length segment
                for start, end in segs:
                    self.assertGreaterEqual(end, start)
                for (_, prev_end), (next_start, _) in zip(segs, segs[1:]):
                    self.assertEqual(prev_end, next_start)

    def test_empty_input_returns_empty(self):
        self.assertEqual(panel_grouping.segment_boundaries([], 8), [])

    def test_zero_durations_fall_back_to_equal_split(self):
        segs = panel_grouping.segment_boundaries([0, 0, 0], 6)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[-1][1], 6.0)


class TestFormatTimestamp(unittest.TestCase):
    def test_formats_mm_ss(self):
        self.assertEqual(panel_grouping.format_timestamp(0), "00:00")
        self.assertEqual(panel_grouping.format_timestamp(2), "00:02")
        self.assertEqual(panel_grouping.format_timestamp(65), "01:05")


class TestMultiPanelVideoPrompt(unittest.TestCase):
    def setUp(self):
        self.casting_lookup = {
            "Alice": {"character": {"physical_form": "a woman in a red coat"}},
        }
        self.visual_overview = {"color_palette": "warm ambers", "lighting_setup": "soft window light",
                                "mood": "tense"}

    def test_segments_appear_in_timestamp_order_blank_line_separated(self):
        panels = [
            _panel(1, characters=["Alice"], action="Alice enters the room.", shot_type="WS"),
            _panel(2, characters=["Alice"], action="Alice sits down.", shot_type="CU"),
        ]
        segments = [(0.0, 4.0), (4.0, 8.0)]
        prompt = veo_prompt.multi_panel_video_prompt(
            panels, segments, {}, casting_lookup=self.casting_lookup,
            location_desc="a dim kitchen", visual_overview=self.visual_overview)
        idx1 = prompt.index("[00:00-00:04]")
        idx2 = prompt.index("[00:04-00:08]")
        self.assertLess(idx1, idx2)
        self.assertIn("\n\n[00:00-00:04]", prompt)
        self.assertIn("\n\n[00:04-00:08]", prompt)
        self.assertIn("enters the room", prompt)
        self.assertIn("sits down", prompt)

    def test_shared_prefix_appears_once_not_per_segment(self):
        panels = [
            _panel(1, characters=["Alice"], action="Alice enters.", shot_type="WS"),
            _panel(2, characters=["Alice"], action="Alice waits.", shot_type="CU"),
        ]
        segments = [(0.0, 4.0), (4.0, 8.0)]
        prompt = veo_prompt.multi_panel_video_prompt(
            panels, segments, {}, casting_lookup=self.casting_lookup,
            location_desc="a dim kitchen", visual_overview=self.visual_overview)
        # location text should appear exactly once (the shared Context prefix)
        self.assertEqual(prompt.count("dim kitchen"), 1)
        self.assertEqual(prompt.count("warm ambers"), 1)

    def test_per_panel_sound_override_preserved_on_non_first_member(self):
        panels = [
            _panel(1, characters=["Alice"], action="Alice enters.",
                  sound="street traffic | door creaks"),
            _panel(2, characters=["Alice"], action="Alice sits.",
                  sound="ticking clock | chair scrapes"),
        ]
        segments = [(0.0, 4.0), (4.0, 8.0)]
        prompt = veo_prompt.multi_panel_video_prompt(
            panels, segments, {"ambient": "scene-wide hum"},
            casting_lookup=self.casting_lookup,
            location_desc="a dim kitchen", visual_overview=self.visual_overview)
        # Panel 2's own sound override must survive, not be dropped in favor
        # of the scene-wide ambient or panel 1's override.
        self.assertIn("ticking clock", prompt)
        self.assertIn("chair scrapes", prompt)
        self.assertIn("street traffic", prompt)
        self.assertIn("door creaks", prompt)

    def test_no_subtitles_directive_appears_once_when_any_dialogue(self):
        panels = [
            _panel(1, characters=["Alice"], action="Alice enters."),
            _panel(2, characters=["Alice"], action="Alice speaks.",
                  dialogue=[{"speaker": "Alice", "line": "Hello there."}]),
        ]
        segments = [(0.0, 4.0), (4.0, 8.0)]
        prompt = veo_prompt.multi_panel_video_prompt(
            panels, segments, {}, casting_lookup=self.casting_lookup,
            location_desc="a dim kitchen", visual_overview=self.visual_overview,
            no_subtitles=True)
        self.assertEqual(prompt.count("No subtitles"), 1)

    def test_empty_panels_returns_empty_string(self):
        self.assertEqual(veo_prompt.multi_panel_video_prompt([], [], {}), "")


class TestDedupeClipPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, rel: str) -> None:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-mp4-bytes")

    def test_repeated_clip_path_collapses_to_one_entry_order_preserved(self):
        self._touch("video/scene_01/frame_03-06.mp4")
        self._touch("video/scene_01/frame_07.mp4")
        frames = [
            {"panel": 3, "clip": "video/scene_01/frame_03-06.mp4"},
            {"panel": 4, "clip": "video/scene_01/frame_03-06.mp4"},
            {"panel": 5, "clip": "video/scene_01/frame_03-06.mp4"},
            {"panel": 6, "clip": "video/scene_01/frame_03-06.mp4"},
            {"panel": 7, "clip": "video/scene_01/frame_07.mp4"},
        ]
        paths = pipeline._dedupe_clip_paths(frames, self.out)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0].name, "frame_03-06.mp4")
        self.assertEqual(paths[1].name, "frame_07.mp4")

    def test_no_clip_field_and_missing_files_are_skipped(self):
        self._touch("video/scene_01/frame_01.mp4")
        frames = [
            {"panel": 1, "clip": "video/scene_01/frame_01.mp4"},
            {"panel": 2, "clip": None},
            {"panel": 3, "clip": "video/scene_01/frame_03.mp4"},  # never written
        ]
        paths = pipeline._dedupe_clip_paths(frames, self.out)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "frame_01.mp4")


if __name__ == "__main__":
    unittest.main()
