"""Validates the `rerender_panels` half of the boundary-aware-seed fix
(see `tests/test_boundary_aware_seed.py` for the `_render_scene_frames`
half, fixed first). `rerender_panels`' own `_resolve_start_frame` had the
SAME gap `_boundary_aware_seed` closed there: it always reused the
immediately-preceding panel's recorded `end_frame`/tail PNG as the seed,
regardless of whether the in-frame cast changed — so a targeted re-render
of a character "shot boundary" panel (someone entering OR leaving) could
seed from a stale frame that doesn't show the panel's actual cast. This
matters most for a single-character boundary panel, where
`_resolve_panel_references` never kicks in (it requires 2+ resolvable
portraits), so the seed path is the ONLY identity anchor available.

End-to-end (mocked i2v) — no live Ollama/Gemini/Veo calls anywhere.

Run with: python -m unittest tests.test_rerender_panels_boundary_seed -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import i2v, pipeline


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-png-bytes")
    return p


class TestRerenderPanelsSeedIsBoundaryAware(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.sdir = self.out / "video" / "scene_01"
        self.sdir.mkdir(parents=True)
        self.alice = _touch(self.out / "casting" / "alice.png")
        self.bob = _touch(self.out / "casting" / "bob.png")

        # 3-panel scene: panel 1 = Alice+Bob together; panel 2 = Bob ALONE
        # (Alice leaves -> cast SET changes -> boundary, even though nobody
        # new enters); panel 3 = Bob alone again (same cast as panel 2 ->
        # not a boundary). Pre-existing manifest/clip files simulate an
        # already-rendered scene, since rerender_panels requires one.
        # Panel 1's recorded tail frame deliberately shows BOTH characters —
        # the stale frame the bug would wrongly reuse for panel 2.
        for n in (1, 2, 3):
            (self.sdir / f"frame_{n:02d}.mp4").write_bytes(b"old-clip")
            (self.sdir / f"frame_{n:02d}_tail.png").write_bytes(b"old-tail")

        (self.out / "video" / "manifest.json").write_text(json.dumps({
            "clips": 3, "failed": 0,
            "scenes": [{"scene_number": 1, "scene_video": "video/scene_01.mp4", "frames": [
                {"panel": 1, "clip": "video/scene_01/frame_01.mp4",
                 "end_frame": "video/scene_01/frame_01_tail.png"},
                {"panel": 2, "clip": "video/scene_01/frame_02.mp4",
                 "end_frame": "video/scene_01/frame_02_tail.png"},
                {"panel": 3, "clip": "video/scene_01/frame_03.mp4",
                 "end_frame": "video/scene_01/frame_03_tail.png"},
            ]}],
        }))

        self.storyboard = {"storyboard": [{
            "scene_number": 1, "header": {"location": ""}, "audio_overview": {},
            "visual_overview": {},
            "panels": [
                {"panel": 1, "characters_in_frame": ["Alice", "Bob"], "action": "together"},
                {"panel": 2, "characters_in_frame": ["Bob"], "action": "Bob alone"},
                {"panel": 3, "characters_in_frame": ["Bob"], "action": "Bob still alone"},
            ],
        }]}
        self.casting = {"casting": [
            {"name": "Alice", "character": {"image_path": "casting/alice.png"}},
            {"name": "Bob", "character": {"image_path": "casting/bob.png"}},
        ]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_targeted_boundary_panel_seeds_fresh_not_from_stale_tail(self):
        calls = {}

        def fake_generate_clip(images, prompt, clip, *, prev_clip=None,
                               duration_seconds=None, reference_images=None,
                               dry_run=False):
            calls[clip.name] = list(images)
            clip.write_bytes(b"new-clip")
            return True

        with mock.patch.object(i2v, "generate_clip", side_effect=fake_generate_clip), \
             mock.patch.object(i2v, "overlays_enabled", return_value=False), \
             mock.patch.object(i2v, "last_frame", return_value=None), \
             mock.patch.object(i2v, "stitch", return_value=True):
            pipeline.rerender_panels(self.storyboard, self.casting, self.out,
                                     scene_number=1, panel_numbers=[2])

        # Panel 2 is targeted and is a boundary (Alice leaves) — must seed
        # from Bob's own casting portrait (_frame_char_anchor), NOT from
        # panel 1's stale tail frame (which still shows Alice+Bob together).
        self.assertEqual(calls.get("frame_02.mp4"), [self.bob])

        # Panel 3 (one-hop cascade target) is NOT a boundary relative to
        # panel 2 (same cast, Bob alone) — must still chain from the
        # (now-updated) previous panel's tail frame, unchanged behavior.
        self.assertEqual(calls.get("frame_03.mp4"),
                         [self.sdir / "frame_02_tail.png"])

    def test_non_boundary_targeted_panel_still_chains_normally(self):
        # Retarget panel 3 directly (not via cascade) — its cast matches
        # panel 2's, so it's NOT a boundary and must keep chaining from the
        # previous panel's recorded tail frame exactly as before this fix.
        calls = {}

        def fake_generate_clip(images, prompt, clip, *, prev_clip=None,
                               duration_seconds=None, reference_images=None,
                               dry_run=False):
            calls[clip.name] = list(images)
            clip.write_bytes(b"new-clip")
            return True

        with mock.patch.object(i2v, "generate_clip", side_effect=fake_generate_clip), \
             mock.patch.object(i2v, "overlays_enabled", return_value=False), \
             mock.patch.object(i2v, "last_frame", return_value=None), \
             mock.patch.object(i2v, "stitch", return_value=True):
            pipeline.rerender_panels(self.storyboard, self.casting, self.out,
                                     scene_number=1, panel_numbers=[3])

        self.assertEqual(calls.get("frame_03.mp4"),
                         [self.sdir / "frame_02_tail.png"])

    def test_targeted_first_panel_of_scene_still_seeds_from_anchor(self):
        # pnum == first_panel_num short-circuits regardless of is_boundary —
        # confirms that branch of _resolve_start_frame is untouched by the fix.
        calls = {}

        def fake_generate_clip(images, prompt, clip, *, prev_clip=None,
                               duration_seconds=None, reference_images=None,
                               dry_run=False):
            calls[clip.name] = list(images)
            clip.write_bytes(b"new-clip")
            return True

        with mock.patch.object(i2v, "generate_clip", side_effect=fake_generate_clip), \
             mock.patch.object(i2v, "overlays_enabled", return_value=False), \
             mock.patch.object(i2v, "last_frame", return_value=None), \
             mock.patch.object(i2v, "stitch", return_value=True):
            pipeline.rerender_panels(self.storyboard, self.casting, self.out,
                                     scene_number=1, panel_numbers=[1])

        # Panel 1 has both Alice and Bob; _frame_char_anchor picks the
        # first RELEVANT name it can resolve — just confirm it's a fresh
        # casting portrait, not a tail frame (there is none before panel 1).
        self.assertIn(calls.get("frame_01.mp4")[0], (self.alice, self.bob))


if __name__ == "__main__":
    unittest.main()
