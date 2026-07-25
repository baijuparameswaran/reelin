"""Validates `pipeline._boundary_aware_seed` — the fix for a real,
previously-documented gap (PROGRESS.md's "Known deferred issue"): seed
selection wasn't boundary-aware the way `effective_prev_clip` already was,
so a character "shot boundary" panel (a new character entering, or an
outgoing one leaving) could seed from the PREVIOUS panel's tail frame —
which still shows the outgoing cast — instead of a fresh identity anchor
for the character(s) actually in this panel.

Pure logic — no live Ollama/Gemini/Veo calls anywhere.

Run with: python -m unittest tests.test_boundary_aware_seed -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reel import pipeline


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-png-bytes")
    return p


class TestBoundaryAwareSeed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.cast_index = {"Alice": "casting/alice.png", "Bob": "casting/bob.png"}
        _touch(self.out / self.cast_index["Alice"])
        _touch(self.out / self.cast_index["Bob"])
        self.prev_tail = _touch(self.out / "video/scene_01/frame_01_tail.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_non_boundary_panel_continues_from_prev_tail(self):
        fr = {"characters_in_frame": ["Alice"]}
        seed = pipeline._boundary_aware_seed(
            fr, is_boundary=False, prev_tail=self.prev_tail, continuity=True,
            cast_index=self.cast_index, out=self.out)
        self.assertEqual(seed, self.prev_tail)

    def test_boundary_panel_seeds_fresh_even_though_prev_tail_exists(self):
        # The exact bug this fixes: a new character (Bob) enters, but a
        # perfectly valid prev_tail (showing Alice alone) is available —
        # the seed must NOT reuse it.
        fr = {"characters_in_frame": ["Alice", "Bob"]}
        seed = pipeline._boundary_aware_seed(
            fr, is_boundary=True, prev_tail=self.prev_tail, continuity=True,
            cast_index=self.cast_index, out=self.out)
        self.assertNotEqual(seed, self.prev_tail)
        self.assertEqual(seed, self.out / "casting/alice.png")

    def test_outgoing_character_leaving_also_seeds_fresh(self):
        # Bob leaves, Alice remains alone — still a boundary (cast SET
        # changed), still must not reuse the two-character tail frame.
        fr = {"characters_in_frame": ["Alice"]}
        seed = pipeline._boundary_aware_seed(
            fr, is_boundary=True, prev_tail=self.prev_tail, continuity=True,
            cast_index=self.cast_index, out=self.out)
        self.assertNotEqual(seed, self.prev_tail)
        self.assertEqual(seed, self.out / "casting/alice.png")

    def test_no_prev_tail_always_seeds_fresh_regardless_of_boundary(self):
        fr = {"characters_in_frame": ["Bob"]}
        seed = pipeline._boundary_aware_seed(
            fr, is_boundary=False, prev_tail=None, continuity=True,
            cast_index=self.cast_index, out=self.out)
        self.assertEqual(seed, self.out / "casting/bob.png")

    def test_continuity_disabled_always_seeds_fresh_even_off_boundary(self):
        fr = {"characters_in_frame": ["Alice"]}
        seed = pipeline._boundary_aware_seed(
            fr, is_boundary=False, prev_tail=self.prev_tail, continuity=False,
            cast_index=self.cast_index, out=self.out)
        self.assertNotEqual(seed, self.prev_tail)
        self.assertEqual(seed, self.out / "casting/alice.png")

    def test_boundary_with_no_resolvable_portrait_returns_none(self):
        fr = {"characters_in_frame": ["Nobody Cast"]}
        seed = pipeline._boundary_aware_seed(
            fr, is_boundary=True, prev_tail=self.prev_tail, continuity=True,
            cast_index=self.cast_index, out=self.out)
        self.assertIsNone(seed)


if __name__ == "__main__":
    unittest.main()
