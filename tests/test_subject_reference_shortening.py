"""Validates that a Veo Subject element skips re-describing a character in
words when an actual seed/reference image of them is already being sent in
the same API call — Veo grounds appearance from the image, so restating the
full `physical_form` text is redundant (and risks the text and image
disagreeing). A character with no resolvable image this call still gets the
full locked description, since for them the text IS the only grounding Veo
has. See `veo_prompt.anchored_character_names`/`veo_prompt.panel_subject`.

Pure logic — no live Ollama/Gemini/Veo calls anywhere.

Run with: python -m unittest tests.test_subject_reference_shortening -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reel import veo_prompt


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-png-bytes")
    return p


class TestAnchoredCharacterNames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.alice_png = _touch(self.out / "casting/alice.png")
        self.bob_png = _touch(self.out / "casting/bob.png")
        self.casting_lookup = {
            "Alice": {"character": {"physical_form": "red hair", "image_path": "casting/alice.png"}},
            "Bob": {"character": {"physical_form": "bald", "image_path": "casting/bob.png"}},
            "Carol": {"character": {"physical_form": "blonde", "image_path": "casting/carol.png"}},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_seed_no_references_anchors_nobody(self):
        anchored = veo_prompt.anchored_character_names(
            ["Alice", "Bob"], self.casting_lookup, self.out, None, None)
        self.assertEqual(anchored, set())

    def test_seed_matching_a_casting_portrait_anchors_only_that_character(self):
        anchored = veo_prompt.anchored_character_names(
            ["Alice", "Bob"], self.casting_lookup, self.out, self.alice_png, None)
        self.assertEqual(anchored, {"Alice"})

    def test_reference_images_anchor_every_matched_character(self):
        anchored = veo_prompt.anchored_character_names(
            ["Alice", "Bob", "Carol"], self.casting_lookup, self.out,
            None, [self.alice_png, self.bob_png])
        self.assertEqual(anchored, {"Alice", "Bob"})

    def test_carol_unresolvable_portrait_never_anchored(self):
        # Carol's casting.json image_path points at a file never rendered to
        # disk — she can never be "shown" in an image that doesn't exist.
        anchored = veo_prompt.anchored_character_names(
            ["Carol"], self.casting_lookup, self.out, None, None)
        self.assertEqual(anchored, set())

    def test_continuity_tail_frame_seed_anchors_every_listed_character(self):
        # A seed that ISN'T any character's casting portrait (e.g. the
        # previous clip's tail frame) is presumed to still show every
        # currently in-frame character, since continuity is only used when
        # the cast hasn't changed panel to panel.
        tail = _touch(self.out / "video/scene_01/frame_01_tail.png")
        anchored = veo_prompt.anchored_character_names(
            ["Alice", "Bob"], self.casting_lookup, self.out, tail, None)
        self.assertEqual(anchored, {"Alice", "Bob"})

    def test_empty_names_list_anchors_nobody(self):
        anchored = veo_prompt.anchored_character_names(
            [], self.casting_lookup, self.out, self.alice_png, None)
        self.assertEqual(anchored, set())


class TestPanelSubjectShortening(unittest.TestCase):
    def setUp(self):
        self.casting_lookup = {
            "Alice": {"character": {"physical_form": "tall, red curly hair, green coat"}},
            "Bob": {"character": {"physical_form": "short, bald, brown jacket"}},
        }
        self.panel = {"characters_in_frame": ["Alice", "Bob"]}

    def test_anchored_character_gets_referential_text_not_full_description(self):
        subject = veo_prompt.panel_subject(self.panel, self.casting_lookup, {"Alice"})
        self.assertIn("Alice (as shown in the reference image)", subject)
        self.assertNotIn("red curly hair", subject)

    def test_unanchored_character_still_gets_full_physical_form(self):
        subject = veo_prompt.panel_subject(self.panel, self.casting_lookup, {"Alice"})
        self.assertIn("Bob (short, bald, brown jacket)", subject)

    def test_no_anchored_set_falls_back_to_full_description_for_everyone(self):
        # None (the default) matches this function's pre-existing behavior —
        # every pre-existing caller with no image context is unaffected.
        subject = veo_prompt.panel_subject(self.panel, self.casting_lookup)
        self.assertIn("red curly hair", subject)
        self.assertIn("brown jacket", subject)

    def test_uncast_character_falls_back_to_bare_name_regardless_of_anchoring(self):
        panel = {"characters_in_frame": ["Ghost"]}
        subject = veo_prompt.panel_subject(panel, self.casting_lookup, {"Ghost"})
        self.assertEqual(subject, "Ghost")


class TestFivePartPromptIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.alice_png = _touch(self.out / "casting/alice.png")
        self.casting_lookup = {
            "Alice": {"character": {"physical_form": "tall, red curly hair, green coat",
                                    "image_path": "casting/alice.png"}},
        }
        self.panel = {
            "characters_in_frame": ["Alice"],
            "shot_type": "WS", "camera_angle": "eye level", "camera_movement": "static",
            "action": "She walks across the room.",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_panel_video_prompt_shortens_subject_when_seed_matches_portrait(self):
        prompt = veo_prompt.panel_video_prompt(
            self.panel, {}, casting_lookup=self.casting_lookup,
            location_desc="a quiet room", visual_overview={"mood": "calm"},
            out=self.out, seed=self.alice_png, reference_images=None)
        self.assertIn("as shown in the reference image", prompt)
        self.assertNotIn("red curly hair", prompt)

    def test_panel_video_prompt_keeps_full_description_with_no_image_context(self):
        # Omitting out/seed/reference_images (their defaults) reproduces the
        # exact pre-existing behavior for callers with no image context.
        prompt = veo_prompt.panel_video_prompt(
            self.panel, {}, casting_lookup=self.casting_lookup,
            location_desc="a quiet room", visual_overview={"mood": "calm"})
        self.assertIn("red curly hair", prompt)

    def test_veo_verify_prompt_still_passes_on_shortened_subject(self):
        from reel import veo_guide
        prompt = veo_prompt.panel_video_prompt(
            self.panel, {}, casting_lookup=self.casting_lookup,
            location_desc="a quiet room", visual_overview={"mood": "calm"},
            out=self.out, seed=self.alice_png, reference_images=None)
        report = veo_guide.verify_prompt(prompt)
        self.assertTrue(report["valid"], report["issues"])


if __name__ == "__main__":
    unittest.main()
