"""Validates `veo_prompt.panel_action` — folding a storyboard panel's
authored `emotional_note` ("the emotion this panel must evoke in the
audience", per-shot creative direction from the cinematography/visuals/
soundscape agents) into the Veo [Action] element as a short performance-
direction clause, instead of the field being authored, shown at the review
gate, and then silently dropped before ever reaching the actual render.

Pure logic — no live Ollama/Gemini/Veo calls anywhere.

Run with: python -m unittest tests.test_panel_action_emotional_note -v
"""
from __future__ import annotations

import unittest

from reel import veo_prompt


class TestPanelAction(unittest.TestCase):
    def test_folds_emotional_note_into_action_as_conveying_clause(self):
        panel = {"action": "Alice reaches for the doorknob",
                 "emotional_note": "quiet dread"}
        result = veo_prompt.panel_action(panel)
        self.assertEqual(result, "Alice reaches for the doorknob, conveying quiet dread")

    def test_no_emotional_note_returns_action_unchanged(self):
        panel = {"action": "Alice reaches for the doorknob"}
        self.assertEqual(veo_prompt.panel_action(panel), "Alice reaches for the doorknob")

    def test_empty_emotional_note_returns_action_unchanged(self):
        panel = {"action": "Alice reaches for the doorknob", "emotional_note": "  "}
        self.assertEqual(veo_prompt.panel_action(panel), "Alice reaches for the doorknob")

    def test_falls_back_to_moment_field_when_no_action(self):
        panel = {"moment": "the door creaks open", "emotional_note": "tension"}
        self.assertEqual(veo_prompt.panel_action(panel),
                         "the door creaks open, conveying tension")

    def test_no_action_or_moment_falls_back_to_bare_note(self):
        panel = {"emotional_note": "sudden hope"}
        self.assertEqual(veo_prompt.panel_action(panel), "sudden hope")

    def test_empty_panel_returns_empty_string(self):
        self.assertEqual(veo_prompt.panel_action({}), "")

    def test_note_already_present_in_action_is_not_duplicated(self):
        panel = {"action": "Alice reaches for the doorknob with quiet dread",
                 "emotional_note": "quiet dread"}
        self.assertEqual(veo_prompt.panel_action(panel),
                         "Alice reaches for the doorknob with quiet dread")

    def test_note_case_insensitive_dedup(self):
        panel = {"action": "Alice reaches for the doorknob, filled with Quiet Dread.",
                 "emotional_note": "quiet dread"}
        result = veo_prompt.panel_action(panel)
        self.assertEqual(result.lower().count("dread"), 1)
        self.assertNotIn("conveying", result)

    def test_lowercases_first_letter_of_note_when_joined(self):
        panel = {"action": "Alice turns", "emotional_note": "Bittersweet nostalgia"}
        result = veo_prompt.panel_action(panel)
        self.assertEqual(result, "Alice turns, conveying bittersweet nostalgia")

    def test_single_character_note_lowercased(self):
        panel = {"action": "Alice turns", "emotional_note": "X"}
        self.assertEqual(veo_prompt.panel_action(panel), "Alice turns, conveying x")


class TestFivePartPromptIncludesEmotionalNote(unittest.TestCase):
    def test_five_part_prompt_carries_emotional_note_in_action_position(self):
        panel = {"panel": 1, "shot_type": "CU", "characters_in_frame": ["Alice"],
                 "action": "Alice reaches for the doorknob",
                 "emotional_note": "quiet dread"}
        casting_lookup = {"Alice": {"character": {"physical_form": "a woman in a red coat"}}}
        prompt = veo_prompt.five_part_veo_prompt(
            panel, casting_lookup=casting_lookup, location_desc="an old house",
            visual_overview={"mood": "ominous"})
        self.assertIn("conveying quiet dread", prompt)

    def test_panel_video_prompt_carries_emotional_note(self):
        panel = {"panel": 1, "shot_type": "CU", "characters_in_frame": ["Alice"],
                 "action": "Alice reaches for the doorknob",
                 "emotional_note": "quiet dread"}
        casting_lookup = {"Alice": {"character": {"physical_form": "a woman in a red coat"}}}
        prompt = veo_prompt.panel_video_prompt(
            panel, {}, casting_lookup=casting_lookup, location_desc="an old house",
            visual_overview={"mood": "ominous"})
        self.assertIn("conveying quiet dread", prompt)

    def test_multi_panel_video_prompt_carries_per_segment_emotional_notes(self):
        panels = [
            {"panel": 1, "characters_in_frame": ["Alice"], "shot_type": "WS",
             "action": "Alice enters the room", "emotional_note": "unease"},
            {"panel": 2, "characters_in_frame": ["Alice"], "shot_type": "CU",
             "action": "Alice freezes", "emotional_note": "sudden fear"},
        ]
        casting_lookup = {"Alice": {"character": {"physical_form": "a woman in a red coat"}}}
        segments = [(0.0, 4.0), (4.0, 8.0)]
        prompt = veo_prompt.multi_panel_video_prompt(
            panels, segments, {}, casting_lookup=casting_lookup,
            location_desc="an old house", visual_overview={"mood": "ominous"})
        self.assertIn("conveying unease", prompt)
        self.assertIn("conveying sudden fear", prompt)


if __name__ == "__main__":
    unittest.main()
