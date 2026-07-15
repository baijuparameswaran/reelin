"""Validates that `storyboard._panel_characters_in_frame` narrows a panel's
character list down to whoever that panel's own action/dialogue text
actually names, instead of defaulting to the ENTIRE scene cast for every
wide/establishing/full shot (or a close shot with no dialogue) regardless of
who's actually depicted. This is the SOURCE-level fix — `characters_in_frame`
itself is now accurate for every downstream consumer (Subject text, dialogue
attribution, Veo reference-image selection), not narrowed after the fact
only at render time (see `pipeline._panel_relevant_characters`, the earlier,
narrower fix this generalizes). See PROGRESS.md's 2026-07-14 session log.

Deterministic — no LLM calls anywhere (the storyboard build path stays
LLM-free by default, per the module's own "BUILD PATH" docstring).

Run with: python -m unittest tests.test_storyboard_characters_in_frame -v
"""
from __future__ import annotations

import unittest

from reel.agents import storyboard


class TestPanelCharactersInFrame(unittest.TestCase):
    def test_wide_shot_narrows_to_characters_named_in_action(self):
        cam = {"type": "WS"}
        result = storyboard._panel_characters_in_frame(
            cam, [], ["Alice", "Bob", "Carol", "Dave"],
            action="Alice and Bob argue quietly by the window.")
        self.assertEqual(result, ["Alice", "Bob"])

    def test_wide_shot_falls_back_to_full_cast_when_action_names_no_one(self):
        cam = {"type": "WS"}
        result = storyboard._panel_characters_in_frame(
            cam, [], ["Alice", "Bob", "Carol"], action="The crowd gathers.")
        self.assertEqual(result, ["Alice", "Bob", "Carol"])

    def test_wide_shot_falls_back_to_full_cast_with_no_action_text_at_all(self):
        cam = {"type": "WS"}
        result = storyboard._panel_characters_in_frame(cam, [], ["Alice", "Bob"], action="")
        self.assertEqual(result, ["Alice", "Bob"])

    def test_close_shot_with_dialogue_still_prefers_the_speaker(self):
        # Unchanged priority: a close-up follows the speaker even if the
        # action text (rarely populated for a reaction shot) names someone else.
        cam = {"type": "CU"}
        result = storyboard._panel_characters_in_frame(
            cam, [{"speaker": "Carol", "line": "Hi"}], ["Alice", "Bob", "Carol"],
            action="Alice watches from across the room.")
        self.assertEqual(result, ["Carol"])

    def test_close_shot_with_no_dialogue_now_also_narrows_by_action_text(self):
        # An improvement over the old behavior (which always defaulted to
        # the full cast for a dialogue-free close shot).
        cam = {"type": "CU"}
        result = storyboard._panel_characters_in_frame(
            cam, [], ["Alice", "Bob"], action="Bob clenches his fist.")
        self.assertEqual(result, ["Bob"])

    def test_dialogue_speaker_counts_as_relevant_for_a_wide_shot_too(self):
        cam = {"type": "WS"}
        result = storyboard._panel_characters_in_frame(
            cam, [{"speaker": "Carol", "line": "Wait!"}], ["Alice", "Bob", "Carol"],
            action="Someone shouts across the yard.")
        self.assertEqual(result, ["Carol"])

    def test_preserves_scene_cast_order_not_match_order(self):
        cam = {"type": "WS"}
        result = storyboard._panel_characters_in_frame(
            cam, [], ["Alice", "Bob", "Carol"], action="Bob turns to Alice.")
        self.assertEqual(result, ["Alice", "Bob"])

    def test_word_boundary_avoids_false_positive_substring_match(self):
        cam = {"type": "WS"}
        result = storyboard._panel_characters_in_frame(
            cam, [], ["Bob", "Alice"], action="Bobby waves while Alice smiles.")
        self.assertEqual(result, ["Alice"])


class TestBuildPanelIntegration(unittest.TestCase):
    """Confirms the narrowing actually reaches _build_panel's final
    `characters_in_frame` output, not just the isolated helper."""

    def _bundle(self):
        return {
            "scene_number": 1, "slugline": "EXT. PARK - DAY",
            "slugline_parsed": {"int_ext": "EXT", "location": "PARK", "time_of_day": "DAY"},
            "summary": "s", "purpose": "p",
            "characters_in_scene": ["Alice", "Bob", "Carol", "Dave"],
            "cast": [{"name": n, "physical_form": ""} for n in
                    ("Alice", "Bob", "Carol", "Dave")],
            "location": {"name": "PARK", "visual_prompt": "a sunny park"},
            "art": {"color_palette": "", "lighting": "", "visual_filter": "",
                    "key_props": [], "visual_moments": [], "emotional_function": ""},
            "audio": {"ambient_bed": "", "silence": False, "sound_events": [],
                      "score_direction": "", "emotional_function": ""},
            "camera": {"coverage": "", "transition_to_next": "", "shots": [
                {"shot_number": 1, "moment": "Alice and Bob argue by the bench.",
                 "type": "WS", "angle": "EYE LEVEL", "movement": "STATIC",
                 "lens": "24mm", "framing": "wide", "emotional_function": ""},
            ]},
            "screenplay_shots": [{"shot": 1, "shot_type": "WS",
                                  "description": "Alice and Bob argue by the bench.",
                                  "voiceover": None, "dialogue": [], "sound": ""}],
            "moodboard_tile": None,
        }

    def test_full_scene_board_narrows_wide_shot_panel(self):
        board = storyboard._build_scene_board(self._bundle())
        panel = board["panels"][0]
        self.assertEqual(panel["characters_in_frame"], ["Alice", "Bob"])
        self.assertNotIn("Carol", panel["characters_in_frame"])
        self.assertNotIn("Dave", panel["characters_in_frame"])


if __name__ == "__main__":
    unittest.main()
