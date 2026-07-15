"""Validates that a storyboard panel's Veo reference/seed images are scoped
to only the character(s) actually relevant to THAT panel, not the raw
`characters_in_frame` list — which, per `storyboard._panel_characters_in_frame`'s
own docstring ("HEURISTIC, not a citation"), defaults to the ENTIRE scene
cast for any wide/establishing/full shot (or a close shot with no dialogue),
regardless of who that specific panel's action text actually depicts.
Feeding that unfiltered list into reference-image selection wasted Veo's
3-reference budget (or the single-seed slot) on characters technically in
the scene but absent from the panel, at best, or produced a wrong
identity-lock at worst. See `pipeline._panel_relevant_characters`.

Pure logic — no live Ollama/Gemini/Veo calls anywhere.

Run with: python -m unittest tests.test_panel_relevant_characters -v
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


class TestPanelRelevantCharacters(unittest.TestCase):
    def test_narrows_to_characters_named_in_action_text(self):
        panel = {"characters_in_frame": ["Alice", "Bob", "Carol", "Dave"],
                 "action": "Alice and Bob argue quietly by the window."}
        self.assertEqual(pipeline._panel_relevant_characters(panel), ["Alice", "Bob"])

    def test_dialogue_speaker_counts_as_relevant_even_if_not_named_in_action(self):
        panel = {"characters_in_frame": ["Alice", "Bob", "Carol"],
                 "action": "Someone laughs across the room.",
                 "dialogue": [{"speaker": "Carol", "line": "Ha!"}]}
        self.assertEqual(pipeline._panel_relevant_characters(panel), ["Carol"])

    def test_falls_back_to_full_list_when_action_names_no_one(self):
        # A genuine group/establishing shot — narrowing to nobody would be
        # worse than the over-inclusion this function exists to fix.
        panel = {"characters_in_frame": ["Alice", "Bob", "Carol"], "action": "The crowd gathers."}
        self.assertEqual(pipeline._panel_relevant_characters(panel), ["Alice", "Bob", "Carol"])

    def test_falls_back_to_full_list_when_no_action_text_at_all(self):
        panel = {"characters_in_frame": ["Alice", "Bob"]}
        self.assertEqual(pipeline._panel_relevant_characters(panel), ["Alice", "Bob"])

    def test_word_boundary_avoids_false_positive_substring_match(self):
        # "Bob" must not match inside "Bobby" — a different, unlisted person.
        # Alice gets a genuine match, so the fallback doesn't mask the bug:
        # if the substring match wrongly fired, Bob would incorrectly appear.
        panel = {"characters_in_frame": ["Bob", "Alice"],
                 "action": "Bobby waves while Alice smiles."}
        self.assertEqual(pipeline._panel_relevant_characters(panel), ["Alice"])

    def test_case_insensitive_match(self):
        panel = {"characters_in_frame": ["Alice"], "action": "ALICE storms out of the room."}
        self.assertEqual(pipeline._panel_relevant_characters(panel), ["Alice"])

    def test_empty_characters_in_frame_returns_empty(self):
        self.assertEqual(pipeline._panel_relevant_characters({"characters_in_frame": []}), [])
        self.assertEqual(pipeline._panel_relevant_characters({}), [])

    def test_preserves_original_list_order_not_match_order(self):
        panel = {"characters_in_frame": ["Alice", "Bob", "Carol"],
                 "action": "Bob turns to Alice."}
        self.assertEqual(pipeline._panel_relevant_characters(panel), ["Alice", "Bob"])


class TestReferenceResolutionUsesRelevantSubset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.cast_index = {}
        for name in ("Alice", "Bob", "Carol", "Dave"):
            rel = f"casting/{name.lower()}.png"
            _touch(self.out / rel)
            self.cast_index[name] = rel

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_panel_references_excludes_irrelevant_scene_cast(self):
        # characters_in_frame defaults to the full 4-person scene cast (the
        # storyboard heuristic for a wide shot), but the action only depicts
        # Alice and Bob — Carol/Dave must not consume the reference budget.
        panel = {"characters_in_frame": ["Alice", "Bob", "Carol", "Dave"],
                 "action": "Alice and Bob argue quietly by the window."}
        refs, char_key = pipeline._resolve_panel_references(panel, None, self.cast_index, self.out)
        names = {p.name for p in refs}
        self.assertEqual(names, {"alice.png", "bob.png"})
        # Boundary/continuity tracking is unaffected — still the FULL cast.
        self.assertEqual(char_key, frozenset({"Alice", "Bob", "Carol", "Dave"}))

    def test_frame_char_anchor_picks_first_relevant_not_first_listed(self):
        # Dave is first in the raw list, but the action only names Carol.
        panel = {"characters_in_frame": ["Dave", "Carol"], "action": "Carol picks up the ball."}
        seed = pipeline._frame_char_anchor(panel, self.cast_index, self.out)
        self.assertEqual(seed.name, "carol.png")

    def test_group_establishing_shot_still_gets_full_reference_set(self):
        # No individual named -> fallback to full list -> unchanged behavior.
        panel = {"characters_in_frame": ["Alice", "Bob"], "action": "The two stand in the empty hall."}
        refs, _ = pipeline._resolve_panel_references(panel, None, self.cast_index, self.out)
        self.assertEqual({p.name for p in refs}, {"alice.png", "bob.png"})


if __name__ == "__main__":
    unittest.main()
