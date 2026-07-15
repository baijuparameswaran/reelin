"""Validates `fountain.to_storyboard`'s 2026-07-14 rework: the standalone
`render` CLI path (screenplay.fountain + cinematography.json, bypassing
storyboard.json) now builds a board SHAPE-COMPATIBLE with the main
pipeline's storyboard.json — multi-character `characters_in_frame` (not a
single best-guess name), scene-level `header.location`/`visual_overview`/
`audio_overview`, structured per-panel `dialogue`, and an `image_prompt`
preview built via the SAME shared `veo_prompt.panel_video_prompt` formula
the real render uses — instead of a separate, staler implementation. See
PROGRESS.md's 2026-07-14 session log for the audit that found the gap.

Pure logic — no live Ollama/Gemini/Veo calls anywhere.

Run with: python -m unittest tests.test_fountain_to_storyboard -v
"""
from __future__ import annotations

import unittest

from reel import fountain


def _casting(*, alice_img=True, bob_img=True, location_img=True):
    entries = [
        {"name": "Alice", "kind": "person",
         "character": {"physical_form": "tall, red hair",
                       **({"image_path": "casting/alice.png"} if alice_img else {})}},
        {"name": "Bob", "kind": "person",
         "character": {"physical_form": "short, bald",
                       **({"image_path": "casting/bob.png"} if bob_img else {})}},
        {"name": "BAR", "kind": "location",
         "character": {"visual_prompt": "a dim, smoky dive bar",
                       **({"image_path": "casting/bar.png"} if location_img else {})}},
    ]
    return {"casting": entries}


class TestBeatCharacters(unittest.TestCase):
    def test_names_action_text_characters(self):
        result = fountain._beat_characters("Alice and Bob argue by the door.", [], [], _casting())
        self.assertEqual(result, ["Alice", "Bob"])

    def test_beat_dialogue_speaker_included(self):
        result = fountain._beat_characters(
            "Someone laughs.", [("Bob", "Ha!")], [("Bob", "Ha!")], _casting())
        self.assertIn("Bob", result)

    def test_falls_back_to_scene_wide_speakers_when_beat_names_no_one(self):
        result = fountain._beat_characters(
            "The room falls silent.", [], [("Alice", "Wait.")], _casting())
        self.assertEqual(result, ["Alice"])

    def test_falls_back_to_first_cast_entry_with_image_when_nothing_matches(self):
        result = fountain._beat_characters("The wind blows.", [], [], _casting())
        self.assertEqual(result, ["Alice"])

    def test_location_never_included_as_a_character(self):
        result = fountain._beat_characters("Someone stands at the BAR.", [], [], _casting())
        self.assertNotIn("BAR", result)

    def test_all_caps_fountain_speaker_cue_resolves_to_canonical_casting_name(self):
        # Fountain speaker cues are conventionally ALL CAPS ("ALICE"), while
        # casting.json uses Title Case ("Alice") — a case-sensitive match
        # would silently drop her from characters_in_frame entirely.
        result = fountain._beat_characters(
            "The room falls silent.", [("ALICE", "Wait.")], [("ALICE", "Wait.")], _casting())
        self.assertEqual(result, ["Alice"])  # canonical casing, not "ALICE"


class TestCanonicalName(unittest.TestCase):
    def test_case_insensitive_match_returns_canonical_casing(self):
        self.assertEqual(fountain._canonical_name("ALICE", ["Alice", "Bob"]), "Alice")

    def test_no_match_returns_none(self):
        self.assertIsNone(fountain._canonical_name("CAROL", ["Alice", "Bob"]))


class TestDistributeDialogue(unittest.TestCase):
    def test_spreads_lines_across_beats_not_duplicated_everywhere(self):
        dialogue = [("Alice", "Hi"), ("Bob", "Hey"), ("Alice", "Bye")]
        buckets = fountain._distribute_dialogue(dialogue, 3)
        self.assertEqual(len(buckets), 3)
        total = sum(len(b) for b in buckets)
        self.assertEqual(total, 3)  # every line placed exactly once

    def test_empty_dialogue_yields_empty_buckets(self):
        buckets = fountain._distribute_dialogue([], 3)
        self.assertEqual(buckets, [[], [], []])

    def test_zero_beats_yields_no_buckets(self):
        self.assertEqual(fountain._distribute_dialogue([("Alice", "Hi")], 0), [])


class TestSceneContext(unittest.TestCase):
    def test_prefers_scenes_json_location_over_slugline(self):
        scenes_json = {"scenes": [{"number": 1, "location": "BAR"}]}
        ctx = fountain._scene_context(1, scenes_json, {}, {}, _casting(), "INT. SALOON - NIGHT")
        self.assertEqual(ctx["header"]["location"], "BAR")

    def test_falls_back_to_slugline_extraction_without_scenes_json(self):
        ctx = fountain._scene_context(1, None, {}, {}, _casting(), "INT. BAR - NIGHT")
        self.assertEqual(ctx["header"]["location"], "BAR")

    def test_visual_overview_field_mapping_matches_storyboard_py(self):
        visuals = {"scenes": [{"scene_number": 1, "color_palette": "warm amber",
                               "lighting": "dim practicals", "emotional_function": "melancholic"}]}
        ctx = fountain._scene_context(1, None, {}, visuals, _casting(), "INT. BAR - NIGHT")
        self.assertEqual(ctx["visual_overview"],
                        {"color_palette": "warm amber", "lighting_setup": "dim practicals",
                         "mood": "melancholic"})

    def test_audio_overview_respects_silence_flag(self):
        soundscape = {"soundscapes": [{"scene_number": 1, "ambient_bed": "murmur",
                                       "score_direction": "soft piano", "silence": True}]}
        ctx = fountain._scene_context(1, None, soundscape, {}, _casting(), "INT. BAR - NIGHT")
        self.assertEqual(ctx["audio_overview"]["ambient"], "")
        self.assertEqual(ctx["audio_overview"]["score_cue"], "soft piano")


class TestPanelSound(unittest.TestCase):
    def test_ambient_and_matching_sfx_combined(self):
        soundscape_scene = {"ambient_bed": "murmur of the crowd",
                            "sound_events": [{"moment": "glass shatters", "sound": "a sharp crash"}]}
        result = fountain._panel_sound("A glass shatters on the floor.", soundscape_scene)
        self.assertIn("murmur of the crowd", result)
        self.assertIn("a sharp crash", result)
        self.assertIn(" | ", result)

    def test_silence_flag_suppresses_ambient(self):
        soundscape_scene = {"ambient_bed": "murmur", "silence": True}
        result = fountain._panel_sound("Nothing happens.", soundscape_scene)
        self.assertEqual(result, "")


class TestToStoryboardIntegration(unittest.TestCase):
    def setUp(self):
        self.scenes = fountain.parse(
            "INT. BAR - NIGHT\n\nAlice and Bob argue quietly by the window.\n\n"
            "ALICE\n\nWe need to talk.\n"
        )
        self.soundscape = {"soundscapes": [{"scene_number": 1, "ambient_bed": "murmur of the crowd"}]}
        self.visuals = {"scenes": [{"scene_number": 1, "color_palette": "warm amber",
                                    "lighting": "dim practicals", "emotional_function": "tense"}]}
        self.casting = _casting()
        self.cinematography = {"scenes": [{"scene_number": 1, "shots": [
            {"type": "WS", "angle": "eye level", "movement": "static", "lens": "24mm",
             "framing": "wide establishing"},
        ]}]}
        self.scenes_json = {"scenes": [{"number": 1, "location": "BAR"}]}

    def test_board_shape_matches_storyboard_py(self):
        board = fountain.to_storyboard(
            self.scenes, self.soundscape, self.visuals, self.casting, None,
            cinematography=self.cinematography, scenes_json=self.scenes_json)
        scene = board["storyboard"][0]
        self.assertIn("header", scene)
        self.assertEqual(scene["header"]["location"], "BAR")
        self.assertIn("visual_overview", scene)
        self.assertIn("audio_overview", scene)
        self.assertIn("panels", scene)
        panel = scene["panels"][0]
        for key in ("panel", "shot_type", "camera_angle", "camera_movement", "lens",
                   "composition", "characters_in_frame", "action", "dialogue", "sound",
                   "image_prompt"):
            self.assertIn(key, panel)

    def test_multi_character_panel_not_single_best_guess(self):
        board = fountain.to_storyboard(
            self.scenes, self.soundscape, self.visuals, self.casting, None,
            cinematography=self.cinematography, scenes_json=self.scenes_json)
        panel = board["storyboard"][0]["panels"][0]
        self.assertEqual(set(panel["characters_in_frame"]), {"Alice", "Bob"})

    def test_camera_fields_veo_normalized(self):
        board = fountain.to_storyboard(
            self.scenes, self.soundscape, self.visuals, self.casting, None,
            cinematography=self.cinematography, scenes_json=self.scenes_json)
        panel = board["storyboard"][0]["panels"][0]
        self.assertEqual(panel["shot_type"], "WS")
        self.assertEqual(panel["camera_angle"], "eye level")
        self.assertEqual(panel["lens"], "24mm")

    def test_speaking_character_in_frame_not_marked_off_screen(self):
        # Regression: the scene's dialogue speaker is Fountain-parsed as
        # "ALICE" (all caps); she must still be recognized as in-frame
        # (modifier == "") rather than wrongly marked off-screen ("O.S.")
        # due to a case-sensitive comparison against casting.json's "Alice".
        board = fountain.to_storyboard(
            self.scenes, self.soundscape, self.visuals, self.casting, None,
            cinematography=self.cinematography, scenes_json=self.scenes_json)
        panel = board["storyboard"][0]["panels"][0]
        alice_line = next(d for d in panel["dialogue"] if d["speaker"] == "Alice")
        self.assertEqual(alice_line["modifier"], "")

    def test_image_prompt_built_via_shared_five_part_formula(self):
        board = fountain.to_storyboard(
            self.scenes, self.soundscape, self.visuals, self.casting, None,
            cinematography=self.cinematography, scenes_json=self.scenes_json)
        panel = board["storyboard"][0]["panels"][0]
        prompt = panel["image_prompt"]
        # Style & Ambiance element (fixed keyword every five-part prompt carries).
        self.assertIn("Cinematic, photorealistic", prompt)
        # Subject element — full physical description (no seed/reference at
        # storyboard-build time, so nothing is shortened yet).
        self.assertIn("red hair", prompt)
        # Context element — the locked location visual_prompt, not the bare name.
        self.assertIn("dim, smoky dive bar", prompt)
        # Style & Ambiance's scene-specific mood/palette.
        self.assertIn("warm amber", prompt)

    def test_veo_verify_prompt_passes_on_the_built_preview(self):
        from reel import veo_guide
        board = fountain.to_storyboard(
            self.scenes, self.soundscape, self.visuals, self.casting, None,
            cinematography=self.cinematography, scenes_json=self.scenes_json)
        panel = board["storyboard"][0]["panels"][0]
        report = veo_guide.verify_prompt(panel["image_prompt"])
        self.assertTrue(report["valid"], report["issues"])

    def test_degrades_gracefully_without_scenes_json(self):
        board = fountain.to_storyboard(
            self.scenes, self.soundscape, self.visuals, self.casting, None,
            cinematography=self.cinematography, scenes_json=None)
        scene = board["storyboard"][0]
        self.assertEqual(scene["header"]["location"], "BAR")  # from slugline fallback


if __name__ == "__main__":
    unittest.main()
