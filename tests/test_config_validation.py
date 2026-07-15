"""Validates `llm.validate_config` — a lightweight, dependency-free shape
check for config/models.yaml catching unknown keys (a typo like
`videos:`/`mutli_character_references`) and wrong types on a handful of
consequential fields, since every individual `.get(key, default)` read
elsewhere in this codebase is robust to a MISSING key but silent on a
MISSPELLED one. See llm.py's own module-level docstring for the design
rationale (deliberately not a full JSON-schema).

Pure logic — no live Ollama/Gemini/Veo calls, no file I/O beyond reading
the real config/models.yaml once (to confirm it validates clean).

Run with: python -m unittest tests.test_config_validation -v
"""
from __future__ import annotations

import unittest

from reel import llm


class TestValidateConfig(unittest.TestCase):
    def test_real_config_file_validates_with_zero_warnings(self):
        # The actual checked-in config/models.yaml must never trip its own
        # validator — a regression guard against the schema and the real
        # file drifting apart.
        self.assertEqual(llm.validate_config(), [])

    def test_unknown_top_level_key_flagged(self):
        warnings = llm.validate_config({"videos": {"backend": "gemini"}})
        self.assertTrue(any("videos" in w and "top-level" in w for w in warnings))

    def test_unknown_nested_key_flagged(self):
        warnings = llm.validate_config({"video": {"mutli_character_references": True}})
        self.assertTrue(any("video.mutli_character_references" in w for w in warnings))

    def test_unknown_deeply_nested_audio_key_flagged(self):
        warnings = llm.validate_config({"video": {"audio": {"no_bg_msuic": True}}})
        self.assertTrue(any("video.audio.no_bg_msuic" in w for w in warnings))

    def test_unknown_deeply_nested_overlay_key_flagged(self):
        warnings = llm.validate_config({"video": {"overlays": {"fontsize": 24}}})
        self.assertTrue(any("video.overlays.fontsize" in w for w in warnings))

    def test_known_keys_produce_no_warnings(self):
        cfg = {"hitl": {"enabled": True, "timeout_seconds": 900},
               "video": {"enabled": True, "audio": {"room_tone": True},
                        "overlays": {"enabled": False}}}
        self.assertEqual(llm.validate_config(cfg), [])

    def test_wrong_type_flagged(self):
        warnings = llm.validate_config({"hitl": {"enabled": "true"}})
        self.assertTrue(any("hitl.enabled" in w and "bool" in w for w in warnings))

    def test_correct_type_produces_no_warning(self):
        warnings = llm.validate_config({"hitl": {"enabled": True}})
        self.assertEqual(warnings, [])

    def test_block_that_should_be_a_mapping_but_isnt(self):
        warnings = llm.validate_config({"video": "gemini"})
        self.assertTrue(any("video" in w and "mapping" in w for w in warnings))

    def test_agent_profile_referencing_undefined_profile_flagged(self):
        warnings = llm.validate_config({
            "profiles": {"fast": {}, "quality": {}},
            "agent_profiles": {"scenes": "qualty"},
        })
        self.assertTrue(any("qualty" in w for w in warnings))

    def test_agent_profile_referencing_defined_profile_produces_no_warning(self):
        warnings = llm.validate_config({
            "profiles": {"fast": {}, "quality": {}},
            "agent_profiles": {"scenes": "quality"},
        })
        self.assertEqual(warnings, [])

    def test_non_dict_root_yields_single_warning_not_a_crash(self):
        warnings = llm.validate_config("not a dict")
        self.assertEqual(len(warnings), 1)
        self.assertIn("not a mapping", warnings[0])

    def test_empty_config_yields_no_warnings(self):
        # Nothing present isn't a typo — every block is optional and
        # degrades via its own .get(..., default) elsewhere.
        self.assertEqual(llm.validate_config({}), [])

    def test_defaults_to_real_config_when_none_given(self):
        # No argument -> validates the actual loaded config/models.yaml.
        self.assertEqual(llm.validate_config(None), [])


if __name__ == "__main__":
    unittest.main()
