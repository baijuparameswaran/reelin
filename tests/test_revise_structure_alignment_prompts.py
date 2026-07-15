"""Validates that a SCOPED revision's prompt to the model explicitly states
the structure (scene/shot/panel count) it's expected to align to, per direct
instruction: "make sure revision prompt to the model need to make sure the
structure as in the number of scenes/shots/panels may be explicitly aligned
during revisions of respective stage .. Deviation as in additions and
deletions may be done only as exceptions."

`revision_merge.merge_by_key` already enforces this at the CODE level (an
unrequested addition from the model is discarded — see
`tests/test_revision_merge_fields.py`), but the model itself had no idea what
structure it was supposed to preserve when scoped. This adds an explicit
"STRUCTURE ALIGNMENT" block to each affected agent's prompt, present only
during a scoped revision (existing + revise_keys given), stating the
EXISTING count so a count change is a deliberate, model-visible exception
rather than an unprompted side effect discovered only after the fact by the
merge layer.

Four agents affected: `scenes.py` (scene count), `cinematography.py` +
`screenplay.py` (shot count per scene), `storyboard.py` (panel count per
scene, LLM-feedback path only).

Pure logic / prompt-text checks with mocked `llm.generate` — no real LLM/API
calls. Run with:
    python -m unittest tests.test_revise_structure_alignment_prompts -v
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from reel.agents import scenes as scenes_agent
from reel.agents import cinematography as cinematography_agent
from reel.agents import screenplay as screenplay_agent
from reel.agents import storyboard as storyboard_agent


class TestScenesStructureNote(unittest.TestCase):
    def test_states_existing_count_and_targeted_numbers(self):
        existing = {"scenes": [{"number": 1}, {"number": 2}, {"number": 3}]}
        note = scenes_agent._structure_alignment_note(existing, {2})
        self.assertIn("3 scene(s)", note)
        self.assertIn("[2]", note)
        self.assertIn("STRUCTURE ALIGNMENT", note)

    def test_empty_for_fresh_non_scoped_run(self):
        self.assertEqual(scenes_agent._structure_alignment_note(None, None), "")
        self.assertEqual(scenes_agent._structure_alignment_note({"scenes": []}, None), "")

    def test_note_reaches_the_actual_prompt_sent_to_the_model(self):
        existing = {"scenes": [{"number": 1, "source_line": "x"}]}
        captured = {}

        def fake_generate(prompt, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"scenes": [{"number": 1, "source_line": "x", "summary": "s"}]})

        with mock.patch.object(scenes_agent.llm, "generate", side_effect=fake_generate):
            scenes_agent.segment_scenes(
                {"title": "T", "text": "some story text"}, {"three_act": {}},
                existing=existing, revise_keys={1})

        self.assertIn("STRUCTURE ALIGNMENT", captured["prompt"])
        self.assertIn("1 scene(s)", captured["prompt"])


class TestScenesRevisionReminderNote(unittest.TestCase):
    """Distinct from `_structure_alignment_note` (scoped-revision-only,
    "keep the same count"): this note fires for a DRASTIC, fully unscoped
    regen where the count is EXPECTED to change — purely informational
    (states the prior count as context), deferring to rule 9 (CAPTURE THE
    STORY FULLY) for how the edited text should actually be segmented — see
    `cli._revise_source`'s fallback."""

    def test_states_prior_count_and_reinforces_rule_9(self):
        note = scenes_agent._revision_reminder_note(2)
        self.assertIn("REVISION", note)
        self.assertIn("2 scene(s)", note)
        self.assertIn("CAPTURE THE STORY FULLY", note)
        self.assertNotIn("MINIMIZE SCENE COUNT", note)

    def test_empty_for_none_or_zero(self):
        self.assertEqual(scenes_agent._revision_reminder_note(None), "")
        self.assertEqual(scenes_agent._revision_reminder_note(0), "")

    def test_note_reaches_the_actual_prompt_for_a_drastic_unscoped_regen(self):
        captured = {}

        def fake_generate(prompt, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"scenes": [{"number": 1, "source_line": "x", "summary": "s"}]})

        with mock.patch.object(scenes_agent.llm, "generate", side_effect=fake_generate):
            # No existing/revise_keys — this is the fully unscoped, drastic
            # full-regen shape `cli._revise_source`'s fallback uses.
            scenes_agent.segment_scenes(
                {"title": "T", "text": "some story text"}, {"three_act": {}},
                prior_scene_count=1)

        self.assertIn("REVISION", captured["prompt"])
        self.assertIn("1 scene(s)", captured["prompt"])

    def test_scoped_alignment_note_wins_when_both_could_apply(self):
        # Mutually exclusive in practice, but confirm the precedence is
        # sane if ever called with both: a genuinely scoped call (existing +
        # revise_keys) should use the "keep the same count" framing, not the
        # "count may change" one, regardless of prior_scene_count.
        existing = {"scenes": [{"number": 1}]}
        note = scenes_agent._structure_alignment_note(existing, {1}) \
            or scenes_agent._revision_reminder_note(5)
        self.assertIn("SCOPED REVISION", note)
        self.assertNotIn("5 scene(s)", note)


class TestCinematographyStructureNote(unittest.TestCase):
    def test_lists_existing_shot_count_per_targeted_scene(self):
        existing = {"scenes": [
            {"scene_number": 1, "shots": [1, 2, 3]},
            {"scene_number": 2, "shots": [1]},
            {"scene_number": 3, "shots": [1, 2]},
        ]}
        note = cinematography_agent._shot_structure_note(existing, {1, 2})
        self.assertIn("Scene 1: currently 3 shot(s)", note)
        self.assertIn("Scene 2: currently 1 shot(s)", note)
        self.assertNotIn("Scene 3", note)

    def test_empty_when_not_scoped_or_nothing_to_compare(self):
        self.assertEqual(cinematography_agent._shot_structure_note(None, {1}), "")
        self.assertEqual(cinematography_agent._shot_structure_note({"scenes": []}, {1}), "")

    def test_new_scene_with_no_prior_shots_is_skipped_not_erroring(self):
        existing = {"scenes": [{"scene_number": 1, "shots": [1]}]}
        # scene 5 is newly targeted but has no existing entry at all
        note = cinematography_agent._shot_structure_note(existing, {1, 5})
        self.assertIn("Scene 1", note)
        self.assertNotIn("Scene 5", note)


class TestScreenplayStructureNote(unittest.TestCase):
    def test_states_this_scenes_existing_shot_count(self):
        note = screenplay_agent._shot_structure_note({"number": 1, "shots": [1, 2]})
        self.assertIn("2 shot(s)", note)
        self.assertIn("STRUCTURE ALIGNMENT", note)

    def test_empty_for_none_or_no_prior_shots(self):
        self.assertEqual(screenplay_agent._shot_structure_note(None), "")
        self.assertEqual(screenplay_agent._shot_structure_note({"number": 1, "shots": []}), "")


class TestStoryboardStructureNote(unittest.TestCase):
    def test_states_existing_panel_count(self):
        note = storyboard_agent._panel_structure_note(3)
        self.assertIn("3 panel(s)", note)
        self.assertIn("STRUCTURE ALIGNMENT", note)

    def test_empty_for_none_or_zero(self):
        self.assertEqual(storyboard_agent._panel_structure_note(None), "")
        self.assertEqual(storyboard_agent._panel_structure_note(0), "")

    def test_note_reaches_llm_generate_scene_prompt_when_scoped_with_feedback(self):
        bundle = {"scene_number": 1, "art": {}, "audio": {}, "camera": {}}
        captured = {}

        def fake_generate(prompt, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"storyboard": [{"scene_number": 1, "panels": []}],
                              "storyboard_style": "x"})

        with mock.patch.object(storyboard_agent.llm, "generate", side_effect=fake_generate):
            storyboard_agent._llm_generate_scene(
                dict(bundle), "logline", "Drama", "somber", "source text",
                "make it darker", "fast", existing_panel_count=4)

        self.assertIn("STRUCTURE ALIGNMENT", captured["prompt"])
        self.assertIn("4 panel(s)", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
