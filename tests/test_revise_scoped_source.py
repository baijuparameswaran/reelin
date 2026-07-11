"""Validates the SCOPED source-text revision flow: a hand-edit to the raw
story text is narrowed down to the specific scene numbers it actually
affects — deterministic candidate pre-filter (`artifact_diff`) confirmed by
an LLM call on the largest local model tier (`reel.agents.revision.
identify_source_text_changes`) — instead of always falling back to a full
regen of every stage. See PROGRESS.md's session log for why this was
previously deferred and what changed.

Pure logic checks with mocked I/O — no LLM/API calls, no real $EDITOR, no
Ollama. Run with: python -m unittest tests.test_revise_scoped_source -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import artifact_diff, cli
from reel.agents import revision as revision_agent


class TestUnifiedSourceDiff(unittest.TestCase):
    def test_shows_only_the_changed_paragraph_with_context(self):
        old = "Marcel walked into the bar.\n\nLater, Marcel drove home.\n\nElena woke up."
        new = "Marcel walked into the bar and lit a cigarette.\n\nLater, Marcel drove home.\n\nElena woke up."
        diff = artifact_diff.unified_source_diff(old, new)
        self.assertIn("-Marcel walked into the bar.", diff)
        self.assertIn("+Marcel walked into the bar and lit a cigarette.", diff)
        # unchanged paragraphs stay OUT of the diff body entirely, beyond
        # whatever context window difflib includes — the whole point is a
        # compact representation, not the full story twice.
        self.assertNotIn("Elena woke up.", diff.replace(" Elena woke up.", ""))

    def test_no_change_yields_empty_diff(self):
        text = "Marcel walked into the bar.\n\nLater, Marcel drove home."
        self.assertEqual(artifact_diff.unified_source_diff(text, text), "")

    def test_single_block_text_still_diffs_by_sentence(self):
        # No paragraph or line breaks at all — must still find granularity
        # via sentence splitting rather than treating the whole story as
        # one giant "changed" unit.
        old = "Marcel walked in. He sat down. He ordered a drink."
        new = "Marcel walked in. He sat down. He ordered two drinks."
        diff = artifact_diff.unified_source_diff(old, new)
        self.assertIn("He ordered a drink.", diff)
        self.assertIn("He ordered two drinks.", diff)
        self.assertNotIn("-Marcel walked in.", diff)


class TestCandidateChangedScenes(unittest.TestCase):
    SCENES = {"scenes": [
        {"number": 1, "source_excerpt": "Marcel walked into the bar. He ordered a whiskey."},
        {"number": 2, "source_excerpt": "Later that night, Marcel drove home through the rain."},
        {"number": 3, "source_excerpt": "The next morning, Elena woke up and made coffee."},
    ]}

    def test_flags_only_the_scene_whose_excerpt_changed(self):
        old = ("Marcel walked into the bar. He ordered a whiskey. "
              "Later that night, Marcel drove home through the rain. "
              "The next morning, Elena woke up and made coffee.")
        new = old.replace("He ordered a whiskey.", "He ordered a whiskey and a cigar.")
        candidates = artifact_diff.candidate_changed_scenes(old, new, self.SCENES)
        self.assertEqual(candidates, [1])

    def test_no_change_yields_no_candidates(self):
        text = ("Marcel walked into the bar. He ordered a whiskey. "
               "Later that night, Marcel drove home through the rain. "
               "The next morning, Elena woke up and made coffee.")
        self.assertEqual(artifact_diff.candidate_changed_scenes(text, text, self.SCENES), [])

    def test_falls_back_to_source_line_for_older_checkpoints(self):
        scenes = {"scenes": [{"number": 5, "source_line": "a very specific quote"}]}
        old = "some story with a very specific quote in it"
        new = "some story with a totally different quote in it"
        self.assertEqual(artifact_diff.candidate_changed_scenes(old, new, scenes), [5])

    def test_whitespace_only_reformatting_is_not_a_false_positive(self):
        old = "Marcel walked into the bar. He ordered a whiskey."
        new = "Marcel walked into the bar.\nHe ordered a whiskey."   # rewrapped, same words
        scenes = {"scenes": [{"number": 1, "source_excerpt": old}]}
        self.assertEqual(artifact_diff.candidate_changed_scenes(old, new, scenes), [])


class TestIdentifySourceTextChanges(unittest.TestCase):
    SCENES = {"scenes": [
        {"number": 1, "source_line": "Marcel walked into the bar", "summary": "s1"},
        {"number": 2, "source_line": "Marcel drove home", "summary": "s2"},
    ]}

    def test_valid_scoped_response(self):
        with mock.patch.object(revision_agent.models, "text", return_value=json.dumps({
                "drastic": False, "changed_scene_numbers": [1], "summary": "x"})):
            result = revision_agent.identify_source_text_changes("diff text", self.SCENES, [1])
        self.assertFalse(result["drastic"])
        self.assertEqual(result["changed_scene_numbers"], [1])

    def test_hallucinated_scene_number_is_sanitized_out(self):
        with mock.patch.object(revision_agent.models, "text", return_value=json.dumps({
                "drastic": False, "changed_scene_numbers": [1, 999], "summary": "x"})):
            result = revision_agent.identify_source_text_changes("diff text", self.SCENES, [1])
        self.assertEqual(result["changed_scene_numbers"], [1])

    def test_malformed_response_fails_safe_to_drastic(self):
        with mock.patch.object(revision_agent.models, "text", return_value="not json"):
            result = revision_agent.identify_source_text_changes("diff text", self.SCENES, [1])
        self.assertTrue(result["drastic"])

    def test_uses_the_configured_revision_profile_by_default(self):
        captured = {}

        def fake_text(prompt, **kwargs):
            captured.update(kwargs)
            return json.dumps({"drastic": False, "changed_scene_numbers": [1]})

        with mock.patch.object(revision_agent.models, "text", side_effect=fake_text):
            revision_agent.identify_source_text_changes("diff text", self.SCENES, [1])
        # agent_profiles.revision is quality_high by default (see config/models.yaml)
        self.assertEqual(captured.get("profile"), "quality_high")

    def test_explicit_profile_overrides_the_configured_default(self):
        captured = {}

        def fake_text(prompt, **kwargs):
            captured.update(kwargs)
            return json.dumps({"drastic": False, "changed_scene_numbers": [1]})

        with mock.patch.object(revision_agent.models, "text", side_effect=fake_text):
            revision_agent.identify_source_text_changes("diff text", self.SCENES, [1], profile="fast")
        self.assertEqual(captured.get("profile"), "fast")


class TestReviseSourceScopedPath(unittest.TestCase):
    """End-to-end (mocked): drive cli._revise_source through the new scoped
    path and confirm it reaches `scenes` + its actual downstream — NOT a
    full regen of every stage — while the drastic/no-scenes cases still
    correctly fall back to the original full-regen behavior."""

    def _setup_run(self, tmp: Path):
        (tmp / "source.json").write_text(json.dumps(
            {"title": "T", "text": "Marcel walked into the bar. He ordered a whiskey. "
                                   "Later, Marcel drove home.", "chunks": []}))
        (tmp / "scenes.json").write_text(json.dumps({"scenes": [
            {"number": 1, "location": "Bar", "characters": ["Marcel"],
             "source_line": "Marcel walked into the bar",
             "source_excerpt": "Marcel walked into the bar. He ordered a whiskey. ",
             "summary": "s1", "purpose": "p1"},
            {"number": 2, "location": "Road", "characters": ["Marcel"],
             "source_line": "Marcel drove home",
             "source_excerpt": "Later, Marcel drove home.",
             "summary": "s2", "purpose": "p2"},
        ]}))
        cli._save_run_params(tmp, max_scenes=None, profile="fast", genre=None, target_duration=30)

    def test_scoped_analysis_only_regenerates_scenes_and_its_downstream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            edited_source = {"title": "T",
                             "text": "Marcel walked into the bar. He ordered a whiskey and "
                                     "a cigar. Later, Marcel drove home.", "chunks": []}

            run_stage_calls = []

            def fake_run_stage(name, *, out, existing=None, revise_keys=None, **kwargs):
                run_stage_calls.append((name, revise_keys))
                return {"scenes": []} if name == "scenes" else {}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.agents.revision.identify_source_text_changes",
                           return_value={"drastic": False, "changed_scene_numbers": [1],
                                        "summary": "Marcel now orders a cigar too"}):
                applied = cli._revise_source(out, edited_override=edited_source,
                                             auto_confirm=True, render=True)

            self.assertTrue(applied)
            called_names = [name for name, _ in run_stage_calls]
            self.assertIn("scenes", called_names)
            # NOT a full-regen replay of the whole STAGES registry — only
            # "scenes" and its actual downstream_of("scenes") should appear.
            self.assertNotIn("structure", called_names)
            self.assertNotIn("characters", called_names)
            # scenes itself got the scoped revise_keys
            scenes_call = next(rk for name, rk in run_stage_calls if name == "scenes")
            self.assertEqual(scenes_call, {1})

    def test_drastic_analysis_falls_back_to_full_regen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self._setup_run(out)
            edited_source = {"title": "T",
                             "text": "A whole new opening scene. Marcel walked into the bar. "
                                     "He ordered a whiskey. Later, Marcel drove home.",
                             "chunks": []}

            run_stage_calls = []

            def fake_run_stage(name, *, out, **kwargs):
                run_stage_calls.append(name)
                return {}

            with mock.patch("reel.stages.run_stage", side_effect=fake_run_stage), \
                 mock.patch("reel.agents.revision.identify_source_text_changes",
                           return_value={"drastic": True,
                                        "reason": "a new opening event has no matching scene",
                                        "changed_scene_numbers": []}):
                applied = cli._revise_source(out, edited_override=edited_source,
                                             auto_confirm=True, render=True)

            self.assertTrue(applied)
            # Full regen: every non-render-skip STAGES entry should have run.
            self.assertIn("structure", run_stage_calls)
            self.assertIn("characters", run_stage_calls)
            self.assertIn("scenes", run_stage_calls)

    def test_no_scenes_json_skips_analysis_entirely_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            (out / "source.json").write_text(json.dumps(
                {"title": "T", "text": "original text", "chunks": []}))
            edited_source = {"title": "T", "text": "a genuinely different text", "chunks": []}

            analysis_calls = []

            def fake_identify(*a, **kw):
                analysis_calls.append((a, kw))
                return {"drastic": False, "changed_scene_numbers": [1]}

            with mock.patch("reel.stages.run_stage", return_value={}), \
                 mock.patch("reel.agents.revision.identify_source_text_changes",
                           side_effect=fake_identify):
                applied = cli._revise_source(out, edited_override=edited_source,
                                             auto_confirm=True, render=True)

            self.assertTrue(applied)
            self.assertEqual(analysis_calls, [],
                             "no scenes.json to scope against — the LLM call should never "
                             "even be attempted")


if __name__ == "__main__":
    unittest.main()
