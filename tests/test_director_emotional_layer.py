"""Validates the DIRECTOR'S INTERPRETIVE EXPANSION layer (`scenes.py` rule
11), added per direct instruction: "refine the scenes prompt to allow an
excellent and awesome Director's view point to expand each line of the story
ingested to capture elaborate emotional and expressional moments in the
story as much as possible. These may not be explicitly spelt out in the
original story."

Three things are worth locking down with tests, because each one is a place
this change could quietly go wrong:

1. THE BOUNDARY. This deliberately loosens the most load-bearing rule set in
   this codebase (`scenes.py` rules 1-2, "SOURCE TEXT IS THE ONLY
   AUTHORITY") — but only for interpretation of HOW a moment is felt and
   performed, never for new FACTS. Rule 11 must state that limit, rules 1-2
   must still forbid invented events/characters/locations/props/outcomes,
   and rule 4 must still quarantine the interpretive layer out of `summary`
   (which fidelity grades against) and into the two new named fields.

2. NO DEAD FIELDS. This project has twice shipped a creative field that
   never reached a rendered frame (`visuals.key_props`, then panel
   `emotional_note` — see PROGRESS.md's 2026-07-10 and 2026-07-23 entries).
   So `emotional_beat`/`expression` must actually survive every hop:
   scenes.json → the three scene-keyed design agents' prompts + screenplay's
   per-scene prompt → storyboard's bundle → a panel's `emotional_note` →
   `veo_prompt.panel_action`'s Action clause.

3. THE EXCERPT COLLISION. Rule 11 makes it legitimate for two scenes to draw
   distinct emotional beats out of the SAME passage, which makes two scenes
   resolving to the same `source_line` offset genuinely likely for the first
   time. `_attach_source_excerpts` previously handed the earlier of two such
   scenes an empty excerpt and a `word_count` of 0.

Offline — mocked `llm.generate`, no LLM/API calls. Run with:
    python -m unittest tests.test_director_emotional_layer -v
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from reel import veo_prompt
from reel.agents import cinematography as cinematography_agent
from reel.agents import scenes as scenes_agent
from reel.agents import screenplay as screenplay_agent
from reel.agents import soundscape as soundscape_agent
from reel.agents import storyboard as storyboard_agent
from reel.agents import visuals as visuals_agent


def _render_scenes_prompt(**overrides):
    kwargs = dict(target="as many as the story needs", beats="{}", title="T",
                  text="some source text", character_names_block="",
                  structure_note="")
    kwargs.update(overrides)
    return scenes_agent.PROMPT.format(**kwargs)


class TestScenesRule11Present(unittest.TestCase):
    """(1) The boundary — the prompt must license interpretation and refuse
    invention, in the same breath."""

    def test_rule_11_and_its_two_new_fields_are_in_the_prompt(self):
        out = _render_scenes_prompt()
        self.assertIn("DIRECTOR'S INTERPRETIVE EXPANSION", out)
        self.assertIn('"emotional_beat":', out)
        self.assertIn('"expression":', out)

    def test_rule_11_states_the_limit_interpretation_not_new_facts(self):
        out = _render_scenes_prompt()
        self.assertIn("THE LIMIT", out)
        # The distinction the whole change rests on, stated explicitly.
        self.assertIn("inferred PLOT POINT is not", out)

    def test_rules_1_and_2_still_forbid_invented_facts(self):
        out = _render_scenes_prompt()
        self.assertIn("SOURCE TEXT IS THE ONLY AUTHORITY", out)
        self.assertIn("No invented characters, plot points, locations, props, "
                      "or outcomes", out)

    def test_rule_4_quarantines_interpretation_out_of_summary(self):
        """`summary` is what fidelity grading reads as the plain record of
        what happens — the interpretive layer must not leak into it."""
        out = _render_scenes_prompt()
        self.assertIn("must describe only what the source text says", out)
        self.assertIn("Keep your interpretive reading OUT of `summary`", out)

    def test_rule_7_still_blocks_two_scenes_restating_one_moment(self):
        """Rule 11 licenses two scenes from one passage; rule 7 has to stop
        that from degrading into duplicate coverage of a single beat."""
        out = _render_scenes_prompt()
        self.assertIn("genuinely different emotional or expressional beat", out)
        self.assertIn("never turns one moment into two scenes", out)

    def test_generic_emotional_reads_are_an_explicit_do_not(self):
        out = _render_scenes_prompt()
        self.assertIn("unperformable", out)

    def test_sandwich_block_reinforces_rule_11_and_its_limit(self):
        """The closing re-check block is the mitigation for a large data
        block pushing early rules out of context — rule 11 and its limit are
        exactly the pair that must not get lost there."""
        tail = _render_scenes_prompt().split("Before you respond")[-1]
        self.assertIn("DIRECTOR'S INTERPRETIVE EXPANSION", tail)
        self.assertIn("never new facts", tail)
        self.assertIn("all 10 schema fields", tail)


class TestSourceExcerptCollision(unittest.TestCase):
    """(3) Two scenes legitimately drawn from the same passage must BOTH get
    a real excerpt — the earlier one used to get "" / word_count 0, which
    silently broke duration estimation for it."""

    SOURCE = ("She set the cup down without drinking. "
              "Outside, the last of the light went out of the street.")

    def test_two_scenes_sharing_a_source_line_both_get_the_passage(self):
        scene_list = [
            {"number": 1, "source_line": "She set the cup down without drinking"},
            {"number": 2, "source_line": "She set the cup down without drinking"},
        ]
        result = scenes_agent._attach_source_excerpts(scene_list, self.SOURCE)
        for sc in result:
            self.assertTrue(sc["source_excerpt"].startswith("She set the cup down"),
                            f"scene {sc['number']} got {sc['source_excerpt']!r}")
            self.assertGreater(sc["word_count"], 0)

    def test_collision_does_not_swallow_a_later_scenes_span(self):
        """The colliding pair shares the span up to the NEXT DISTINCT anchor;
        the scene at that anchor keeps its own."""
        scene_list = [
            {"number": 1, "source_line": "She set the cup down"},
            {"number": 2, "source_line": "She set the cup down"},
            {"number": 3, "source_line": "the last of the light went out"},
        ]
        result = scenes_agent._attach_source_excerpts(scene_list, self.SOURCE)
        by_num = {s["number"]: s for s in result}
        self.assertNotIn("light went out", by_num[1]["source_excerpt"])
        self.assertNotIn("light went out", by_num[2]["source_excerpt"])
        self.assertTrue(by_num[3]["source_excerpt"].startswith("the last of the light"))

    def test_distinct_anchors_still_partition_as_before(self):
        """Regression guard: the ordinary non-colliding case is unchanged."""
        scene_list = [
            {"number": 1, "source_line": "She set the cup down"},
            {"number": 2, "source_line": "the last of the light went out"},
        ]
        result = scenes_agent._attach_source_excerpts(scene_list, self.SOURCE)
        self.assertNotIn("light went out", result[0]["source_excerpt"])
        self.assertTrue(result[1]["source_excerpt"].startswith("the last of the light"))


class TestFieldsReachTheDesignAgents(unittest.TestCase):
    """(2) No dead fields — each design agent projects an explicit subset of
    each scene's keys into its prompt, so a new field is dropped unless it's
    added to that projection."""

    SCENES = {"scenes": [{
        "number": 1, "slugline": "INT. KITCHEN - NIGHT", "location": "the kitchen",
        "summary": "She sets the cup down.", "purpose": "the refusal",
        "source_line": "She set the cup down", "chunk_indices": [0], "props": ["a cup"],
        "emotional_beat": "the exhaustion of someone done asking",
        "expression": "hands flat on the table, eyes fixed on nothing",
    }]}
    STRUCTURE = {"logline": "L", "genre": "G", "tone": "T", "themes": ["t"]}

    def _capture(self, agent, call):
        captured = {}

        def fake_generate(prompt, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"scenes": [], "soundscapes": []})

        with mock.patch.object(agent.llm, "generate", side_effect=fake_generate):
            call()
        return captured["prompt"]

    def test_cinematography_prompt_carries_and_grounds_them(self):
        prompt = self._capture(cinematography_agent, lambda: cinematography_agent
                               .plan_cinematography(self.STRUCTURE, self.SCENES))
        self.assertIn("the exhaustion of someone done asking", prompt)
        self.assertIn("hands flat on the table", prompt)
        # Grounded to an output field, not merely present in the data — the
        # gap PROGRESS.md's 2026-07-09 entry found seven instances of.
        self.assertIn("HONOR THE SCENE'S EMOTIONAL DIRECTION", prompt)
        self.assertIn("`emotional_function`", prompt)

    def test_visuals_prompt_carries_and_grounds_them(self):
        prompt = self._capture(visuals_agent, lambda: visuals_agent
                               .design_visuals(self.STRUCTURE, self.SCENES))
        self.assertIn("the exhaustion of someone done asking", prompt)
        self.assertIn("HONOR THE SCENE'S EMOTIONAL DIRECTION", prompt)

    def test_soundscape_prompt_carries_and_grounds_them(self):
        prompt = self._capture(soundscape_agent, lambda: soundscape_agent
                               .design_soundscape(self.STRUCTURE, self.SCENES))
        self.assertIn("the exhaustion of someone done asking", prompt)
        self.assertIn("HONOR THE SCENE'S EMOTIONAL DIRECTION", prompt)

    def test_every_design_agent_notes_the_older_scenes_json_fallback(self):
        """A scenes.json checkpoint predating these fields must not read as a
        missing-data error to the model."""
        for agent, call in (
            (cinematography_agent, lambda: cinematography_agent.plan_cinematography(
                self.STRUCTURE, self.SCENES)),
            (visuals_agent, lambda: visuals_agent.design_visuals(
                self.STRUCTURE, self.SCENES)),
            (soundscape_agent, lambda: soundscape_agent.design_soundscape(
                self.STRUCTURE, self.SCENES)),
        ):
            with self.subTest(agent=agent.__name__):
                self.assertIn("an older scene list", self._capture(agent, call))


class TestScreenplayDirectorBlock(unittest.TestCase):
    """(2) continued — screenplay writes the action/parentheticals a panel's
    `action` text comes from, so the direction has to reach it too."""

    def test_includes_both_fields_and_restates_the_limit(self):
        block = screenplay_agent._director_block({
            "emotional_beat": "the exhaustion of someone done asking",
            "expression": "hands flat on the table",
        })
        self.assertIn("the exhaustion of someone done asking", block)
        self.assertIn("hands flat on the table", block)
        # Must not read as a licence to invent, given this prompt's own
        # FIDELITY FIRST / SOURCE OVER COVERAGE rules.
        self.assertIn("never WHAT happens", block)

    def test_partial_fields_still_produce_a_block(self):
        self.assertIn("done asking", screenplay_agent._director_block(
            {"emotional_beat": "someone done asking"}))
        self.assertIn("hands flat", screenplay_agent._director_block(
            {"expression": "hands flat on the table"}))

    def test_noop_for_a_scene_without_the_fields(self):
        self.assertEqual(screenplay_agent._director_block({}), "")
        self.assertEqual(screenplay_agent._director_block(
            {"emotional_beat": "", "expression": "  "}), "")


class TestStoryboardPanelEmotionalNote(unittest.TestCase):
    """(2) continued — the last hop: a panel's `emotional_note` is what
    `veo_prompt.panel_action` actually folds into the rendered Veo prompt."""

    def _bundle(self, **overrides):
        bundle = {
            "art": {}, "audio": {}, "cast": [], "location": {},
            "characters_in_scene": ["Alice"], "camera": {},
            "director": {"emotional_beat": "the exhaustion of someone done asking",
                         "expression": "hands flat on the table, eyes fixed on nothing"},
        }
        bundle.update(overrides)
        return bundle

    def _panel(self, bundle, cam=None):
        return storyboard_agent._build_panel(
            1, cam or {"type": "close-up"}, {"description": "Alice sets the cup down"},
            bundle, is_last=False)

    def test_falls_back_to_the_scenes_director_direction(self):
        panel = self._panel(self._bundle())
        self.assertEqual(panel["emotional_note"],
                         "hands flat on the table, eyes fixed on nothing")

    def test_prefers_expression_over_emotional_beat(self):
        """`expression` is already phrased as something visible/performable,
        which is what the Action clause needs."""
        panel = self._panel(self._bundle())
        self.assertNotIn("done asking", panel["emotional_note"])

    def test_emotional_beat_used_when_expression_is_absent(self):
        panel = self._panel(self._bundle(
            director={"emotional_beat": "the exhaustion of someone done asking"}))
        self.assertEqual(panel["emotional_note"],
                         "the exhaustion of someone done asking")

    def test_more_specific_per_shot_read_still_wins(self):
        """The DP's own per-shot read is more specific than a scene-wide one,
        so the existing precedence must be preserved, not overridden."""
        panel = self._panel(self._bundle(),
                            cam={"type": "close-up",
                                 "emotional_function": "the moment she stops hoping"})
        self.assertEqual(panel["emotional_note"], "the moment she stops hoping")

    def test_scene_wide_art_read_also_still_wins_over_the_floor(self):
        panel = self._panel(self._bundle(
            art={"emotional_function": "cold light, no comfort in the room"}))
        self.assertEqual(panel["emotional_note"], "cold light, no comfort in the room")

    def test_no_director_block_is_not_an_error(self):
        panel = self._panel({"art": {}, "audio": {}, "cast": [], "location": {},
                             "characters_in_scene": [], "camera": {}})
        self.assertEqual(panel["emotional_note"], "")

    def test_the_direction_reaches_the_rendered_veo_action_clause(self):
        """End of the chain: scenes.json rule 11 → a Veo [Action] clause."""
        panel = self._panel(self._bundle())
        action = veo_prompt.panel_action(panel)
        self.assertIn("Alice sets the cup down", action)
        self.assertIn("conveying hands flat on the table", action)
