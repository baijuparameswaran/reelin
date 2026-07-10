"""Validates that agent PROMPT templates actually follow this project's
established prompting conventions, and that the deterministic post-processing
which enforces some of those rules does what the prompts describe.

Two kinds of checks, both pure text/logic — no LLM or API calls, so this
suite is fully offline and fast:

  1. PROMPT TEXT checks: every "sandwiched" prompt (see AGENTS.md's
     "Sandwiched?" column) actually contains its reminder block, positioned
     at the true end of the rendered string with no leftover unresolved
     `{placeholder}`s; every prompt's highest-stakes rule keywords (SOURCE
     FIDELITY, ISOLATION, MINIMIZE SCENE COUNT, TIE-BREAKER, SOURCE OVER
     COVERAGE, ...) are actually present in the text sent to the model, not
     just described in a docstring.

  2. ENFORCEMENT checks: several rules stated in a prompt are backed by a
     deterministic Python function that catches the case a model doesn't
     reliably follow through prompting alone (documented precedent in this
     project — see PROGRESS.md's session log for the real "Woman" vs.
     "Young Woman" character-name-drift incident). Those functions
     (`scenes._validate`, `scenes._attach_source_excerpts`,
     `scenes._reconcile_character_names`, `casting._location_entries`,
     `casting._prop_entries`, `storyboard._build_scene_board`,
     `duration_budget.suggest_shots_per_scene`) are tested directly here.

Run with: python -m unittest discover -s tests -v
      or: python -m unittest tests.test_prompt_rules -v
"""
from __future__ import annotations

import inspect
import re
import unittest

from reel import duration_budget
from reel.agents import (
    casting,
    characters as characters_agent,
    cinematography,
    fidelity,
    genre as genre_agent,
    moodboard as moodboard_agent,
    revision,
    scenes,
    screenplay,
    soundscape,
    storyboard,
    structure as structure_agent,
    visuals,
)

SANDWICH_MARKER = "Before you respond"


def leftover_placeholders(text: str) -> list[str]:
    """Any `{word}`-shaped substring still in the string after `.format()` —
    a real bug (a kwarg the caller forgot to supply, or a literal brace in
    added prompt text that needed `{{`/`}}` escaping)."""
    return re.findall(r"\{[a-zA-Z_]+\}", text)


class SandwichAssertionsMixin:
    """Shared assertions for the seven prompts this project's 2026-07-09
    "prompt-pitfall audit" gave a lost-in-the-middle reminder block to
    (scenes, casting, soundscape, visuals, cinematography, screenplay,
    storyboard — see AGENTS.md's reference table)."""

    def assert_sandwiched(self, rendered: str, expected_tail_substr: str):
        self.assertIn(SANDWICH_MARKER, rendered,
                      "expected the lost-in-the-middle reminder block")
        self.assertTrue(
            rendered.rstrip().endswith(expected_tail_substr),
            "reminder block isn't the true tail of the prompt "
            f"— got ...{rendered.rstrip()[-100:]!r}",
        )
        leftover = leftover_placeholders(rendered)
        self.assertFalse(leftover, f"unresolved placeholders: {leftover}")

    def assert_not_sandwiched(self, rendered: str):
        self.assertNotIn(SANDWICH_MARKER, rendered,
                         "this prompt was deliberately left un-sandwiched "
                         "(short, data-last, no multi-rule list) — if it now "
                         "needs one, update AGENTS.md's table too")


# ─────────────────────────── scenes.py ──────────────────────────────────

class TestScenesPrompt(SandwichAssertionsMixin, unittest.TestCase):
    def _render(self, **overrides):
        kwargs = dict(target="as few as possible", beats="{}", title="T",
                      text="some source text", character_names_block="")
        kwargs.update(overrides)
        return scenes.PROMPT.format(**kwargs)

    def test_sandwiched(self):
        self.assert_sandwiched(self._render(), "not one you inferred would look good on screen.")

    def test_source_is_the_only_authority_rule_present(self):
        self.assertIn("SOURCE TEXT IS THE ONLY AUTHORITY", self._render())

    def test_minimize_scene_count_rule_present(self):
        out = self._render()
        self.assertIn("MINIMIZE SCENE COUNT", out)
        # Reinforced in both the task-intro line and the rules list, not just once.
        self.assertGreaterEqual(out.count("scene"), 10)

    def test_props_field_in_schema(self):
        self.assertIn('"props":', self._render())

    def test_location_consistency_rule_present(self):
        self.assertIn("MUST use the exact identical `location` string", self._render())

    def test_canonical_names_block_appears_when_characters_given(self):
        block = scenes._canonical_names_block({"characters": [{"name": "Marcel"}]})
        self.assertIn("Marcel", block)
        rendered = self._render(character_names_block=block)
        self.assertIn("CANONICAL CHARACTER NAMES", rendered)
        self.assertIn("Marcel", rendered)

    def test_canonical_names_block_empty_when_no_characters(self):
        self.assertEqual(scenes._canonical_names_block(None), "")
        self.assertEqual(scenes._canonical_names_block({}), "")


class TestScenesValidate(unittest.TestCase):
    """Enforces prompt rule 3 ("source_line is mandatory... If you cannot
    find a matching phrase, the scene does not belong in the list") —
    prompt instruction alone can't guarantee this, so `_validate` checks it
    deterministically after the fact."""

    def test_drops_scene_with_hallucinated_source_line(self):
        source = "Marcel walked into the bar and ordered a whiskey."
        good = {"number": 1, "source_line": "Marcel walked into the bar"}
        bad = {"number": 2, "source_line": "a dragon burst through the ceiling"}
        valid, dropped = scenes._validate([good, bad], source)
        self.assertEqual([s["number"] for s in valid], [1])
        self.assertEqual([s["number"] for s in dropped], [2])
        self.assertIn("drop_reason", dropped[0])

    def test_keeps_real_quote_spanning_a_hard_linewrap(self):
        # The ingested source keeps literal line-wraps; a model-written
        # source_line naturally uses normal spacing — must not be flagged
        # as hallucinated just because of that whitespace mismatch.
        source = "Marcel\nhad watched the tide for an hour."
        scene = {"number": 1, "source_line": "Marcel had watched the tide"}
        valid, dropped = scenes._validate([scene], source)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(dropped), 0)


class TestAttachSourceExcerpts(unittest.TestCase):
    """Enforces the module docstring's claim that every scene captures "the
    actual PORTION of the story that defined it" — a deterministic,
    non-overlapping partition of the source text by scene."""

    SOURCE = (
        "Marcel walked into the bar and ordered a whiskey. "
        "Later that night, Marcel drove home through the rain. "
        "The next morning, Elena woke up and made coffee."
    )

    def test_partitions_contiguously_in_source_order(self):
        scene_list = [
            {"number": 1, "source_line": "Marcel walked into the bar"},
            {"number": 2, "source_line": "Marcel drove home through the rain"},
            {"number": 3, "source_line": "Elena woke up and made coffee"},
        ]
        result = scenes._attach_source_excerpts(scene_list, self.SOURCE)
        self.assertTrue(result[0]["source_excerpt"].startswith("Marcel walked into the bar"))
        self.assertTrue(result[1]["source_excerpt"].startswith("Marcel drove home"))
        self.assertTrue(result[2]["source_excerpt"].startswith("Elena woke up"))
        # Last scene's excerpt runs to the end of the text.
        self.assertTrue(result[2]["source_excerpt"].rstrip().endswith("made coffee."))
        self.assertTrue(all(s["word_count"] > 0 for s in result))

    def test_order_by_position_not_by_number(self):
        # Scene "numbers" out of source order shouldn't break the partition —
        # excerpts are computed from where source_line actually occurs.
        scene_list = [
            {"number": 2, "source_line": "Elena woke up and made coffee"},
            {"number": 1, "source_line": "Marcel walked into the bar"},
        ]
        result = scenes._attach_source_excerpts(scene_list, self.SOURCE)
        by_number = {s["number"]: s for s in result}
        self.assertTrue(by_number[1]["source_excerpt"].startswith("Marcel walked into the bar"))
        self.assertIn("Elena", by_number[2]["source_excerpt"])

    def test_unfindable_source_line_degrades_to_empty_not_a_crash(self):
        scene_list = [{"number": 9, "source_line": "this never appears anywhere"}]
        result = scenes._attach_source_excerpts(scene_list, self.SOURCE)
        self.assertEqual(result[0]["source_excerpt"], "")
        self.assertEqual(result[0]["word_count"], 0)


class TestReconcileCharacterNames(unittest.TestCase):
    """Deterministic safety net for prompt rule 5 (reuse the exact
    canonical name) — the real "Woman" vs. canonical "Young Woman" incident
    this was built to catch, per PROGRESS.md's session log."""

    def test_fixes_unambiguous_subset_match(self):
        characters = {"characters": [{"name": "Young Woman"}]}
        result = scenes._reconcile_character_names(
            [{"characters": ["Woman"]}], characters)
        self.assertEqual(result[0]["characters"], ["Young Woman"])

    def test_leaves_ambiguous_match_untouched(self):
        characters = {"characters": [{"name": "Young Woman"}, {"name": "Old Woman"}]}
        result = scenes._reconcile_character_names(
            [{"characters": ["Woman"]}], characters)
        self.assertEqual(result[0]["characters"], ["Woman"])

    def test_leaves_unrelated_name_untouched(self):
        characters = {"characters": [{"name": "Young Woman"}]}
        result = scenes._reconcile_character_names(
            [{"characters": ["Marcel"]}], characters)
        self.assertEqual(result[0]["characters"], ["Marcel"])

    def test_noop_when_no_characters_given(self):
        original = [{"characters": ["Woman"]}]
        result = scenes._reconcile_character_names(original, None)
        self.assertEqual(result[0]["characters"], ["Woman"])


# ─────────────────────────── casting.py ─────────────────────────────────

class TestCastingPrompt(SandwichAssertionsMixin, unittest.TestCase):
    def _render(self, **overrides):
        kwargs = dict(logline="L", genre="Drama", tone="melancholic",
                      characters="[]", locations_block="", props_block="")
        kwargs.update(overrides)
        return casting.PROMPT.format(**kwargs)

    def test_sandwiched(self):
        self.assert_sandwiched(self._render(), "generic description.")

    def test_isolation_rule_covers_person_and_prop(self):
        out = self._render()
        self.assertIn("ISOLATION (strict", out)
        self.assertIn("prop entries", out)

    def test_story_fidelity_rule_present(self):
        self.assertIn("STORY FIDELITY", self._render())

    def test_gender_rule_present(self):
        self.assertIn("GENDER:", self._render())

    def test_props_rule_present(self):
        out = self._render()
        self.assertIn("PROPS:", out)
        self.assertIn("VERY DESCRIPTIVE", out)

    def test_recurring_props_guidance_present(self):
        self.assertIn("recurring_props", self._render())

    def test_invent_actor_vs_story_fidelity_conflict_clarified(self):
        # 2026-07-09 audit: "invent the actor" must not read as license to
        # invent physical attributes too.
        self.assertIn("does NOT relax the STORY FIDELITY rule", self._render())


class TestLocationEntries(unittest.TestCase):
    """Enforces the LOCATIONS rule's `recurring_props` heuristic: only a
    prop plausibly FIXED to the location (recurs across multiple scenes set
    there, or the location has just one scene) is offered as a candidate —
    a single-scene prop at a multi-scene location must not leak in, since
    baking it into the reused reference would wrongly force it into every
    OTHER scene at that place."""

    def test_prop_recurring_at_multi_scene_location_survives(self):
        scenes_doc = {"scenes": [
            {"number": 1, "location": "Bar", "props": ["a brass mirror", "a letter"]},
            {"number": 2, "location": "Bar", "props": ["a brass mirror"]},
        ]}
        entries = casting._location_entries(scenes_doc)
        bar = next(e for e in entries if e["name"] == "Bar")
        self.assertEqual(bar["recurring_props"], ["a brass mirror"])
        self.assertNotIn("a letter", bar["recurring_props"])

    def test_props_at_single_scene_location_all_survive(self):
        scenes_doc = {"scenes": [
            {"number": 1, "location": "Lighthouse", "props": ["an old brass lamp"]},
        ]}
        entries = casting._location_entries(scenes_doc)
        lighthouse = next(e for e in entries if e["name"] == "Lighthouse")
        self.assertEqual(lighthouse["recurring_props"], ["an old brass lamp"])

    def test_one_entry_per_distinct_location_regardless_of_scene_count(self):
        scenes_doc = {"scenes": [
            {"number": i, "location": "Bar", "slugline": f"INT. BAR - DAY {i}"}
            for i in range(1, 6)
        ]}
        entries = casting._location_entries(scenes_doc)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "location")


class TestPropEntries(unittest.TestCase):
    """Enforces the PROPS rule's "cast only what recurs" scope: a prop
    mentioned in 2+ DISTINCT scenes ANYWHERE in the story (not scoped to
    one location — unlike `_location_entries`'s `recurring_props` — a
    portable prop like a character's watch can travel across locations)."""

    def test_prop_recurring_across_different_locations_is_cast(self):
        scenes_doc = {"scenes": [
            {"number": 1, "location": "Bar", "props": ["Marcel's pocket watch"]},
            {"number": 2, "location": "Road", "props": ["Marcel's pocket watch"]},
        ]}
        entries = casting._prop_entries(scenes_doc)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "Marcel's pocket watch")
        self.assertEqual(entries[0]["kind"], "prop")

    def test_single_scene_prop_is_not_cast(self):
        scenes_doc = {"scenes": [
            {"number": 1, "location": "Bar", "props": ["a spilled drink"]},
        ]}
        self.assertEqual(casting._prop_entries(scenes_doc), [])

    def test_sorted_by_recurrence_descending(self):
        scenes_doc = {"scenes": [
            {"number": 1, "props": ["watch", "ring"]},
            {"number": 2, "props": ["watch"]},
            {"number": 3, "props": ["watch", "ring"]},
        ]}
        entries = casting._prop_entries(scenes_doc)
        self.assertEqual([e["name"] for e in entries], ["watch", "ring"])


# ──────────────────── soundscape / visuals / cinematography ─────────────

class TestSoundscapePrompt(SandwichAssertionsMixin, unittest.TestCase):
    def test_sandwiched(self):
        rendered = soundscape.PROMPT.format(
            logline="L", genre="G", tone="T", themes="x", scenes="[]")
        self.assert_sandwiched(rendered, "renumbered.")

    def test_location_consistency_rule_present(self):
        rendered = soundscape.PROMPT.format(
            logline="L", genre="G", tone="T", themes="x", scenes="[]")
        self.assertIn("share the same base ambient_bed", rendered)


class TestVisualsPrompt(SandwichAssertionsMixin, unittest.TestCase):
    def test_sandwiched(self):
        rendered = visuals.PROMPT.format(
            logline="L", genre="G", tone="T", themes="x", scenes="[]")
        self.assert_sandwiched(rendered, "renumbered.")

    def test_location_consistency_rule_present(self):
        rendered = visuals.PROMPT.format(
            logline="L", genre="G", tone="T", themes="x", scenes="[]")
        self.assertIn("share the same base color_palette", rendered)

    def test_key_props_grounded_by_scenes_props_field(self):
        rendered = visuals.PROMPT.format(
            logline="L", genre="G", tone="T", themes="x", scenes="[]")
        self.assertIn("source-grounded inventory", rendered)


class TestCinematographyPrompt(SandwichAssertionsMixin, unittest.TestCase):
    def _render(self, **overrides):
        kwargs = dict(logline="L", genre="G", tone="T", themes="x",
                      duration_rule="", scenes="[]")
        kwargs.update(overrides)
        return cinematography.PROMPT.format(**kwargs)

    def test_sandwiched(self):
        self.assert_sandwiched(self._render(), "renumbered.")

    def test_shot_floor_is_one_not_two(self):
        out = self._render()
        self.assertIn("at least 1 shot", out)
        self.assertNotIn("at least 2 shots", out)

    def test_tie_breaker_present(self):
        out = self._render()
        self.assertIn("TIE-BREAKER", out)
        self.assertIn("location rule above and the motif rule above", out)

    def test_duration_rule_injected_when_given(self):
        out = self._render(duration_rule=" — aim for about 3 shots per scene")
        self.assertIn("aim for about 3 shots per scene", out)


class TestDurationBudgetShotFloor(unittest.TestCase):
    """Enforces that `suggest_shots_per_scene` (fed into cinematography.py
    as `duration_rule`) never floors below 1 shot/scene — a scene only
    needs one decisive shot when the beat is simple; a 2-shot floor would
    silently double the guided count for a short --target-duration."""

    def test_short_target_floors_at_one_not_two(self):
        guidance = duration_budget.suggest_shots_per_scene(10, 10)
        self.assertIn("about 1 shot", guidance)

    def test_empty_when_no_scenes(self):
        self.assertEqual(duration_budget.suggest_shots_per_scene(45, 0), "")


# ─────────────────────────── screenplay.py ───────────────────────────────

class TestScreenplayPrompt(SandwichAssertionsMixin, unittest.TestCase):
    def _render(self, **overrides):
        kwargs = dict(revision_note="", logline="L", tone="T", story_block="",
                      characters="c", casting_block="", location_block="",
                      prior_scenes_block="", slugline="INT. X - DAY",
                      scene_number=3, summary="s", purpose="p",
                      soundscape_block="", visuals_block="", cinema_block="")
        kwargs.update(overrides)
        return screenplay.PROMPT.format(**kwargs)

    def test_sandwiched(self):
        self.assert_sandwiched(self._render(), "names given above.")

    def test_source_over_coverage_tie_breaker_present(self):
        out = self._render()
        self.assertIn("SOURCE OVER COVERAGE", out)

    def test_scene_number_placeholder_resolves_and_is_reinforced(self):
        out = self._render(scene_number=7)
        self.assertIn("is exactly 7", out)
        self.assertNotIn("{scene_number}", out)

    def test_economy_rule_prefers_voiceover(self):
        self.assertIn("KEEP ACTUAL SPOKEN DIALOGUE TO A MINIMUM", self._render())


# ─────────────────────────── storyboard.py ───────────────────────────────

class TestStoryboardPrompt(SandwichAssertionsMixin, unittest.TestCase):
    def _render(self, **overrides):
        kwargs = dict(logline="L", genre="G", tone="T", story_block="", bundles="[]")
        kwargs.update(overrides)
        return storyboard.PROMPT.format(**kwargs)

    def test_sandwiched(self):
        self.assert_sandwiched(self._render(), "exactly.")

    def test_self_contained_vs_brief_clarification_present(self):
        self.assertIn("SELF-CONTAINED VS. BRIEF", self._render())

    def test_key_props_in_schema_and_merge_source(self):
        out = self._render()
        self.assertIn('"key_props":', out)
        self.assertIn("visual_overview.key_props", out)

    def test_dialogue_locked_rule_covers_voiceover_separately(self):
        out = self._render()
        self.assertIn("Dialogue is LOCKED", out)
        self.assertIn("voiceover` is a SEPARATE field", out)


class TestBuildSceneBoardDeterministic(unittest.TestCase):
    """Enforces the module docstring's claim that the default (no-feedback)
    build path makes zero LLM calls and every field traces to an upstream
    artifact — including that `key_props` (visuals.json) actually reaches
    `visual_overview`, the real gap fixed in the 2026-07-10 session."""

    def _bundle(self, key_props):
        return {
            "scene_number": 1, "slugline": "INT. BAR - NIGHT",
            "slugline_parsed": {"int_ext": "INT", "location": "BAR", "time_of_day": "NIGHT"},
            "summary": "s", "purpose": "p", "characters_in_scene": ["Marcel"],
            "cast": [{"name": "Marcel", "physical_form": "weathered man"}],
            "location": {"name": "Rusty Anchor Bar", "visual_prompt": "a dim dive bar"},
            "art": {"color_palette": "warm amber", "lighting": "dim practicals",
                    "visual_filter": "grainy", "key_props": key_props,
                    "visual_moments": [], "emotional_function": "melancholic"},
            "audio": {"ambient_bed": "", "silence": False, "sound_events": [],
                      "score_direction": "", "emotional_function": ""},
            "camera": {"coverage": "", "transition_to_next": "", "shots": [
                {"shot_number": 1, "moment": "m", "type": "WS", "angle": "EYE LEVEL",
                 "movement": "STATIC", "lens": "24mm", "framing": "wide",
                 "emotional_function": ""},
            ]},
            "screenplay_shots": [{"shot": 1, "shot_type": "WS",
                                  "description": "Marcel enters.", "voiceover": None,
                                  "dialogue": [], "sound": ""}],
            "moodboard_tile": None,
        }

    def test_key_props_copied_verbatim_into_visual_overview(self):
        bundle = self._bundle([{"prop": "a brass mirror", "function": "reflects isolation"}])
        board = storyboard._build_scene_board(bundle)
        self.assertEqual(board["visual_overview"]["key_props"], ["a brass mirror"])

    def test_empty_prop_names_filtered(self):
        bundle = self._bundle([{"prop": "", "function": "should be dropped"}])
        board = storyboard._build_scene_board(bundle)
        self.assertEqual(board["visual_overview"]["key_props"], [])

    def test_no_key_props_yields_empty_list_not_missing_key(self):
        bundle = self._bundle([])
        board = storyboard._build_scene_board(bundle)
        self.assertIn("key_props", board["visual_overview"])
        self.assertEqual(board["visual_overview"]["key_props"], [])


# ────────────── short, deliberately un-sandwiched prompts ────────────────

class TestShortPromptsRemainUnsandwiched(SandwichAssertionsMixin, unittest.TestCase):
    """Regression guard for the 2026-07-09 audit's explicit decision: these
    four prompts are short, schema/rules-then-data-last, with no multi-rule
    list ahead of a large data block — the lost-in-the-middle risk that
    motivated sandwiching elsewhere doesn't apply, so they were deliberately
    left as-is. If one of these ever grows enough rules to need a reminder
    block, that's a deliberate choice to make (and AGENTS.md's table to
    update) — this test just stops it from silently drifting either way."""

    def test_characters_prompt(self):
        out = characters_agent.PROMPT.format(title="T", text="story text")
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])

    def test_structure_prompt(self):
        out = structure_agent.PROMPT.format(title="T", text="story text")
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])

    def test_genre_determine_prompt(self):
        out = genre_agent.DETERMINE_PROMPT.format(hint="Infer the genre.", story="a story")
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])

    def test_genre_enforce_prompt(self):
        out = genre_agent.ENFORCE_PROMPT.format(
            stage="scenes", genre="Drama, melancholic", genre_name="Drama", artifact="{}")
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])

    def test_moodboard_prompt(self):
        out = moodboard_agent.PROMPT.format(
            revision_note="", logline="L", genre="Drama", tone="T", themes="x",
            genre_block="", story_block="", tiles=4)
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])


class TestGraderPromptsRemainUnsandwiched(SandwichAssertionsMixin, unittest.TestCase):
    """Same rationale as above, for the two grader prompts."""

    def test_fidelity_stage_prompt(self):
        out = fidelity.STAGE_PROMPT.format(stage="scenes", story="a story", artifact="{}")
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])

    def test_fidelity_holistic_prompt(self):
        out = fidelity.PROMPT.format(story="a story", screenplay="FADE IN:", storyboard="{}")
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])

    def test_revision_ripple_prompt(self):
        out = revision.RIPPLE_PROMPT.format(
            story="a story", scenes="[]", changed_scene_numbers=[1], change_summary="x")
        self.assert_not_sandwiched(out)
        self.assertEqual(leftover_placeholders(out), [])


# ────────────────── provider policy: steer vs. neutral ───────────────────

class TestProviderPolicySteeringSplit(unittest.TestCase):
    """Enforces ARCHITECTURE.md's provider-policy rule at the source level:
    creative agents call `llm.generate` directly (steered — they receive the
    process-wide creative direction), while graders and the two
    direction-setting agents (genre/moodboard) call `models.text`
    (`steer=False`, neutral) so they never grade or set direction using the
    very direction they're supposed to judge/define. A static source check,
    not a live call — this is the same split AGENTS.md's "Steered?" column
    documents, tested here as an actual invariant rather than just prose."""

    STEERED_MODULES = (
        scenes, casting, soundscape, visuals, cinematography, screenplay,
        storyboard, characters_agent, structure_agent,
    )
    NEUTRAL_MODULES = (genre_agent, moodboard_agent, fidelity, revision)

    def test_creative_agents_call_llm_generate(self):
        for mod in self.STEERED_MODULES:
            with self.subTest(module=mod.__name__):
                src = inspect.getsource(mod)
                self.assertIn("llm.generate(", src,
                              f"{mod.__name__} should be steered via llm.generate")

    def test_neutral_agents_call_models_text_never_llm_generate(self):
        for mod in self.NEUTRAL_MODULES:
            with self.subTest(module=mod.__name__):
                src = inspect.getsource(mod)
                self.assertIn("models.text(", src,
                              f"{mod.__name__} should call the neutral models.text")
                self.assertNotIn("llm.generate(", src,
                                 f"{mod.__name__} must not call llm.generate directly "
                                 "(would bypass steer=False and pick up creative direction)")


if __name__ == "__main__":
    unittest.main()
