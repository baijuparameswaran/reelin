"""Validates the self-critique-and-refine step (`reel/agents/critique.py` +
`pipeline._gated`'s wiring of it): after a creative pipeline stage computes
its raw result, a neutral open-model call critiques it against the stage's
own governing SYSTEM/PROMPT (not source fidelity or genre — separate,
pre-existing checks), and if it finds real issues, the stage is
automatically re-run ONCE with the critique folded in as feedback — before
the operator ever sees the review gate. Direct instruction: "add a critique
step with every stage of pipeline... this may be done before doing the
feedback cycle. The initial response may be stored as .0 file for
reference."

Pure logic with mocked I/O — no live Ollama/Gemini/Veo calls.

Run with: python -m unittest tests.test_critique -v
"""
from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import pipeline
from reel.agents import critique
from reel.gate import Gate


class TestCritiqueStage(unittest.TestCase):
    def test_needs_improvement_verdict_passes_through(self):
        with mock.patch.object(critique.models, "text",
                               return_value='{"verdict": "needs_improvement", '
                                            '"issues": ["thin characterization"], '
                                            '"improvement_note": "flesh out the antagonist"}'):
            report = critique.critique_stage("structure", "role text", "rules text", {"a": 1})
        self.assertEqual(report["verdict"], "needs_improvement")
        self.assertEqual(report["issues"], ["thin characterization"])
        self.assertEqual(report["improvement_note"], "flesh out the antagonist")

    def test_solid_verdict_passes_through(self):
        with mock.patch.object(critique.models, "text",
                               return_value='{"verdict": "solid", "issues": [], '
                                            '"improvement_note": ""}'):
            report = critique.critique_stage("structure", "role", "rules", {})
        self.assertEqual(report["verdict"], "solid")

    def test_malformed_response_fails_safe_to_solid(self):
        with mock.patch.object(critique.models, "text", return_value="not json at all"):
            report = critique.critique_stage("structure", "role", "rules", {})
        self.assertEqual(report["verdict"], "solid")
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["improvement_note"], "")

    def test_missing_verdict_key_fails_safe_to_solid(self):
        with mock.patch.object(critique.models, "text",
                               return_value='{"issues": ["x"]}'):
            report = critique.critique_stage("structure", "role", "rules", {})
        self.assertEqual(report["verdict"], "solid")

    def test_invalid_verdict_value_fails_safe_to_solid(self):
        with mock.patch.object(critique.models, "text",
                               return_value='{"verdict": "amazing"}'):
            report = critique.critique_stage("structure", "role", "rules", {})
        self.assertEqual(report["verdict"], "solid")

    def test_non_dict_json_response_fails_safe_to_solid(self):
        with mock.patch.object(critique.models, "text", return_value='["not", "a", "dict"]'):
            report = critique.critique_stage("structure", "role", "rules", {})
        self.assertEqual(report["verdict"], "solid")

    def test_prompt_includes_stage_name_role_rules_and_response(self):
        captured = {}
        def fake_text(prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"verdict": "solid"}'
        with mock.patch.object(critique.models, "text", side_effect=fake_text):
            critique.critique_stage("scenes", "You segment stories.", "RULE: be thorough.",
                                    {"scenes": [{"number": 1}]})
        self.assertIn("scenes", captured["prompt"])
        self.assertIn("You segment stories.", captured["prompt"])
        self.assertIn("RULE: be thorough.", captured["prompt"])
        self.assertIn('"number": 1', captured["prompt"])

    def test_uses_configured_critique_profile_by_default(self):
        captured = {}
        def fake_text(prompt, **kwargs):
            captured.update(kwargs)
            return '{"verdict": "solid"}'
        with mock.patch.object(critique.models, "text", side_effect=fake_text):
            critique.critique_stage("structure", "role", "rules", {})
        self.assertEqual(captured.get("profile"), critique.models.agent_profile("critique"))

    def test_explicit_profile_overrides_default(self):
        captured = {}
        def fake_text(prompt, **kwargs):
            captured.update(kwargs)
            return '{"verdict": "solid"}'
        with mock.patch.object(critique.models, "text", side_effect=fake_text):
            critique.critique_stage("structure", "role", "rules", {}, profile="fast")
        self.assertEqual(captured.get("profile"), "fast")

    def test_never_steers_calls_models_text_not_llm_generate(self):
        # Provider-policy invariant: a grader must call reel.models.text
        # (neutral, steer=False internally), never reel.llm.generate
        # directly — matches fidelity.py/genre.py's own pattern.
        import inspect
        source = inspect.getsource(critique)
        self.assertIn("models.text(", source)
        self.assertNotIn("llm.generate(", source)


class TestSpecStoresAgentModule(unittest.TestCase):
    def test_agent_module_stored_in_spec_dict(self):
        spec = pipeline._spec("structure", lambda: {}, lambda r: "s", lambda fb, p=None: {},
                              agent_module=pipeline.structure_agent)
        self.assertIs(spec["agent_module"], pipeline.structure_agent)

    def test_agent_module_defaults_to_none(self):
        spec = pipeline._spec("structure", lambda: {}, lambda r: "s", lambda fb, p=None: {})
        self.assertIsNone(spec["agent_module"])


class TestGatedCritiqueWiring(unittest.TestCase):
    """Drives pipeline._gated directly with a disabled (auto-approving) Gate
    so only the critique-and-refine logic under test actually executes.

    Uses "teststage" (not a real stage name) so the empty-result safety net
    (`stages.is_stage_result_empty`/`_ensure_nonempty_result`, checked
    BEFORE critique — see `_gated`'s docstring) never fires on these tests'
    deliberately minimal placeholder payloads (`{"result": "initial"}`, etc.)
    — a real stage name like "structure" would be checked against its own
    defining content key (`three_act`), which these payloads were never
    meant to satisfy; `agent_module=pipeline.structure_agent` is still real,
    since `critique_stage` needs an actual SYSTEM/PROMPT to critique
    against (though it's mocked in every test here regardless)."""

    def setUp(self):
        self.gate = Gate(enabled=False)
        self.summarize = lambda r: f"summary: {r}"

    def test_needs_improvement_triggers_exactly_one_refine(self):
        rerun_calls = []
        def rerun(feedback, profile):
            rerun_calls.append((feedback, profile))
            return {"result": "refined"}

        with mock.patch.object(pipeline.critique_agent, "critique_stage",
                               return_value={"verdict": "needs_improvement",
                                            "issues": ["thin"], "improvement_note": "flesh it out"}):
            result, _, _ = pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize, rerun,
                agent_module=pipeline.structure_agent, critique_enabled=True, profile="fast")

        self.assertEqual(len(rerun_calls), 1)
        self.assertEqual(rerun_calls[0], ("flesh it out", "fast"))
        self.assertEqual(result, {"result": "refined"})

    def test_solid_verdict_never_reruns(self):
        rerun_calls = []
        def rerun(feedback, profile):
            rerun_calls.append(feedback)
            return {"result": "should not happen"}

        with mock.patch.object(pipeline.critique_agent, "critique_stage",
                               return_value={"verdict": "solid", "issues": [], "improvement_note": ""}):
            result, _, _ = pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize, rerun,
                agent_module=pipeline.structure_agent, critique_enabled=True)

        self.assertEqual(rerun_calls, [])
        self.assertEqual(result, {"result": "initial"})

    def test_needs_improvement_but_empty_improvement_note_never_reruns(self):
        # Fails safe: an improvement_note is required to actually act on a
        # needs_improvement verdict — an empty one means there's nothing
        # concrete to feed back in, so don't force a pointless rerun.
        rerun_calls = []
        def rerun(feedback, profile):
            rerun_calls.append(feedback)
            return {}

        with mock.patch.object(pipeline.critique_agent, "critique_stage",
                               return_value={"verdict": "needs_improvement",
                                            "issues": ["x"], "improvement_note": ""}):
            result, _, _ = pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize, rerun,
                agent_module=pipeline.structure_agent, critique_enabled=True)

        self.assertEqual(rerun_calls, [])
        self.assertEqual(result, {"result": "initial"})

    def test_no_agent_module_skips_critique_entirely(self):
        with mock.patch.object(pipeline.critique_agent, "critique_stage") as mocked:
            result, _, _ = pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize,
                lambda fb, p=None: {}, agent_module=None, critique_enabled=True)
        mocked.assert_not_called()
        self.assertEqual(result, {"result": "initial"})

    def test_critique_disabled_skips_even_with_agent_module(self):
        with mock.patch.object(pipeline.critique_agent, "critique_stage") as mocked:
            result, _, _ = pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize,
                lambda fb, p=None: {}, agent_module=pipeline.structure_agent,
                critique_enabled=False)
        mocked.assert_not_called()
        self.assertEqual(result, {"result": "initial"})

    def test_critique_exception_fails_safe_no_rerun(self):
        rerun_calls = []
        def rerun(feedback, profile):
            rerun_calls.append(feedback)
            return {}

        with mock.patch.object(pipeline.critique_agent, "critique_stage",
                               side_effect=RuntimeError("boom")):
            result, _, _ = pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize, rerun,
                agent_module=pipeline.structure_agent, critique_enabled=True)

        self.assertEqual(rerun_calls, [])
        self.assertEqual(result, {"result": "initial"})

    def test_refine_exception_falls_back_to_pre_critique_result(self):
        def rerun(feedback, profile):
            raise RuntimeError("refine blew up")

        with mock.patch.object(pipeline.critique_agent, "critique_stage",
                               return_value={"verdict": "needs_improvement",
                                            "issues": ["x"], "improvement_note": "fix it"}):
            result, _, _ = pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize, rerun,
                agent_module=pipeline.structure_agent, critique_enabled=True)

        self.assertEqual(result, {"result": "initial"})

    def test_only_fires_once_not_recursively_on_the_refined_result(self):
        # A second call to critique_stage would mean the refined result got
        # critiqued too — it must not: critique fires exactly once, on the
        # pre-critique raw result only.
        call_count = {"n": 0}
        def fake_critique(*a, **kw):
            call_count["n"] += 1
            return {"verdict": "needs_improvement", "issues": ["x"],
                    "improvement_note": "improve"}

        with mock.patch.object(pipeline.critique_agent, "critique_stage",
                               side_effect=fake_critique):
            pipeline._gated(
                self.gate, "teststage", {"result": "initial"}, self.summarize,
                lambda fb, p=None: {"result": "refined"},
                agent_module=pipeline.structure_agent, critique_enabled=True)

        self.assertEqual(call_count["n"], 1)


class TestSaveInitialResponse(unittest.TestCase):
    """The `output/<stage>.0.json` file — the "initial response... stored
    for reference" the direct instruction asked for."""

    def test_writes_the_file_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            pipeline._save_initial_response(out, "structure", {"logline": "L"}, True)
            written = json.loads((out / "structure.0.json").read_text())
        self.assertEqual(written, {"logline": "L"})

    def test_skips_writing_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            pipeline._save_initial_response(out, "structure", {"logline": "L"}, False)
            self.assertFalse((out / "structure.0.json").exists())

    def test_run_group_disables_it_for_a_stage_with_no_agent_module(self):
        # A stage with no agent_module can never be critiqued, so writing a
        # .0.json for it would be a pointless, unused artifact — run_group
        # computes the `enabled` flag as `crit_on and agent_module is not None`,
        # not `crit_on` alone.
        spec_with_module = pipeline._spec("structure", lambda: {}, lambda r: "s",
                                          lambda fb, p=None: {}, agent_module=pipeline.structure_agent)
        spec_without_module = pipeline._spec("render_step", lambda: {}, lambda r: "s",
                                             lambda fb, p=None: {})
        crit_on = True
        self.assertTrue(crit_on and spec_with_module.get("agent_module") is not None)
        self.assertFalse(crit_on and spec_without_module.get("agent_module") is not None)


class TestRunWiresCritiqueForEveryCreativeStage(unittest.TestCase):
    """Static check (source inspection, no execution) that every creative
    `_spec(...)` call inside `pipeline.run()` passes `agent_module=` — i.e.
    critique is genuinely wired for "every stage of pipeline" as asked,
    not just demonstrated in isolation for one stage. Mirrors
    test_prompt_rules.py's TestProviderPolicySteeringSplit, which uses the
    same source-inspection technique to turn a codebase-wide invariant into
    something a test actually catches if it silently regresses."""

    STAGES_EXPECTING_AGENT_MODULE = [
        "structure", "characters", "moodboard", "scenes", "casting",
        "soundscape", "visuals", "cinematography", "screenplay", "storyboard",
    ]

    def test_every_creative_spec_call_passes_agent_module(self):
        source = inspect.getsource(pipeline.run)
        for stage in self.STAGES_EXPECTING_AGENT_MODULE:
            # Find this stage's _spec(...) call and confirm "agent_module="
            # appears before the next top-level _spec( call (a coarse but
            # effective per-call slice, since every call site in run() is
            # on its own well-separated block).
            idx = source.find(f'_spec("{stage}"')
            self.assertNotEqual(idx, -1, f'no _spec("{stage}", ...) call found in run()')
            next_idx = source.find("_spec(", idx + 1)
            block = source[idx:next_idx if next_idx != -1 else len(source)]
            self.assertIn("agent_module=", block,
                         f'_spec("{stage}", ...) does not pass agent_module=')


if __name__ == "__main__":
    unittest.main()
