"""Validates the `frontier` hosted-text profile — a deliberate, narrow
exception to what used to be an absolute rule ("no Gemini text path exists by
design"), added per direct instruction to reuse the existing Gemini key rather
than stand up a second provider.

Three properties have to hold, and each is a place this could quietly go wrong:

1. THE PROVIDER LIVES ON THE PROFILE. This project's rule is "agents pick a
   profile, never a model name." Opting a stage into the hosted model must
   stay a one-line `agent_profiles` change, with no agent module learning a
   provider or model string.

2. IT DEGRADES, NEVER BREAKS. No API key, or a typo'd provider name, must fall
   back to a local profile and keep running — the same graceful-no-op contract
   image/video already have without a key. The text pipeline still has to work
   fully offline.

3. GRADERS STAY LOCAL. `models.text` (fidelity/genre/critique) must refuse a
   hosted profile. This is the provider-level half of the existing
   `steer=False` neutrality rule, and it matters more now that a creative
   stage can run on Gemini: the grader must not be the same model that wrote
   the artifact.

Offline — no network, no key required. Run with:
    python -m unittest tests.test_frontier_profile -v
"""
from __future__ import annotations

import unittest
from unittest import mock

from reel import gemini, llm, models


class TestProfileCarriesProvider(unittest.TestCase):
    """(1) — provider/fallback_profile parse off the profile body."""

    def test_existing_profiles_default_to_ollama(self):
        for name in ("fast", "quality", "synthesis", "quality_high"):
            with self.subTest(profile=name):
                self.assertEqual(llm.get_profile(name).provider, "ollama")

    def test_frontier_profile_is_configured_and_hosted(self):
        p = llm.get_profile("frontier")
        self.assertEqual(p.provider, "gemini")
        self.assertTrue(p.model.startswith("gemini-"), p.model)
        self.assertEqual(p.fallback_profile, "quality_high")

    def test_frontier_fallback_is_itself_a_real_local_profile(self):
        """A fallback pointing at a hosted or nonexistent profile would make
        the degradation path fail exactly when it's needed most."""
        fb = llm.get_profile(llm.get_profile("frontier").fallback_profile)
        self.assertEqual(fb.provider, "ollama")

    def test_scenes_is_the_only_stage_on_the_hosted_tier(self):
        """`scenes` is opted in deliberately — it's the one stage local
        context length capped outright (~14 scenes at num_ctx 8192) rather
        than merely traded off against. Every OTHER stage staying local is
        the assertion that matters here: hosting a stage is a cost and a
        network dependency, so it should never spread by accident."""
        hosted = {stage for stage, tier in (llm.config().get("agent_profiles") or {}).items()
                  if llm.get_profile(tier).provider != "ollama"}
        self.assertEqual(hosted, {"scenes"})

    def test_no_grader_is_on_the_hosted_tier(self):
        """Belt-and-braces alongside `models.local_profile`: graders must not
        be hosted even by configuration, so a grader is never the same model
        that authored the artifact it's judging."""
        profiles = llm.config().get("agent_profiles") or {}
        for grader in ("fidelity", "critique", "genre", "revision"):
            tier = profiles.get(grader)
            if tier:
                with self.subTest(grader=grader):
                    self.assertEqual(llm.get_profile(tier).provider, "ollama")


class TestGenerateDispatch(unittest.TestCase):
    """(1)/(2) — routing and degradation inside `llm.generate`."""

    def setUp(self):
        llm._warned_providers.clear()
        self.addCleanup(llm._warned_providers.clear)
        self.addCleanup(llm.set_direction, None)

    def test_hosted_profile_routes_to_the_provider(self):
        with mock.patch.object(gemini, "available", return_value=True), \
             mock.patch.object(gemini, "generate_text", return_value="hosted!") as gen:
            out = llm.generate("p", profile="frontier", system="S")
        self.assertEqual(out, "hosted!")
        self.assertEqual(gen.call_args.kwargs["model"],
                         llm.get_profile("frontier").model)

    def test_profile_options_reach_the_provider(self):
        with mock.patch.object(gemini, "available", return_value=True), \
             mock.patch.object(gemini, "generate_text", return_value="x") as gen:
            llm.generate("p", profile="frontier")
        self.assertEqual(gen.call_args.kwargs["max_output_tokens"], 32000)

    def test_as_json_is_forwarded_so_the_response_format_is_constrained(self):
        with mock.patch.object(gemini, "available", return_value=True), \
             mock.patch.object(gemini, "generate_text", return_value="{}") as gen:
            llm.generate("p", profile="frontier", as_json=True)
        self.assertTrue(gen.call_args.kwargs["as_json"])

    def test_steering_direction_reaches_the_hosted_call(self):
        """A hosted creative stage must get the same genre+moodboard steering
        a local one does, or it silently becomes the run's one unsteered stage."""
        llm.set_direction("NOIR, rain-slick streets")
        with mock.patch.object(gemini, "available", return_value=True), \
             mock.patch.object(gemini, "generate_text", return_value="x") as gen:
            llm.generate("p", profile="frontier", system="SYS", steer=True)
        sent = gen.call_args.kwargs["system"]
        self.assertIn("NOIR, rain-slick streets", sent)
        self.assertIn("SYS", sent)

    def test_steer_false_keeps_direction_out(self):
        llm.set_direction("NOIR")
        with mock.patch.object(gemini, "available", return_value=True), \
             mock.patch.object(gemini, "generate_text", return_value="x") as gen:
            llm.generate("p", profile="frontier", system="SYS", steer=False)
        self.assertEqual(gen.call_args.kwargs["system"], "SYS")

    def test_no_key_degrades_to_the_local_fallback(self):
        # Assert on the profile the LOCAL path resolved with, rather than
        # driving a real HTTP call — `urlopen` is stubbed to stop the run right
        # after `resolve_model` has been reached.
        with mock.patch.object(gemini, "available", return_value=False), \
             mock.patch.object(llm, "resolve_model", return_value="qwen3:30b") as rm, \
             mock.patch.object(llm.urllib.request, "urlopen",
                               side_effect=RuntimeError("stop here")):
            with self.assertRaises(RuntimeError):
                llm.generate("p", profile="frontier")
        self.assertEqual(rm.call_args.args[0].name, "quality_high")

    def test_unknown_provider_degrades_rather_than_raising(self):
        bogus = llm.Profile("bogus", "m", provider="mystery",
                            fallback_profile="quality_high")
        self.assertIsNone(
            llm._generate_hosted(bogus, "p", sys_msg=None, as_json=False, schema=None))

    def test_provider_error_propagates_instead_of_silently_going_local(self):
        """A configured, reachable provider that fails is a real error — quietly
        re-running a scene breakdown on a small local model would hide it."""
        with mock.patch.object(gemini, "available", return_value=True), \
             mock.patch.object(gemini, "generate_text",
                               side_effect=RuntimeError("blocked")):
            with self.assertRaises(RuntimeError):
                llm.generate("p", profile="frontier")

    def test_local_profiles_never_touch_the_hosted_path(self):
        with mock.patch.object(gemini, "generate_text") as gen, \
             mock.patch.object(llm.urllib.request, "urlopen",
                               side_effect=RuntimeError("stop here")):
            with self.assertRaises(RuntimeError):
                llm.generate("p", profile="fast")
        gen.assert_not_called()


class TestGradersStayLocal(unittest.TestCase):
    """(3) — the provider-level half of grader independence."""

    def test_hosted_profile_is_forced_down_to_its_local_fallback(self):
        self.assertEqual(models.local_profile("frontier"), "quality_high")

    def test_local_profile_passes_through_untouched(self):
        self.assertEqual(models.local_profile("quality"), "quality")

    def test_none_passes_through(self):
        self.assertIsNone(models.local_profile(None))

    def test_unknown_profile_is_left_for_llm_to_reject(self):
        """Fail-open: this function strips a hosted provider, it doesn't
        second-guess profile resolution."""
        self.assertEqual(models.local_profile("nope"), "nope")

    def test_models_text_pins_a_grader_to_the_local_profile(self):
        with mock.patch.object(models.llm, "generate", return_value="{}") as gen:
            models.text("prompt", profile="frontier")
        self.assertEqual(gen.call_args.kwargs["profile"], "quality_high")
        self.assertFalse(gen.call_args.kwargs["steer"])


class TestConfigValidation(unittest.TestCase):
    """A misspelled `provider:` silently routes back to Ollama, which reads as
    the hosted model performing badly rather than never having been called."""

    def _warn(self, profiles):
        return llm.validate_config({"profiles": profiles})

    def test_flags_unknown_profile_key(self):
        w = self._warn({"x": {"model": "m", "provdier": "gemini"}})
        self.assertTrue(any("profiles.x.provdier" in m for m in w), w)

    def test_flags_unknown_provider_value(self):
        w = self._warn({"x": {"model": "m", "provider": "openai"}})
        self.assertTrue(any("provider" in m and "openai" in m for m in w), w)

    def test_flags_fallback_to_undefined_profile(self):
        w = self._warn({"x": {"model": "m", "provider": "gemini",
                              "fallback_profile": "ghost"}})
        self.assertTrue(any("ghost" in m for m in w), w)

    def test_real_config_is_clean(self):
        self.assertEqual(llm.validate_config(), [])


class TestGeminiTextResponseHandling(unittest.TestCase):
    """Response handling — the cases where reading the text naively is wrong.
    Each would otherwise surface as an unexplained JSON parse failure several
    steps downstream, rather than at the call that caused it."""

    def _resp(self, **over):
        r = {"candidates": [{"finishReason": "STOP",
                             "content": {"parts": [{"text": '{"scenes": []}'}]}}],
             "usageMetadata": {"totalTokenCount": 10}}
        r.update(over)
        return r

    def _run(self, resp, **kw):
        with mock.patch.object(gemini, "_post", return_value=resp):
            return gemini.generate_text("p", **kw)

    def test_returns_text_on_success(self):
        self.assertEqual(self._run(self._resp()), '{"scenes": []}')

    def test_blocked_prompt_raises(self):
        resp = self._resp(promptFeedback={"blockReason": "SAFETY"}, candidates=[])
        with self.assertRaises(RuntimeError) as cm:
            self._run(resp)
        self.assertIn("SAFETY", str(cm.exception))

    def test_truncation_raises_rather_than_yielding_unparseable_json(self):
        resp = self._resp(candidates=[{"finishReason": "MAX_TOKENS",
                                       "content": {"parts": [{"text": '{"scenes": [{'}]}}])
        with self.assertRaises(RuntimeError) as cm:
            self._run(resp, max_output_tokens=100)
        self.assertIn("truncated", str(cm.exception))
        self.assertIn("max_output_tokens", str(cm.exception))

    def test_other_early_stop_raises(self):
        resp = self._resp(candidates=[{"finishReason": "RECITATION",
                                       "content": {"parts": [{"text": "x"}]}}])
        with self.assertRaises(RuntimeError) as cm:
            self._run(resp)
        self.assertIn("RECITATION", str(cm.exception))

    def test_no_candidates_raises(self):
        with self.assertRaises(RuntimeError):
            self._run(self._resp(candidates=[]))

    def test_empty_text_raises(self):
        resp = self._resp(candidates=[{"finishReason": "STOP",
                                       "content": {"parts": [{"text": "   "}]}}])
        with self.assertRaises(RuntimeError):
            self._run(resp)

    def test_as_json_sets_the_response_mime_type(self):
        with mock.patch.object(gemini, "_post", return_value=self._resp()) as post:
            gemini.generate_text("p", as_json=True)
        cfg = post.call_args.args[1]["generationConfig"]
        self.assertEqual(cfg["responseMimeType"], "application/json")

    def test_schema_is_sent_and_implies_json(self):
        schema = {"type": "object"}
        with mock.patch.object(gemini, "_post", return_value=self._resp()) as post:
            gemini.generate_text("p", schema=schema)
        cfg = post.call_args.args[1]["generationConfig"]
        self.assertEqual(cfg["responseSchema"], schema)
        self.assertEqual(cfg["responseMimeType"], "application/json")

    def test_system_goes_in_systemInstruction_not_the_user_turn(self):
        with mock.patch.object(gemini, "_post", return_value=self._resp()) as post:
            gemini.generate_text("p", system="SYS")
        body = post.call_args.args[1]
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "SYS")
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "p")

    def test_plain_call_sends_no_generation_config(self):
        with mock.patch.object(gemini, "_post", return_value=self._resp()) as post:
            gemini.generate_text("p")
        self.assertNotIn("generationConfig", post.call_args.args[1])


class TestPerProfileSourceBudget(unittest.TestCase):
    """`llm.max_chars` — how much SOURCE TEXT a stage may send, by the model
    actually serving it. Before this, the 12,000-char constant applied to
    every stage regardless, so switching `scenes` onto a 1M-context model
    still truncated the story to roughly its first 2,000 words: the cut
    happens in the agent, before the prompt is built, so a profile change
    alone never reached it."""

    def test_local_budget_is_derived_from_each_profiles_num_ctx(self):
        """Hardware-derived per model, not one flat number: `num_ctx` is
        chosen per profile against what fits in VRAM + the CPU-RAM KV cache,
        so a tier configured for more context may read more of the story."""
        by_ctx = {}
        for name in ("fast", "quality", "synthesis", "quality_high"):
            p = llm.get_profile(name)
            by_ctx.setdefault(p.options.get("num_ctx"), set()).add(llm.max_chars(name))
        # Same num_ctx -> same budget; more num_ctx -> strictly more budget.
        for ctx, budgets in by_ctx.items():
            with self.subTest(num_ctx=ctx):
                self.assertEqual(len(budgets), 1, f"num_ctx {ctx} gave {budgets}")
        ordered = sorted((ctx, next(iter(b))) for ctx, b in by_ctx.items())
        self.assertEqual([b for _, b in ordered], sorted(b for _, b in ordered))
        self.assertGreater(llm.max_chars("synthesis"), llm.max_chars("quality_high"))

    def test_the_8k_tiers_stay_close_to_the_old_flat_cap(self):
        """The derivation must not quietly move the profiles this project
        actually runs on — 8192 ctx should land near the hand-tuned 12,000
        that was measured against real prompt sizes."""
        derived = llm.max_chars("quality_high")
        self.assertLess(abs(derived - llm.MAX_CHARS) / llm.MAX_CHARS, 0.10,
                        f"{derived} drifted >10% from {llm.MAX_CHARS}")

    def test_reserves_leave_room_for_prompt_and_response(self):
        """The budget is source text only — the agent's rules and the JSON
        reply must still fit inside the same num_ctx."""
        for name in ("quality_high", "synthesis"):
            with self.subTest(profile=name):
                p = llm.get_profile(name)
                spent = llm.max_chars(name) / llm._CHARS_PER_TOKEN
                self.assertLessEqual(
                    spent + llm._PROMPT_RESERVE_TOKENS + llm._OUTPUT_RESERVE_TOKENS,
                    p.options["num_ctx"])

    def test_tiny_num_ctx_floors_instead_of_going_negative(self):
        p = llm.Profile("tiny", "m", options={"num_ctx": 2048})
        self.assertEqual(llm.local_max_chars(p), llm._MIN_SOURCE_CHARS)

    def test_undeclared_or_malformed_num_ctx_behaves_like_the_default_tier(self):
        for opts in ({}, {"num_ctx": "big"}, {"num_ctx": None}):
            with self.subTest(options=opts):
                self.assertEqual(llm.local_max_chars(llm.Profile("x", "m", options=opts)),
                                 llm.local_max_chars(llm.Profile("y", "m",
                                     options={"num_ctx": llm._DEFAULT_NUM_CTX})))

    def test_hosted_profile_gets_the_provider_budget(self):
        with mock.patch.object(gemini, "available", return_value=True):
            self.assertGreater(llm.max_chars("frontier"), llm.MAX_CHARS)

    def test_unusable_provider_falls_back_to_the_local_budget(self):
        """The case that matters most: `llm.generate` degrades a keyless
        hosted profile to a local model, so the truncation decision has to
        degrade with it — otherwise 600K chars go at an 8K-context model."""
        with mock.patch.object(gemini, "available", return_value=False):
            self.assertEqual(llm.max_chars("frontier"),
                             llm.max_chars(llm.get_profile("frontier").fallback_profile))

    def test_explicit_option_wins_over_everything(self):
        p = llm.Profile("x", "m", provider="gemini",
                        options={"max_source_chars": 4242})
        with mock.patch.object(llm, "get_profile", return_value=p):
            self.assertEqual(llm.max_chars("x"), 4242)

    def test_malformed_explicit_option_degrades_to_the_provider_budget(self):
        p = llm.Profile("x", "m", provider="ollama",
                        options={"max_source_chars": "lots", "num_ctx": 8192})
        with mock.patch.object(llm, "get_profile", return_value=p):
            self.assertEqual(llm.max_chars("x"), llm.local_max_chars(p))

    def test_none_and_unknown_profiles_are_the_safe_floor(self):
        self.assertEqual(llm.max_chars(None), llm.MAX_CHARS)
        self.assertEqual(llm.max_chars("no-such-profile"), llm.MAX_CHARS)

    def test_self_referencing_fallback_cannot_recurse(self):
        p = llm.Profile("loop", "m", provider="gemini", fallback_profile="loop")
        with mock.patch.object(llm, "get_profile", return_value=p), \
             mock.patch.object(gemini, "available", return_value=False):
            self.assertEqual(llm.max_chars("loop"), llm.MAX_CHARS)

    def test_scenes_agent_uses_the_resolved_budget_not_the_constant(self):
        """End to end through the real agent: the text actually placed in the
        prompt is cut at the profile's budget, not at MAX_CHARS."""
        from reel.agents import scenes as scenes_agent
        long_story = "x" * 200_000
        captured = {}

        def fake_generate(prompt, **kw):
            captured["prompt"] = prompt
            return '{"scenes": []}'

        with mock.patch.object(gemini, "available", return_value=True), \
             mock.patch.object(scenes_agent.llm, "generate", side_effect=fake_generate):
            scenes_agent.segment_scenes({"title": "T", "text": long_story},
                                        {"three_act": {}}, profile="frontier")
        self.assertGreater(captured["prompt"].count("x"), llm.MAX_CHARS * 2)
