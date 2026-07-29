"""Validates `reel/hardware_config.py` (the `make setup` step that matches each
profile's model to the host's GPU/RAM) and the `config/models.local.yaml`
overlay it writes (`llm.local_overrides` / the deep merge in `llm.config`).

Why this exists: `config/models.yaml` pins `quality_high: qwen3:30b`, a 19 GB
model chosen against ONE machine (ARCHITECTURE.md's "Hardware reality"). On a
smaller host that's a huge download that then thrashes, and
`llm.resolve_model`'s run-time adaptation can't help — it only picks among
models already pulled, which is decided by the step this module runs before.

The load-bearing guarantees, all covered below: it never invents a model tag
outside the profile's own candidate ladder; it writes NOTHING on the machine
models.yaml is tuned for; and it removes a stale override when the host grows,
rather than pinning a tier to a smaller machine's choice forever.

Pure logic — no network, no Ollama daemon, no API calls (hardware is injected,
and every write goes to a tmpdir).

Run with: python -m unittest tests.test_hardware_config -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from reel import hardware_config, llm


_CFG = {
    "profiles": {
        "fast": {"model": "qwen3:4b", "fallbacks": ["phi3:mini"]},
        "quality": {"model": "qwen3:8b", "fallbacks": ["llama3:8b", "qwen3:4b"]},
        "quality_high": {
            "model": "qwen3:30b",
            "fallbacks": ["qwen3:32b", "gemma3:27b", "gemma3:12b", "qwen3:8b"],
        },
        "frontier": {"provider": "gemini", "model": "gemini-3.6-flash",
                     "fallback_profile": "quality"},
    }
}

# A big host (everything fits) and a small one (only the 4b tier does).
BIG = {"vram_mb": 24_000, "ram_mb": 64_000}
SMALL = {"vram_mb": 0, "ram_mb": 4_000}


def _stub_config():
    """Point both llm.config and the installed-size lookup at the fixture.

    `_installed_model_sizes` is stubbed empty so sizes come from llm's built-in
    table rather than whatever this developer machine happens to have pulled.
    """
    return mock.patch.multiple(
        llm,
        config=mock.DEFAULT,
        _installed_model_sizes=mock.DEFAULT,
    )


class _Base(unittest.TestCase):
    def setUp(self):
        patcher = _stub_config()
        mocks = patcher.start()
        self.addCleanup(patcher.stop)
        mocks["config"].return_value = _CFG
        mocks["_installed_model_sizes"].return_value = {}

    def plan(self, **hw):
        return hardware_config.plan(_CFG, **hw)


class PlanPicksAFittingModel(_Base):
    def test_keeps_tracked_model_when_it_fits(self):
        chosen = {c.profile: c.chosen for c in self.plan(**BIG)}
        self.assertEqual(chosen["quality_high"], "qwen3:30b")
        self.assertFalse(any(c.changed for c in self.plan(**BIG)))

    def test_downgrades_only_the_profiles_that_dont_fit(self):
        # 8 GB VRAM + 12 GB RAM: usable ~17.7 GB, so the 19 GB 30b and 20 GB 32b
        # are out but gemma3:27b (17 GB) still fits.
        got = hardware_config.overrides(self.plan(vram_mb=8_192, ram_mb=12_000))
        self.assertEqual(got, {"quality_high": "gemma3:27b"})

    def test_walks_the_ladder_down_on_a_tiny_host(self):
        got = hardware_config.overrides(self.plan(**SMALL))
        self.assertEqual(got, {"quality": "qwen3:4b", "quality_high": "qwen3:8b"})

    def test_only_ever_chooses_from_the_profiles_own_candidates(self):
        for hw in (BIG, SMALL, {"vram_mb": 8_192, "ram_mb": 12_000}):
            for c in self.plan(**hw):
                allowed = [_CFG["profiles"][c.profile]["model"],
                           *_CFG["profiles"][c.profile].get("fallbacks", [])]
                self.assertIn(c.chosen, allowed, f"{c.profile} on {hw}")

    def test_falls_back_to_the_smallest_candidate_when_nothing_fits(self):
        tiny = self.plan(vram_mb=0, ram_mb=256)
        fast = next(c for c in tiny if c.profile == "fast")
        self.assertEqual(fast.chosen, "phi3:mini")   # 2,300 MB, the smallest
        self.assertFalse(fast.fits)
        self.assertIn("WARNING", fast.describe())

    def test_skips_hosted_profiles(self):
        # A Gemini model runs on someone else's hardware — local VRAM says
        # nothing about it, and its tag isn't a pullable Ollama model.
        self.assertNotIn("frontier", {c.profile for c in self.plan(**SMALL)})


class ApplyWritesTheOverlay(_Base):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "models.local.yaml"

    def _apply(self, **hw):
        return hardware_config.apply(self.plan(**hw), self.path, hardware="test host")

    def test_writes_nothing_on_a_host_the_tracked_config_already_fits(self):
        self._apply(**BIG)
        self.assertFalse(self.path.exists())

    def test_writes_only_the_model_field(self):
        self._apply(**SMALL)
        data = yaml.safe_load(self.path.read_text())
        self.assertEqual(data["profiles"]["quality_high"], {"model": "qwen3:8b"})
        # fallbacks/options are NOT copied down — the deep merge keeps them.
        self.assertNotIn("fallbacks", data["profiles"]["quality_high"])

    def test_untouched_profiles_are_absent_from_the_overlay(self):
        self._apply(**SMALL)
        data = yaml.safe_load(self.path.read_text())
        self.assertNotIn("fast", data["profiles"])

    def test_file_explains_itself(self):
        self._apply(**SMALL)
        text = self.path.read_text()
        self.assertIn("GENERATED", text)
        self.assertIn("hardware_config", text)
        self.assertIn("test host", text)

    def test_stale_override_is_removed_when_the_host_grows(self):
        self._apply(**SMALL)
        self.assertTrue(self.path.exists())
        self._apply(**BIG)          # re-run after a hardware upgrade
        self.assertFalse(self.path.exists())

    def test_hand_added_keys_survive_a_rewrite(self):
        self.path.write_text(yaml.safe_dump({
            "ollama_host": "http://box:11434",
            "profiles": {"fast": {"options": {"temperature": 0.1}}},
        }))
        self._apply(**SMALL)
        data = yaml.safe_load(self.path.read_text())
        self.assertEqual(data["ollama_host"], "http://box:11434")
        self.assertEqual(data["profiles"]["fast"], {"options": {"temperature": 0.1}})
        self.assertEqual(data["profiles"]["quality_high"]["model"], "qwen3:8b")

    def test_a_profiles_stale_model_is_dropped_without_taking_its_neighbours(self):
        self.path.write_text(yaml.safe_dump({
            "profiles": {"quality_high": {"model": "gemma3:12b",
                                          "options": {"num_ctx": 4096}}},
        }))
        self._apply(**BIG)
        data = yaml.safe_load(self.path.read_text())
        self.assertEqual(data["profiles"]["quality_high"], {"options": {"num_ctx": 4096}})


class PlanIgnoresAnOverrideAlreadyOnDisk(unittest.TestCase):
    """The override on disk must never become the baseline it's judged against.

    Caught live: reading the MERGED config made a previously-written downgrade
    look like the configured model, so it reported "ok" and the tier could
    never be restored after a hardware upgrade.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "models.local.yaml"
        llm.config.cache_clear()
        self.addCleanup(llm.config.cache_clear)
        patcher = mock.patch.multiple(
            llm, LOCAL_CONFIG_PATH=self.path,
            base_config=mock.Mock(return_value=_CFG),
            _installed_model_sizes=mock.Mock(return_value={}))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_upgrade_path_restores_the_tracked_model(self):
        hardware_config.apply(hardware_config.plan(**SMALL), self.path, hardware="small")
        self.assertEqual(yaml.safe_load(self.path.read_text())["profiles"]
                         ["quality_high"]["model"], "qwen3:8b")

        # Same machine, more RAM: the plan must see qwen3:30b as configured
        # (from the tracked file), not qwen3:8b (from the overlay it just wrote).
        after = {c.profile: c for c in hardware_config.plan(**BIG)}
        self.assertEqual(after["quality_high"].configured, "qwen3:30b")
        self.assertEqual(after["quality_high"].chosen, "qwen3:30b")
        hardware_config.apply(list(after.values()), self.path, hardware="big")
        self.assertFalse(self.path.exists())


class OverlayMergesIntoConfig(unittest.TestCase):
    """`llm.config()` layering — the half that makes the overlay take effect."""

    def setUp(self):
        # config() is lru_cached for the process, so a test that swaps the
        # overlay path has to invalidate it — before AND after, so a stubbed
        # config never leaks into the next test.
        llm.config.cache_clear()
        self.addCleanup(llm.config.cache_clear)

    def test_deep_merge_is_field_level(self):
        base = {"profiles": {"quality_high": {"model": "qwen3:30b",
                                              "fallbacks": ["qwen3:8b"],
                                              "options": {"num_ctx": 8192,
                                                          "temperature": 0.8}}}}
        merged = llm._deep_merge(base, {"profiles": {"quality_high": {"model": "gemma3:12b"}}})
        entry = merged["profiles"]["quality_high"]
        self.assertEqual(entry["model"], "gemma3:12b")
        self.assertEqual(entry["fallbacks"], ["qwen3:8b"])
        self.assertEqual(entry["options"], {"num_ctx": 8192, "temperature": 0.8})

    def test_deep_merge_does_not_mutate_the_base(self):
        base = {"profiles": {"fast": {"model": "qwen3:4b"}}}
        llm._deep_merge(base, {"profiles": {"fast": {"model": "phi3:mini"}}})
        self.assertEqual(base["profiles"]["fast"]["model"], "qwen3:4b")

    def test_config_applies_the_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.local.yaml"
            path.write_text(yaml.safe_dump({"profiles": {"fast": {"model": "phi3:mini"}}}))
            with mock.patch.object(llm, "LOCAL_CONFIG_PATH", path):
                self.assertEqual(llm.get_profile("fast").model, "phi3:mini")
                # Untouched fields still come from the tracked file.
                self.assertTrue(llm.get_profile("fast").fallbacks)

    def test_missing_overlay_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(llm, "LOCAL_CONFIG_PATH", Path(tmp) / "nope.yaml"):
                self.assertEqual(llm.local_overrides(), {})
                self.assertIn("profiles", llm.config())

    def test_malformed_overlay_degrades_to_the_tracked_config(self):
        # Best-effort like every other config read here: a broken per-host file
        # must not take the whole pipeline down.
        for bad in ("{{{ not yaml", "- a\n- list\n", ""):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "models.local.yaml"
                path.write_text(bad)
                with mock.patch.object(llm, "LOCAL_CONFIG_PATH", path):
                    self.assertEqual(llm.local_overrides(), {}, f"for {bad!r}")
                    self.assertIn("profiles", llm.config())


class OverlayIsGitignored(unittest.TestCase):
    def test_generated_overlay_is_not_tracked(self):
        # It's per-host and generated; committing it would push one machine's
        # downgrades onto every other clone.
        root = Path(__file__).resolve().parent.parent
        self.assertIn("config/models.local.yaml", (root / ".gitignore").read_text())


if __name__ == "__main__":
    unittest.main()
