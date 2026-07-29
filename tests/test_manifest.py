"""Validates `reel/manifest.py` — the single source of truth for WHICH models
get pulled (`make setup`/`make setup-models`/`scripts/update-models.sh` all
consume its output and hand each line straight to `ollama pull`).

The load-bearing case is the hosted frontier tier: a profile whose `provider`
isn't Ollama carries a provider-side model name (`gemini-3.6-flash`), which is
not a pullable Ollama tag. Emitted here it would have made every setup/update
run attempt `ollama pull gemini-3.6-flash` and log a spurious failure.

Pure logic — no network, no Ollama daemon, no API calls (profiles come from a
stubbed config, and the real-config check reads config/models.yaml off disk).

Run with: python -m unittest tests.test_manifest -v
"""
from __future__ import annotations

import unittest
from unittest import mock

from reel import llm, manifest


_PROFILES = {
    "fast": {"model": "qwen3:4b", "fallbacks": ["qwen2.5:latest"]},
    "quality": {"model": "qwen3:8b", "fallbacks": ["llama3:8b", "qwen2.5:latest"]},
    "frontier": {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "fallback_profile": "quality",
        # A hosted profile has no business declaring Ollama fallbacks, but if a
        # future edit adds some they must not be pulled either.
        "fallbacks": ["gemini-3.5-flash"],
    },
}


def _stub_config():
    return mock.patch.object(llm, "config", lambda: {"profiles": _PROFILES})


class ManifestSkipsHostedProfiles(unittest.TestCase):
    def test_hosted_model_is_not_emitted(self):
        with _stub_config():
            self.assertNotIn("gemini-3.6-flash", manifest.models())

    def test_local_models_are_still_emitted(self):
        with _stub_config():
            self.assertEqual(manifest.models(), ["qwen3:4b", "qwen3:8b"])

    def test_hosted_fallback_tags_are_not_emitted_either(self):
        with _stub_config():
            tags = manifest.models(include_fallbacks=True)
        self.assertNotIn("gemini-3.5-flash", tags)
        self.assertEqual(tags, ["qwen3:4b", "qwen2.5:latest", "qwen3:8b", "llama3:8b"])

    def test_deduplicates_across_profiles(self):
        with _stub_config():
            tags = manifest.models(include_fallbacks=True)
        self.assertEqual(len(tags), len(set(tags)))

    def test_runnable_models_also_excludes_hosted(self):
        # can_run_model returns True for an unknown tag ("assume it fits"), so
        # without the provider filter the --runnable-only path the updater uses
        # would let the Gemini name straight through.
        with _stub_config(), mock.patch.object(llm, "gpu_vram_mb", lambda: 8192), \
                mock.patch.object(llm, "system_ram_mb", lambda: 23000), \
                mock.patch.object(llm, "can_run_model", lambda *a, **k: True):
            self.assertNotIn("gemini-3.6-flash", manifest.runnable_models())


class RealConfigHasNoUnpullableTags(unittest.TestCase):
    def test_every_emitted_tag_is_an_ollama_tag(self):
        # Guards the real config/models.yaml, not just the stub above: a new
        # hosted profile added later must not leak into the pull list.
        for tag in manifest.models(include_fallbacks=True):
            profile_providers = {
                llm.get_profile(n).provider for n in llm.config()["profiles"]
                if llm.get_profile(n).model == tag
            }
            self.assertNotIn("gemini", profile_providers, f"{tag} is a hosted model")


if __name__ == "__main__":
    unittest.main()
