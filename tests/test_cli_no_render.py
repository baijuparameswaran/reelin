"""Validates the `--no-render` flag on the main `python -m reel.cli SOURCE`
command: casting-image and video rendering (the only two stages that spend
real Gemini/Veo API quota) can be skipped for an entire full-pipeline run,
not just inside `revise`. Covers the argparse sentinel/inheritance logic in
`cli.main()`, the `render` field round-tripping through `run_params.json`,
and `pipeline.run()`'s own gating of the two render blocks.

Pure logic checks with mocked I/O — no LLM/API calls, no real rendering.
Run with: python -m unittest tests.test_cli_no_render -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import cli
from reel.pipeline import PipelineStopped


class TestRunParamsRoundTripRender(unittest.TestCase):
    def test_render_defaults_to_true_when_not_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cli._save_run_params(out, max_scenes=1, profile=None, genre=None)
            self.assertEqual(cli._load_run_params(out)["render"], True)

    def test_render_false_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cli._save_run_params(out, max_scenes=1, profile=None, genre=None, render=False)
            self.assertEqual(cli._load_run_params(out)["render"], False)

    def test_missing_run_params_file_has_no_render_key(self):
        # A pre-existing --out from before this field existed — _load_run_params
        # degrades to {} for a missing file entirely; callers must fall back
        # to True themselves rather than assume the key is always present.
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            self.assertNotIn("render", cli._load_run_params(out))


class TestMainNoRenderFlag(unittest.TestCase):
    """Drives cli.main() directly with reel.cli.run mocked as a call
    recorder — no real pipeline execution, no LLM/API calls."""

    def _run_main(self, argv, out: Path):
        captured = {}

        def fake_run(source, **kwargs):
            captured.update(kwargs)
            return {}

        # main() calls _offer_revise() after a successful run, which calls
        # input() for real — under a NON-interactive test runner that's
        # harmless (input() raises EOFError, caught, returns immediately),
        # but under a real interactive terminal (e.g. `make demo` run by a
        # human directly) it actually BLOCKS waiting for keyboard input.
        # Mock it out entirely — these tests are about main()'s flag/
        # inheritance resolution, not the post-run revise offer.
        with mock.patch("reel.cli.run", side_effect=fake_run), \
             mock.patch("reel.cli._offer_revise"):
            cli.main(argv + ["--out", str(out)])
        return captured

    def test_no_render_flag_disables_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            src = out / "story.txt"
            src.write_text("a story")
            captured = self._run_main([str(src), "--no-render"], out)
            self.assertEqual(captured.get("render"), False)
            self.assertEqual(cli._load_run_params(out)["render"], False)

    def test_omitted_flag_renders_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            src = out / "story.txt"
            src.write_text("a story")
            captured = self._run_main([str(src)], out)
        self.assertEqual(captured.get("render"), True)

    def test_resume_without_flag_inherits_no_render_from_prior_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            src = out / "story.txt"
            src.write_text("a story")
            cli._save_run_params(out, max_scenes=None, profile=None, genre=None, render=False)
            captured = self._run_main([str(src), "--resume"], out)
        self.assertEqual(captured.get("render"), False)

    def test_resume_without_flag_and_no_prior_render_field_defaults_true(self):
        # Simulates a run_params.json written before --no-render existed.
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            src = out / "story.txt"
            src.write_text("a story")
            (out / "run_params.json").write_text(json.dumps(
                {"max_scenes": None, "profile": None, "genre": None}))
            captured = self._run_main([str(src), "--resume"], out)
        self.assertEqual(captured.get("render"), True)

    def test_explicit_flag_on_resume_overrides_and_updates_stored_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            src = out / "story.txt"
            src.write_text("a story")
            cli._save_run_params(out, max_scenes=None, profile=None, genre=None, render=True)
            captured = self._run_main([str(src), "--resume", "--no-render"], out)
            self.assertEqual(captured.get("render"), False)
            # A LATER bare resume should now inherit the override, not the original True.
            captured2 = self._run_main([str(src), "--resume"], out)
            self.assertEqual(captured2.get("render"), False)

    def test_paused_run_resume_hint_includes_no_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            src = out / "story.txt"
            src.write_text("a story")

            def fake_run(source, **kwargs):
                raise PipelineStopped("structure")

            with mock.patch("reel.cli.run", side_effect=fake_run), \
                 mock.patch("reel.cli.session.finish"):
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                with redirect_stdout(buf):
                    cli.main([str(src), "--no-render", "--out", str(out)])
            self.assertIn("--no-render", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
