"""Validates multi-character Veo `reference_images` support: at a "shot
boundary" (a scene's first panel, or any panel whose in-frame character SET
changes from the one before it) with 2+ in-frame characters, every one of
them gets an identity-anchoring reference image, not just whichever
character `pipeline._frame_char_anchor` happened to pick first. See
PROGRESS.md's session log and project memory `veo-character-consistency`
for the background this closes.

Pure logic + mocked-SDK checks — no live Ollama/Gemini/Veo calls anywhere.
The SDK-shape test below patches `google.genai.Client` (network layer only)
rather than mocking `google.genai.types` itself, so it still exercises real
Pydantic validation of the request shape, same as the project's established
practice for verifying SDK call shapes offline.

Run with: python -m unittest tests.test_multi_character_references -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reel import gemini, i2v, pipeline


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-png-bytes")
    return p


class TestResolvePanelReferences(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.cast_index = {
            "Alice": "casting/alice.png",
            "Bob": "casting/bob.png",
            "Carol": "casting/carol.png",
            "Dave": "casting/dave.png",
            "Eve": "casting/eve.png",     # never rendered to disk
        }
        for name in ("Alice", "Bob", "Carol", "Dave"):
            _touch(self.out / self.cast_index[name])

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_panel_of_scene_with_two_characters_returns_both(self):
        fr = {"characters_in_frame": ["Alice", "Bob"]}
        refs, key = pipeline._resolve_panel_references(fr, None, self.cast_index, self.out)
        self.assertEqual(len(refs), 2)
        self.assertEqual(key, frozenset({"Alice", "Bob"}))

    def test_first_panel_with_a_single_character_returns_no_references(self):
        # Not "multi" — the caller keeps using the proven single-seed path.
        fr = {"characters_in_frame": ["Alice"]}
        refs, key = pipeline._resolve_panel_references(fr, None, self.cast_index, self.out)
        self.assertEqual(refs, [])
        self.assertEqual(key, frozenset({"Alice"}))

    def test_non_boundary_panel_same_cast_as_previous_returns_no_references(self):
        prev_key = frozenset({"Alice", "Bob"})
        fr = {"characters_in_frame": ["Alice", "Bob"]}
        refs, key = pipeline._resolve_panel_references(fr, prev_key, self.cast_index, self.out)
        self.assertEqual(refs, [])
        self.assertEqual(key, prev_key)

    def test_cast_change_mid_scene_is_a_boundary(self):
        prev_key = frozenset({"Alice"})
        fr = {"characters_in_frame": ["Alice", "Bob"]}   # Bob joins -> boundary
        refs, key = pipeline._resolve_panel_references(fr, prev_key, self.cast_index, self.out)
        self.assertEqual(len(refs), 2)
        self.assertEqual(key, frozenset({"Alice", "Bob"}))

    def test_fewer_than_two_resolvable_portraits_returns_no_references(self):
        # Eve has no rendered casting image on disk.
        fr = {"characters_in_frame": ["Alice", "Eve"]}
        refs, key = pipeline._resolve_panel_references(fr, None, self.cast_index, self.out)
        self.assertEqual(refs, [])
        self.assertEqual(key, frozenset({"Alice", "Eve"}))

    def test_no_characters_in_frame_keeps_callers_prev_key(self):
        prev_key = frozenset({"Alice", "Bob"})
        fr = {"characters_in_frame": []}
        refs, key = pipeline._resolve_panel_references(fr, prev_key, self.cast_index, self.out)
        self.assertEqual(refs, [])
        self.assertEqual(key, prev_key)

    def test_capped_at_three_even_with_four_characters(self):
        fr = {"characters_in_frame": ["Alice", "Bob", "Carol", "Dave"]}
        refs, key = pipeline._resolve_panel_references(fr, None, self.cast_index, self.out)
        self.assertEqual(len(refs), 3)


class TestRenderOnePanelThreadsReferenceImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.sdir = self.out / "video" / "scene_01"
        self.sdir.mkdir(parents=True)
        self.ref_a = _touch(self.out / "casting" / "alice.png")
        self.ref_b = _touch(self.out / "casting" / "bob.png")

    def tearDown(self):
        self.tmp.cleanup()

    def _panel(self):
        return {"panel": 1, "characters_in_frame": ["Alice", "Bob"],
               "action": "they talk", "shot_type": "WS"}

    def test_reference_images_passed_through_to_generate_clip(self):
        captured = {}

        def fake_generate_clip(images, prompt, clip, **kwargs):
            captured.update(kwargs)
            clip.write_bytes(b"fake-mp4")
            return True

        with mock.patch.object(i2v, "generate_clip", side_effect=fake_generate_clip), \
             mock.patch.object(i2v, "overlays_enabled", return_value=False), \
             mock.patch.object(i2v, "last_frame", return_value=None):
            res = pipeline._render_one_panel(
                self._panel(), 1, self.sdir, None, None, self.out,
                casting_lookup={}, location_desc="", visual_overview={},
                voice_index={}, audio_overview={},
                no_bg_music=True, room_tone=True, no_subtitles=True,
                reference_images=[self.ref_a, self.ref_b],
            )

        self.assertTrue(res["rendered"])
        self.assertEqual(captured.get("reference_images"), [self.ref_a, self.ref_b])
        self.assertEqual(
            res["frame_record"]["reference_images"],
            [str(self.ref_a.relative_to(self.out)), str(self.ref_b.relative_to(self.out))],
        )

    def test_revising_a_referenced_casting_image_invalidates_the_panel_hash(self):
        def fake_generate_clip(images, prompt, clip, **kwargs):
            clip.write_bytes(b"fake-mp4")
            return True

        with mock.patch.object(i2v, "generate_clip", side_effect=fake_generate_clip), \
             mock.patch.object(i2v, "overlays_enabled", return_value=False), \
             mock.patch.object(i2v, "last_frame", return_value=None):
            pipeline._render_one_panel(
                self._panel(), 1, self.sdir, None, None, self.out,
                casting_lookup={}, location_desc="", visual_overview={},
                voice_index={}, audio_overview={},
                no_bg_music=True, room_tone=True, no_subtitles=True,
                reference_images=[self.ref_a, self.ref_b],
            )
            hash_path = self.sdir / "frame_01.hash"
            first_hash = hash_path.read_text()

            # Simulate a casting revision: Alice's portrait bytes change.
            self.ref_a.write_bytes(b"revised-portrait-bytes")

            res2 = pipeline._render_one_panel(
                self._panel(), 1, self.sdir, None, None, self.out,
                casting_lookup={}, location_desc="", visual_overview={},
                voice_index={}, audio_overview={},
                no_bg_music=True, room_tone=True, no_subtitles=True,
                reference_images=[self.ref_a, self.ref_b],
            )
        self.assertTrue(res2["rendered"], "changed reference image bytes must re-trigger a render")
        self.assertNotEqual(hash_path.read_text(), first_hash)


class TestGenGeminiReferenceBranch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_path = Path(self.tmp.name) / "clip.mp4"
        self.ref_a = _touch(Path(self.tmp.name) / "a.png")
        self.ref_b = _touch(Path(self.tmp.name) / "b.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_or_more_references_uses_the_reference_call(self):
        with mock.patch.object(i2v, "_cfg", return_value={"multi_character_references": True}), \
             mock.patch.object(gemini, "generate_video_with_references",
                              return_value=True) as m_refs, \
             mock.patch.object(gemini, "generate_video") as m_seed:
            ok = i2v._gen_gemini([], "a prompt", self.out_path,
                                 reference_images=[self.ref_a, self.ref_b])
        self.assertTrue(ok)
        m_refs.assert_called_once()
        m_seed.assert_not_called()

    def test_disabled_by_config_falls_back_to_seed_path(self):
        with mock.patch.object(i2v, "_cfg", return_value={"multi_character_references": False}), \
             mock.patch.object(gemini, "generate_video_with_references") as m_refs, \
             mock.patch.object(gemini, "generate_video", return_value=True) as m_seed:
            ok = i2v._gen_gemini([self.ref_a], "a prompt", self.out_path,
                                 reference_images=[self.ref_a, self.ref_b])
        self.assertTrue(ok)
        m_refs.assert_not_called()
        m_seed.assert_called_once()

    def test_a_single_reference_image_is_not_enough_falls_back_to_seed_path(self):
        with mock.patch.object(i2v, "_cfg", return_value={"multi_character_references": True}), \
             mock.patch.object(gemini, "generate_video_with_references") as m_refs, \
             mock.patch.object(gemini, "generate_video", return_value=True) as m_seed:
            ok = i2v._gen_gemini([self.ref_a], "a prompt", self.out_path,
                                 reference_images=[self.ref_a])
        self.assertTrue(ok)
        m_refs.assert_not_called()
        m_seed.assert_called_once()

    def test_reference_call_failure_falls_back_to_seed_path(self):
        with mock.patch.object(i2v, "_cfg", return_value={"multi_character_references": True}), \
             mock.patch.object(gemini, "generate_video_with_references",
                              side_effect=RuntimeError("boom")), \
             mock.patch.object(gemini, "generate_video", return_value=True) as m_seed:
            ok = i2v._gen_gemini([self.ref_a], "a prompt", self.out_path,
                                 reference_images=[self.ref_a, self.ref_b])
        self.assertTrue(ok)
        m_seed.assert_called_once()


class TestGenerateVideoWithReferencesSdkShape(unittest.TestCase):
    """Exercises the REAL google-genai Pydantic types (installed in this
    project's .venv) to catch a request-shape bug the way earlier sessions'
    SDK fixes were verified — only the network layer (`Client`) is mocked."""

    def test_builds_a_valid_config_and_extracts_the_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_a = _touch(Path(tmpdir) / "a.png")
            ref_b = _touch(Path(tmpdir) / "b.png")
            out_path = Path(tmpdir) / "clip.mp4"

            fake_operation = mock.MagicMock()
            fake_operation.done = True
            fake_operation.error = None
            fake_video = mock.MagicMock()
            fake_operation.result.generated_videos = [mock.MagicMock(video=fake_video)]

            fake_client = mock.MagicMock()
            fake_client.models.generate_videos.return_value = fake_operation
            fake_client.files.download.return_value = b"fake-video-bytes"

            with mock.patch("google.genai.Client", return_value=fake_client):
                ok = gemini._generate_video_with_references_once(
                    "a prompt", out_path, [ref_a, ref_b],
                    model="veo-3.1-generate-preview", aspect_ratio="16:9",
                    resolution="720p", poll_seconds=0.01, timeout_seconds=5,
                )

            self.assertTrue(ok)
            self.assertEqual(out_path.read_bytes(), b"fake-video-bytes")
            call_kwargs = fake_client.models.generate_videos.call_args.kwargs
            config = call_kwargs["config"]
            self.assertEqual(config.duration_seconds, 8)
            self.assertEqual(config.person_generation, "allow_adult")
            self.assertEqual(len(config.reference_images), 2)
            from google.genai import types
            for ref in config.reference_images:
                self.assertEqual(ref.reference_type, types.VideoGenerationReferenceType.ASSET)

    def test_caps_at_three_reference_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            refs = [_touch(Path(tmpdir) / f"{i}.png") for i in range(5)]
            out_path = Path(tmpdir) / "clip.mp4"

            fake_operation = mock.MagicMock()
            fake_operation.done = True
            fake_operation.error = None
            fake_operation.result.generated_videos = [mock.MagicMock(video=mock.MagicMock())]
            fake_client = mock.MagicMock()
            fake_client.models.generate_videos.return_value = fake_operation
            fake_client.files.download.return_value = b"x"

            with mock.patch("google.genai.Client", return_value=fake_client):
                gemini._generate_video_with_references_once(
                    "a prompt", out_path, refs,
                    model="veo-3.1-generate-preview", aspect_ratio="16:9",
                    resolution="720p", poll_seconds=0.01, timeout_seconds=5,
                )
            config = fake_client.models.generate_videos.call_args.kwargs["config"]
            self.assertEqual(len(config.reference_images), 3)


class TestCharSetChanged(unittest.TestCase):
    def test_scene_start_is_a_boundary(self):
        self.assertTrue(pipeline._char_set_changed({"characters_in_frame": ["Alice"]}, None))

    def test_same_cast_is_not_a_boundary(self):
        key = frozenset({"Alice", "Bob"})
        fr = {"characters_in_frame": ["Bob", "Alice"]}   # order-independent
        self.assertFalse(pipeline._char_set_changed(fr, key))

    def test_different_cast_is_a_boundary(self):
        key = frozenset({"Alice"})
        fr = {"characters_in_frame": ["Alice", "Bob"]}
        self.assertTrue(pipeline._char_set_changed(fr, key))

    def test_no_characters_listed_is_never_a_boundary(self):
        key = frozenset({"Alice"})
        self.assertFalse(pipeline._char_set_changed({"characters_in_frame": []}, key))
        self.assertFalse(pipeline._char_set_changed({}, key))


class TestRenderSceneFramesOnlyExtendsWhenCastUnchanged(unittest.TestCase):
    """End-to-end (mocked i2v): `continuity_mode: extend` must only be
    attempted (via a non-None `prev_clip` reaching `i2v.generate_clip`) when
    a panel's in-frame characters match the immediately preceding panel's —
    a cast change should null it out, falling back to seed/reference-image
    grounding instead, even though a previous clip genuinely exists."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        _touch(self.out / "casting" / "alice.png")
        _touch(self.out / "casting" / "bob.png")
        self.storyboard = {"storyboard": [{
            "scene_number": 1, "header": {"location": ""}, "audio_overview": {},
            "visual_overview": {},
            "panels": [
                {"panel": 1, "characters_in_frame": ["Alice"], "action": "a"},
                {"panel": 2, "characters_in_frame": ["Alice"], "action": "b"},
                {"panel": 3, "characters_in_frame": ["Alice", "Bob"], "action": "c"},
            ],
        }]}
        self.casting = {"casting": [
            {"name": "Alice", "character": {"image_path": "casting/alice.png"}},
            {"name": "Bob", "character": {"image_path": "casting/bob.png"}},
        ]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_unchanged_cast_extends_changed_cast_does_not(self):
        prev_clips = {}

        def fake_generate_clip(images, prompt, clip, *, prev_clip=None,
                               duration_seconds=None, reference_images=None,
                               dry_run=False):
            prev_clips[clip.name] = prev_clip
            clip.write_bytes(b"clip")
            return True

        with mock.patch.object(i2v, "enabled", return_value=True), \
             mock.patch.object(i2v, "available", return_value=True), \
             mock.patch.object(i2v, "_cfg", return_value={"continuity_mode": "seed",
                                                          "multi_character_references": True}), \
             mock.patch.object(i2v, "generate_clip", side_effect=fake_generate_clip), \
             mock.patch.object(i2v, "overlays_enabled", return_value=False), \
             mock.patch.object(i2v, "last_frame", return_value=None), \
             mock.patch.object(i2v, "stitch", return_value=True):
            pipeline._render_scene_frames(self.storyboard, self.casting, self.out)

        # panel 2: same cast as panel 1 -> eligible to extend from panel 1's clip
        self.assertIsNotNone(prev_clips.get("frame_02.mp4"))
        # panel 3: Bob joins -> a character boundary -> must NOT extend,
        # even though panel 2's clip genuinely exists on disk by then
        self.assertIsNone(prev_clips.get("frame_03.mp4"))


class TestRerenderPanelsUsesReferencesAtABoundary(unittest.TestCase):
    """End-to-end (mocked i2v): `rerender_panels`' targeted re-render must
    compute the SAME boundary/reference decision `_render_scene_frames`'
    normal walk would have for that panel — otherwise its `_content_hash`
    would disagree with a later full render pass (the exact class of bug
    `_resolve_prev_clip_path` was already written to avoid for
    `prev_clip_path`; this test is the equivalent guard for
    `reference_images`)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.sdir = self.out / "video" / "scene_01"
        self.sdir.mkdir(parents=True)
        self.alice = _touch(self.out / "casting" / "alice.png")
        self.bob = _touch(self.out / "casting" / "bob.png")

        # 3-panel scene: panel 1 = Alice only; panel 2 = Alice+Bob (Bob JOINS
        # -> boundary); panel 3 = Alice+Bob (same cast as panel 2 -> not a
        # boundary). Pre-existing manifest/clip files simulate an already-
        # rendered scene, since rerender_panels requires one.
        for n in (1, 2, 3):
            clip = self.sdir / f"frame_{n:02d}.mp4"
            tail = self.sdir / f"frame_{n:02d}_tail.png"
            clip.write_bytes(b"old-clip")
            tail.write_bytes(b"old-tail")
        (self.out / "video" / "manifest.json").write_text(__import__("json").dumps({
            "clips": 3, "failed": 0,
            "scenes": [{"scene_number": 1, "scene_video": "video/scene_01.mp4", "frames": [
                {"panel": 1, "clip": "video/scene_01/frame_01.mp4",
                 "end_frame": "video/scene_01/frame_01_tail.png"},
                {"panel": 2, "clip": "video/scene_01/frame_02.mp4",
                 "end_frame": "video/scene_01/frame_02_tail.png"},
                {"panel": 3, "clip": "video/scene_01/frame_03.mp4",
                 "end_frame": "video/scene_01/frame_03_tail.png"},
            ]}],
        }))

        self.storyboard = {"storyboard": [{
            "scene_number": 1, "header": {"location": ""}, "audio_overview": {},
            "visual_overview": {},
            "panels": [
                {"panel": 1, "characters_in_frame": ["Alice"], "action": "a"},
                {"panel": 2, "characters_in_frame": ["Alice", "Bob"], "action": "b"},
                {"panel": 3, "characters_in_frame": ["Alice", "Bob"], "action": "c"},
            ],
        }]}
        self.casting = {"casting": [
            {"name": "Alice", "character": {"image_path": "casting/alice.png"}},
            {"name": "Bob", "character": {"image_path": "casting/bob.png"}},
        ]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_targeted_boundary_panel_gets_references_cascade_panel_does_not(self):
        calls = {}

        def fake_generate_clip(images, prompt, clip, *, prev_clip=None,
                               duration_seconds=None, reference_images=None,
                               dry_run=False):
            calls[clip.name] = reference_images
            clip.write_bytes(b"new-clip")
            return True

        with mock.patch.object(i2v, "generate_clip", side_effect=fake_generate_clip), \
             mock.patch.object(i2v, "overlays_enabled", return_value=False), \
             mock.patch.object(i2v, "last_frame", return_value=None), \
             mock.patch.object(i2v, "stitch", return_value=True):
            pipeline.rerender_panels(self.storyboard, self.casting, self.out,
                                     scene_number=1, panel_numbers=[2])

        # panel 2 (targeted): Bob joins relative to panel 1 -> boundary -> refs
        self.assertEqual(calls.get("frame_02.mp4"), [self.alice, self.bob])
        # panel 3 (one-hop cascade): same cast as panel 2 -> not a boundary -> no refs
        self.assertIsNone(calls.get("frame_03.mp4"))


if __name__ == "__main__":
    unittest.main()
