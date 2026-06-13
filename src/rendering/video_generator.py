"""
Video generator that produces actual MP4 output from pipeline results.

Uses Pillow for frame generation and moviepy for video assembly.
Generates a storyboard-style animated video with:
- Colored scene backgrounds based on genre/style palette
- Text overlays with scene descriptions, dialogue
- Simple transitions (fade, dissolve)
- 9:16 aspect ratio (1080x1920) for short-form content
"""
import os
from typing import Optional
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont, ImageFilter


@dataclass
class VideoConfig:
    """Configuration for video output."""
    width: int = 1080
    height: int = 1920
    fps: int = 24
    output_dir: str = "/tmp/reel/output"
    font_size_title: int = 72
    font_size_body: int = 48
    font_size_dialogue: int = 40
    default_scene_duration: float = 5.0
    transition_duration: float = 0.5


GENRE_PALETTES = {
    "comedy": {"bg": (255, 245, 200), "text": (60, 60, 60), "accent": (255, 165, 0)},
    "drama": {"bg": (40, 40, 60), "text": (220, 220, 220), "accent": (180, 100, 100)},
    "horror": {"bg": (20, 10, 10), "text": (200, 0, 0), "accent": (100, 0, 0)},
    "action": {"bg": (30, 30, 50), "text": (255, 255, 255), "accent": (255, 50, 50)},
    "romance": {"bg": (255, 220, 230), "text": (80, 40, 60), "accent": (220, 80, 120)},
    "sci-fi": {"bg": (10, 20, 40), "text": (0, 220, 255), "accent": (100, 0, 200)},
    "default": {"bg": (30, 30, 30), "text": (240, 240, 240), "accent": (100, 180, 255)},
}


class VideoGenerator:
    """Generates MP4 video from pipeline stage outputs."""

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def generate(
        self,
        screenplay: dict,
        style_guide: dict,
        characters: list,
        genre: str = "default",
        output_path: Optional[str] = None,
        use_ai_images: bool = True,
    ) -> str:
        """
        Generate a video from pipeline outputs.

        Args:
            use_ai_images: If True, use Stable Diffusion for scene images.
                          Falls back to text-based frames if SD unavailable.

        Returns the path to the generated MP4 file.
        """
        from moviepy import (
            ImageClip,
            concatenate_videoclips,
        )

        palette = self._get_palette(genre, style_guide)
        scenes = screenplay.get("scenes", [])
        if not scenes:
            scenes = [{"description": "Generated reel", "duration": 5}]

        # Try AI image generation
        scene_images = []
        if use_ai_images:
            scene_images = self._generate_ai_images(scenes, genre, characters, style_guide)

        clips = []

        # Title card
        title_clip = self._make_title_card(screenplay, genre, palette)
        clips.append(title_clip)

        # Scene clips (with AI images if available)
        for i, scene in enumerate(scenes):
            if i < len(scene_images) and scene_images[i]:
                scene_clip = self._make_image_scene_clip(
                    scene_images[i], scene, i + 1, palette
                )
            else:
                scene_clip = self._make_scene_clip(scene, i + 1, palette, characters)
            clips.append(scene_clip)

        # End card
        end_clip = self._make_end_card(palette)
        clips.append(end_clip)

        # Concatenate all clips
        final = concatenate_videoclips(clips, method="compose")

        # Write output
        if not output_path:
            output_path = os.path.join(self.config.output_dir, "reel_output.mp4")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final.write_videofile(
            output_path,
            fps=self.config.fps,
            codec="libx264",
            audio=False,
            logger=None,
        )

        # Cleanup
        final.close()
        for clip in clips:
            clip.close()

        return output_path

    def _generate_ai_images(self, scenes: list, genre: str, characters: list = None, style_guide: dict = None) -> list:
        """Generate AI images for scenes using Stable Diffusion."""
        try:
            from src.rendering.image_generator import SceneImageGenerator
            gen = SceneImageGenerator()
            if not gen.is_available():
                return []
            return gen.generate_all_scenes(scenes, genre=genre, characters=characters or [], style_guide=style_guide or {})
        except Exception:
            return []

    def _make_image_scene_clip(self, image_path: str, scene: dict, number: int, palette: dict):
        """Create a clip from an AI-generated image with Ken Burns effect."""
        from moviepy import ImageClip
        import numpy as np

        duration = scene.get("duration", scene.get("duration_seconds", self.config.default_scene_duration))
        if isinstance(duration, str):
            try:
                duration = float(duration.replace("s", ""))
            except (ValueError, TypeError):
                duration = self.config.default_scene_duration
        duration = min(float(duration), 15.0)

        # Load and resize to 9:16
        img = Image.open(image_path)
        img = img.resize((self.config.width, self.config.height), Image.Resampling.LANCZOS)

        # Add subtle text overlay with scene description
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except (OSError, IOError):
            font = ImageFont.load_default()

        desc = scene.get("description", scene.get("action", ""))
        if desc:
            # Semi-transparent text bar at bottom
            bar_height = 120
            overlay = Image.new("RGBA", (self.config.width, bar_height), (0, 0, 0, 160))
            img = img.convert("RGBA")
            img.paste(overlay, (0, self.config.height - bar_height), overlay)
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            wrapped = self._wrap_text(str(desc)[:120], font, self.config.width - 40)
            y = self.config.height - bar_height + 15
            for line in wrapped[:2]:
                bbox = draw.textbbox((0, 0), line, font=font)
                x = (self.config.width - (bbox[2] - bbox[0])) // 2
                draw.text((x, y), line, fill=(255, 255, 255), font=font)
                y += 46

        frame = np.array(img)
        return ImageClip(frame, duration=duration)

    def _get_palette(self, genre: str, style_guide: dict) -> dict:
        """Get color palette from genre or style guide."""
        palette = GENRE_PALETTES.get(genre.lower(), GENRE_PALETTES["default"])
        if isinstance(style_guide, dict):
            sg_palette = style_guide.get("palette") or style_guide.get("color_palette")
            if isinstance(sg_palette, dict):
                if "background" in sg_palette:
                    palette = {**palette, "bg": self._parse_color(sg_palette["background"])}
                if "text" in sg_palette:
                    palette = {**palette, "text": self._parse_color(sg_palette["text"])}
        return palette

    def _parse_color(self, color) -> tuple:
        """Parse a color value to RGB tuple."""
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            return tuple(int(c) for c in color[:3])
        if isinstance(color, str):
            color = color.lstrip("#")
            if len(color) == 6:
                return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        return (128, 128, 128)

    def _make_frame(self, text_lines: list, palette: dict, duration: float):
        """Create a video clip from text lines on a colored background."""
        from moviepy import ImageClip
        import numpy as np

        img = Image.new("RGB", (self.config.width, self.config.height), palette["bg"])
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.config.font_size_body)
        except (OSError, IOError):
            font = ImageFont.load_default()

        y = self.config.height // 4
        for line in text_lines:
            wrapped = self._wrap_text(str(line), font, self.config.width - 120)
            for wline in wrapped:
                bbox = draw.textbbox((0, 0), wline, font=font)
                text_width = bbox[2] - bbox[0]
                x = (self.config.width - text_width) // 2
                draw.text((x, y), wline, fill=palette["text"], font=font)
                y += self.config.font_size_body + 10
            y += 20

        frame = np.array(img)
        clip = ImageClip(frame, duration=duration)
        return clip

    def _make_title_card(self, screenplay: dict, genre: str, palette: dict):
        """Create title card clip."""
        title = screenplay.get("title", "Untitled Reel")
        lines = [
            str(title).upper(),
            "",
            f"Genre: {genre.title()}",
            "",
            f"{len(screenplay.get('scenes', []))} scenes",
        ]
        return self._make_frame(lines, palette, 3.0)

    def _make_scene_clip(self, scene: dict, number: int, palette: dict, characters: list):
        """Create a clip for a single scene."""
        duration = scene.get("duration", scene.get("duration_seconds", self.config.default_scene_duration))
        if isinstance(duration, str):
            try:
                duration = float(duration.replace("s", ""))
            except (ValueError, TypeError):
                duration = self.config.default_scene_duration
        duration = min(float(duration), 15.0)

        description = scene.get("description", scene.get("action", scene.get("setting", f"Scene {number}")))
        dialogue = scene.get("dialogue", scene.get("lines", []))

        lines = [f"--- Scene {number} ---", "", str(description)]

        if isinstance(dialogue, list):
            lines.append("")
            for d in dialogue[:3]:
                if isinstance(d, dict):
                    char = d.get("character", "???")
                    text = d.get("line", d.get("text", ""))
                    lines.append(f'{char}: "{text}"')
                elif isinstance(d, str):
                    lines.append(f'"{d}"')

        scene_chars = scene.get("characters", scene.get("characters_in_frame", []))
        if scene_chars and isinstance(scene_chars, list):
            lines.append("")
            lines.append(f"Characters: {', '.join(str(c) for c in scene_chars[:3])}")

        return self._make_frame(lines, palette, duration)

    def _make_end_card(self, palette: dict):
        """Create end card."""
        lines = ["", "", "--- FIN ---", "", "Generated by Reel Factory"]
        return self._make_frame(lines, palette, 2.0)

    def _wrap_text(self, text: str, font, max_width: int) -> list:
        """Wrap text to fit within max_width pixels."""
        if not text:
            return [""]
        words = text.split()
        lines = []
        current = ""
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]
