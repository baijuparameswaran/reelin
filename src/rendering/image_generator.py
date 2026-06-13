"""
AI image generator using Stable Diffusion (CPU) for scene visualization.

Downloads and caches the model on first use (~2-5GB).
Generates images from text prompts derived from scene descriptions.
Falls back to Pillow-based procedural art if SD is unavailable.
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from PIL import Image


MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE = os.path.expanduser("~/.reel-factory/models/sd")


@dataclass
class ImageGenConfig:
    """Configuration for image generation."""
    width: int = 512
    height: int = 768  # 2:3 aspect ratio (close to 9:16)
    num_inference_steps: int = 15  # Lower = faster on CPU
    guidance_scale: float = 7.5
    model_id: str = MODEL_ID
    cache_dir: str = MODEL_CACHE
    output_dir: str = "/tmp/reel/frames"


class SceneImageGenerator:
    """Generates images for video scenes using Stable Diffusion on CPU."""

    def __init__(self, config: Optional[ImageGenConfig] = None):
        self.config = config or ImageGenConfig()
        self._pipe = None
        os.makedirs(self.config.output_dir, exist_ok=True)

    @property
    def pipe(self):
        """Lazy-load the SD pipeline (downloads model on first use)."""
        if self._pipe is None:
            self._pipe = self._load_pipeline()
        return self._pipe

    def _load_pipeline(self):
        """Load Stable Diffusion pipeline for CPU inference."""
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float32,
            cache_dir=self.config.cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to("cpu")

        # Optimize for CPU
        pipe.enable_attention_slicing()

        return pipe

    def generate_scene_image(
        self,
        prompt: str,
        scene_number: int,
        negative_prompt: str = "blurry, low quality, distorted, watermark, text",
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate an image for a scene from a text prompt.

        Args:
            prompt: Scene description / image generation prompt
            scene_number: Scene number (for filename)
            negative_prompt: What to avoid in generation
            seed: Random seed for reproducibility

        Returns:
            Path to the generated image file
        """
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.config.width,
            height=self.config.height,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        ).images[0]

        output_path = os.path.join(self.config.output_dir, f"scene_{scene_number:03d}.png")
        image.save(output_path)
        return output_path

    def generate_all_scenes(
        self,
        scenes: list[dict],
        style_prefix: str = "",
        genre: str = "cinematic",
        characters: list = None,
        style_guide: dict = None,
    ) -> list[str]:
        """
        Generate images for all scenes with visual continuity.

        Consecutive scenes sharing the same location/characters use a
        consistent base prompt (setting, lighting, color palette) to
        maintain visual continuity. A new shot/location breaks continuity
        and establishes fresh visual context.

        Genre cues from the style_guide (produced by the genre_style agent)
        are incorporated into every image prompt for consistent visual tone.

        Args:
            scenes: List of scene dicts with 'description' or 'image_prompt'
            style_prefix: Style prefix to prepend to all prompts
            genre: Genre for style enhancement
            characters: Character descriptions for consistent appearance
            style_guide: Full style guide from genre_style agent (color palette,
                        lighting, mood, filters, typography, pacing)

        Returns:
            List of paths to generated images
        """
        # Build style string from genre + style_guide
        style = self._build_style_from_guide(genre, style_guide, style_prefix)

        # Build character appearance descriptions for consistency
        char_desc = self._build_character_context(characters or [])

        # Group scenes into continuity sequences
        sequences = self._group_continuity_sequences(scenes)

        image_paths = []
        scene_idx = 0

        for seq in sequences:
            # Establish visual context for this sequence
            context = self._build_continuity_context(seq, style)

            for i, scene in enumerate(seq):
                prompt = scene.get("image_prompt", scene.get("description", f"Scene {scene_idx+1}"))

                # Compose prompt with continuity context
                full_prompt = self._compose_continuity_prompt(
                    prompt, context, char_desc, style, is_first=(i == 0)
                )

                # Use related seeds within a sequence for visual similarity
                base_seed = hash(context.get("location", "")) % 10000
                seed = base_seed + scene_idx

                path = self.generate_scene_image(
                    prompt=full_prompt,
                    scene_number=scene_idx + 1,
                    seed=seed,
                )
                image_paths.append(path)
                scene_idx += 1

        return image_paths

    def _build_style_from_guide(self, genre: str, style_guide: dict = None, style_prefix: str = "") -> str:
        """
        Build comprehensive style prompt from genre and style_guide.

        Extracts visual cues from the style_guide produced by the genre_style
        agent: color palette, lighting direction, mood, filters, pacing feel.
        """
        # Base genre styles as fallback
        style_map = {
            "comedy": "bright colorful cartoon style, fun vibrant, consistent lighting",
            "drama": "cinematic dramatic lighting, moody atmospheric, film grain",
            "horror": "dark eerie horror style, shadows and fog, desaturated",
            "action": "dynamic action scene, dramatic angles, intense, high contrast",
            "romance": "soft warm romantic lighting, dreamy pastel, bokeh",
            "sci-fi": "futuristic sci-fi style, neon lights, cyberpunk, volumetric",
        }

        if style_prefix:
            base_style = style_prefix
        else:
            base_style = style_map.get(genre.lower(), "cinematic high quality")

        if not style_guide or not isinstance(style_guide, dict):
            return base_style

        # Extract rich genre cues from style_guide
        parts = [base_style]

        # Color palette cues
        palette = style_guide.get("color_palette", style_guide.get("palette", {}))
        if isinstance(palette, dict):
            primary = palette.get("primary", "")
            secondary = palette.get("secondary", "")
            accent = palette.get("accent", "")
            if primary:
                parts.append(f"primary color {primary}")
            if secondary:
                parts.append(f"secondary color {secondary}")
            if accent:
                parts.append(f"accent color {accent}")
        elif isinstance(palette, list) and palette:
            parts.append(f"color palette: {', '.join(str(c) for c in palette[:4])}")

        # Lighting direction
        lighting = style_guide.get("lighting", style_guide.get("lighting_style", ""))
        if isinstance(lighting, dict):
            light_type = lighting.get("type", lighting.get("style", ""))
            light_mood = lighting.get("mood", "")
            if light_type:
                parts.append(f"{light_type} lighting")
            if light_mood:
                parts.append(f"{light_mood} mood lighting")
        elif lighting:
            parts.append(f"{lighting} lighting")

        # Mood/atmosphere
        mood = style_guide.get("mood", style_guide.get("atmosphere", ""))
        if mood:
            parts.append(f"{mood} atmosphere")

        # Visual filters/effects
        filters = style_guide.get("filters", style_guide.get("post_processing", ""))
        if isinstance(filters, list):
            parts.extend(str(f) for f in filters[:3])
        elif isinstance(filters, dict):
            filter_names = filters.get("effects", filters.get("look", []))
            if isinstance(filter_names, list):
                parts.extend(str(f) for f in filter_names[:3])
        elif filters:
            parts.append(str(filters))

        # Sub-genre refinement
        sub_genre = style_guide.get("sub_genre", "")
        if sub_genre:
            parts.append(f"{sub_genre} style")

        # Camera/framing guidance
        camera = style_guide.get("camera_style", style_guide.get("framing", ""))
        if camera:
            parts.append(str(camera))

        # Texture/grain
        texture = style_guide.get("texture", style_guide.get("grain", ""))
        if texture:
            parts.append(str(texture))

        return ", ".join(p for p in parts if p)

    def _build_character_context(self, characters: list) -> str:
        """Build consistent character appearance description."""
        if not characters:
            return ""
        descs = []
        for char in characters[:3]:
            if isinstance(char, dict):
                name = char.get("name", "")
                visual = char.get("visual", char.get("appearance", char.get("description", "")))
                if name and visual:
                    descs.append(f"{name}: {visual}")
            elif isinstance(char, str):
                descs.append(char)
        return ", ".join(descs)

    def _group_continuity_sequences(self, scenes: list[dict]) -> list[list[dict]]:
        """
        Group scenes into continuity sequences.

        Scenes in the same location/setting form a sequence with shared
        visual context. A new location or explicit scene break starts a
        new sequence.
        """
        if not scenes:
            return []

        sequences = []
        current_seq = [scenes[0]]
        prev_location = self._extract_location(scenes[0])

        for scene in scenes[1:]:
            curr_location = self._extract_location(scene)
            is_new_shot = self._is_scene_break(scene, prev_location, curr_location)

            if is_new_shot:
                sequences.append(current_seq)
                current_seq = [scene]
            else:
                current_seq.append(scene)

            prev_location = curr_location

        if current_seq:
            sequences.append(current_seq)

        return sequences

    def _extract_location(self, scene: dict) -> str:
        """Extract location/setting from a scene."""
        for key in ("location", "setting", "place", "environment", "backdrop"):
            val = scene.get(key)
            if val:
                return str(val).lower().strip()
        # Infer from description
        desc = scene.get("description", "").lower()
        return desc[:50]

    def _is_scene_break(self, scene: dict, prev_loc: str, curr_loc: str) -> bool:
        """Determine if this scene is a visual break from previous."""
        # Explicit scene break markers
        if scene.get("new_shot") or scene.get("scene_break") or scene.get("transition") == "cut":
            return True

        # Different location
        if prev_loc and curr_loc and prev_loc != curr_loc:
            # Check if locations share keywords (e.g., "rooftop" vs "rooftop at night")
            prev_words = set(prev_loc.split())
            curr_words = set(curr_loc.split())
            overlap = prev_words & curr_words
            if len(overlap) < min(len(prev_words), len(curr_words)) * 0.5:
                return True

        return False

    def _build_continuity_context(self, sequence: list[dict], style: str) -> dict:
        """Build shared visual context for a continuity sequence."""
        first_scene = sequence[0]
        location = self._extract_location(first_scene)
        lighting = first_scene.get("lighting", first_scene.get("time_of_day", ""))
        weather = first_scene.get("weather", "")
        mood = first_scene.get("mood", "")

        return {
            "location": location,
            "lighting": str(lighting),
            "weather": str(weather),
            "mood": str(mood),
            "style": style,
            "scene_count": len(sequence),
        }

    def _compose_continuity_prompt(
        self, scene_prompt: str, context: dict, char_desc: str, style: str, is_first: bool
    ) -> str:
        """
        Compose a prompt that maintains continuity with the sequence context.

        First scene in sequence establishes the setting.
        Subsequent scenes reference the same environment.
        """
        parts = [style]

        # Add continuity context (same setting/lighting for all scenes in sequence)
        if context.get("location"):
            parts.append(f"setting: {context['location']}")
        if context.get("lighting"):
            parts.append(f"lighting: {context['lighting']}")
        if context.get("weather"):
            parts.append(f"weather: {context['weather']}")
        if context.get("mood"):
            parts.append(f"mood: {context['mood']}")

        # Add character descriptions for consistency
        if char_desc:
            parts.append(f"characters: {char_desc}")

        # Continuity instruction for non-first scenes
        if not is_first and context.get("scene_count", 0) > 1:
            parts.append("same environment and color palette as previous frame")

        # The specific scene action/description
        parts.append(scene_prompt)

        return ", ".join(parts)

    def is_available(self) -> bool:
        """Check if Stable Diffusion can be loaded."""
        try:
            import torch
            import diffusers
            return True
        except ImportError:
            return False
