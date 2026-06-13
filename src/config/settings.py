"""Pipeline configuration loaded from YAML."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PipelineConfig:
    # LLM mode
    llm_mode: str = "local_first"

    # Cloud model defaults (fallback)
    screenplay_model: str = "gpt-4o"
    character_model: str = "gpt-4o"
    image_gen_model: str = "flux-1.1-pro"
    voice_model: str = "elevenlabs-v2"
    music_model: str = "suno-v4"
    video_renderer: str = "runway-gen3"
    review_model: str = "gpt-4o"

    # Local model assignments per stage
    local_models: dict = field(default_factory=lambda: {
        "screenplay": "llama3:8b-instruct-q5_K_M",
        "character_design": "llama3:8b-instruct-q5_K_M",
        "genre_style": "mistral:7b-instruct-q5_K_M",
        "visual_rendering": "mistral:7b-instruct-q5_K_M",
        "audio_music": "mistral:7b-instruct-q5_K_M",
        "effects_filters": "qwen2:7b-instruct-q5_K_M",
        "assembly": "phi3:mini",
        "review": "llama3:8b-instruct-q5_K_M",
    })

    # General settings
    user_interaction: str = "prompt"
    output_format: str = "vertical_9_16"
    max_iterations: int = 3
    character_rendering_mode: str = "animated"

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 300
    ollama_auto_pull: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        cloud_models = data.get("cloud_models", {})
        local_models = data.get("local_models", {})
        ollama_cfg = data.get("ollama", {})

        return cls(
            llm_mode=data.get("llm_mode", cls.llm_mode),
            screenplay_model=cloud_models.get("screenplay_model", cls.screenplay_model),
            character_model=cloud_models.get("character_model", cls.character_model),
            image_gen_model=cloud_models.get("image_gen_model", cls.image_gen_model),
            voice_model=cloud_models.get("voice_model", cls.voice_model),
            music_model=cloud_models.get("music_model", cls.music_model),
            video_renderer=cloud_models.get("video_renderer", cls.video_renderer),
            review_model=cloud_models.get("review_model", cls.review_model),
            local_models=local_models if local_models else cls.local_models,
            user_interaction=data.get("user_interaction", cls.user_interaction),
            output_format=data.get("output_format", cls.output_format),
            max_iterations=data.get("max_iterations", cls.max_iterations),
            character_rendering_mode=data.get("character_rendering", {}).get(
                "default_mode", cls.character_rendering_mode
            ),
            ollama_host=ollama_cfg.get("host", cls.ollama_host),
            ollama_timeout=ollama_cfg.get("timeout_seconds", cls.ollama_timeout),
            ollama_auto_pull=ollama_cfg.get("auto_pull", cls.ollama_auto_pull),
        )
