"""
Default configurations and boundary contracts for each pipeline stage.

Each stage has well-defined:
- Input contract (what it expects from state)
- Output contract (what it produces)
- Default parameters (used when user skips/times out)
- Timeout for user intervention
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageDefaults:
    """Default behavior when user does not intervene (timeout or auto mode)."""
    intervention_timeout_seconds: float = 30.0
    auto_approve_on_timeout: bool = True
    min_confidence_for_auto: float = 0.6
    max_retries: int = 2
    retry_with_feedback: bool = True
    fallback_models: list[str] = field(default_factory=list)
    max_tokens: int = 2048
    llm_timeout_seconds: float = 600.0


@dataclass
class StageBoundary:
    """
    Defines the contract boundary for a pipeline stage.
    Agents MUST only read required_inputs and write produced_outputs.
    """
    stage_name: str
    required_inputs: list[str]
    produced_outputs: list[str]
    optional_inputs: list[str] = field(default_factory=list)
    description: str = ""


STAGE_BOUNDARIES: dict[str, StageBoundary] = {
    "screenplay": StageBoundary(
        stage_name="screenplay",
        required_inputs=["script"],
        produced_outputs=["screenplay"],
        optional_inputs=["genre", "last_feedback"],
        description="Converts raw script text into structured screenplay with scenes and characters.",
    ),
    "character_design": StageBoundary(
        stage_name="character_design",
        required_inputs=["screenplay"],
        produced_outputs=["characters"],
        optional_inputs=["genre", "style_guide"],
        description="Creates visual character descriptions and reference images.",
    ),
    "genre_style": StageBoundary(
        stage_name="genre_style",
        required_inputs=["screenplay"],
        produced_outputs=["style_guide"],
        optional_inputs=["genre", "characters"],
        description="Establishes visual style, color palette, pacing, and audio mood.",
    ),
    "visual_rendering": StageBoundary(
        stage_name="visual_rendering",
        required_inputs=["screenplay", "characters", "style_guide"],
        produced_outputs=["visual_clips"],
        optional_inputs=["last_feedback"],
        description="Generates video frames or clips for each scene.",
    ),
    "audio_music": StageBoundary(
        stage_name="audio_music",
        required_inputs=["screenplay", "style_guide"],
        produced_outputs=["audio_tracks"],
        optional_inputs=["characters", "visual_clips"],
        description="Produces background music, voiceover, and sound effects.",
    ),
    "effects_filters": StageBoundary(
        stage_name="effects_filters",
        required_inputs=["visual_clips", "style_guide"],
        produced_outputs=["processed_clips"],
        optional_inputs=["audio_tracks"],
        description="Applies color grading, filters, transitions, and overlays.",
    ),
    "assembly": StageBoundary(
        stage_name="assembly",
        required_inputs=["processed_clips", "audio_tracks", "screenplay"],
        produced_outputs=["final_output"],
        optional_inputs=["style_guide"],
        description="Assembles all clips and audio into the final 9:16 MP4.",
    ),
    "review": StageBoundary(
        stage_name="review",
        required_inputs=["final_output", "screenplay", "style_guide"],
        produced_outputs=["review_score"],
        optional_inputs=["genre", "characters"],
        description="Evaluates the final reel for quality and engagement.",
    ),
}


STAGE_DEFAULTS: dict[str, StageDefaults] = {
    "screenplay": StageDefaults(
        intervention_timeout_seconds=45.0,
        auto_approve_on_timeout=True,
        min_confidence_for_auto=0.7,
        max_retries=3,
        fallback_models=["gpt-4o-mini", "claude-haiku-3-20240307"],
        max_tokens=3072,
        llm_timeout_seconds=600.0,
    ),
    "character_design": StageDefaults(
        intervention_timeout_seconds=30.0,
        auto_approve_on_timeout=True,
        min_confidence_for_auto=0.6,
        max_retries=2,
        fallback_models=["gpt-4o-mini"],
        max_tokens=2048,
        llm_timeout_seconds=600.0,
    ),
    "genre_style": StageDefaults(
        intervention_timeout_seconds=20.0,
        auto_approve_on_timeout=True,
        min_confidence_for_auto=0.8,
        max_retries=1,
        fallback_models=["gpt-4o-mini"],
        max_tokens=2048,
        llm_timeout_seconds=600.0,
    ),
    "visual_rendering": StageDefaults(
        intervention_timeout_seconds=60.0,
        auto_approve_on_timeout=True,
        min_confidence_for_auto=0.5,
        max_retries=2,
        fallback_models=[],
        max_tokens=4096,
        llm_timeout_seconds=900.0,
    ),
    "audio_music": StageDefaults(
        intervention_timeout_seconds=30.0,
        auto_approve_on_timeout=True,
        min_confidence_for_auto=0.6,
        max_retries=2,
        fallback_models=[],
        max_tokens=2048,
        llm_timeout_seconds=600.0,
    ),
    "effects_filters": StageDefaults(
        intervention_timeout_seconds=20.0,
        auto_approve_on_timeout=True,
        min_confidence_for_auto=0.8,
        max_retries=1,
        fallback_models=[],
        max_tokens=2048,
        llm_timeout_seconds=600.0,
    ),
    "assembly": StageDefaults(
        intervention_timeout_seconds=15.0,
        auto_approve_on_timeout=True,
        min_confidence_for_auto=0.9,
        max_retries=1,
        fallback_models=[],
        max_tokens=1536,
        llm_timeout_seconds=600.0,
    ),
    "review": StageDefaults(
        intervention_timeout_seconds=60.0,
        auto_approve_on_timeout=False,
        min_confidence_for_auto=0.5,
        max_retries=1,
        fallback_models=["gpt-4o-mini"],
        max_tokens=2048,
        llm_timeout_seconds=600.0,
    ),
}


def validate_stage_inputs(stage_name: str, state) -> tuple[bool, list[str]]:
    """Validate that required inputs are present in state for a given stage."""
    boundary = STAGE_BOUNDARIES.get(stage_name)
    if not boundary:
        return False, [f"Unknown stage: {stage_name}"]

    missing = []
    for field_name in boundary.required_inputs:
        value = getattr(state, field_name, None)
        if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
            missing.append(field_name)

    return len(missing) == 0, missing