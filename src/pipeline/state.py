"""Pipeline state management."""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StageResult:
    """Output from a single pipeline stage."""
    stage_name: str
    success: bool
    output: Any
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)


@dataclass
class PipelineState:
    """
    Shared state flowing through the pipeline.
    Each stage reads from and writes to this state.
    
    Fields are organized by which stage produces them (see stage_defaults.py boundaries).
    """
    # Input
    script: str = ""
    genre: Optional[str] = None

    # Control
    iteration_count: int = 0
    last_feedback: str = ""

    # Stage 1 output: screenplay
    screenplay: Optional[dict] = None

    # Stage 2 output: character_design
    characters: list = field(default_factory=list)

    # Stage 3 output: genre_style
    style_guide: Optional[dict] = None

    # Stage 4 output: visual_rendering
    visual_clips: list = field(default_factory=list)

    # Stage 5 output: audio_music
    audio_tracks: list = field(default_factory=list)

    # Stage 6 output: effects_filters
    processed_clips: list = field(default_factory=list)

    # Stage 7 output: assembly
    final_output: Optional[str] = None

    # Stage 8 output: review
    review_score: Optional[float] = None

    # Token tracking (populated at end of run)
    token_summary: Optional[dict] = None

    # Full history of all stage results
    results: dict = field(default_factory=dict)

    def add_result(self, stage_name: str, result: StageResult):
        self.results[stage_name] = result
        if stage_name == "screenplay":
            self.screenplay = result.output
        elif stage_name == "character_design":
            self.characters = result.output if isinstance(result.output, list) else [result.output] if result.output else []
        elif stage_name == "genre_style":
            self.style_guide = result.output
        elif stage_name == "visual_rendering":
            self.visual_clips = result.artifacts if result.artifacts else [f"/tmp/reel/clips/scene_{i}.mp4" for i in range(1, 4)]
        elif stage_name == "audio_music":
            self.audio_tracks = result.artifacts if result.artifacts else ["/tmp/reel/audio/main_track.mp3"]
        elif stage_name == "effects_filters":
            self.processed_clips = result.artifacts if result.artifacts else self.visual_clips
        elif stage_name == "assembly":
            self.final_output = (result.artifacts[0] if result.artifacts else "/tmp/reel/output/final.mp4")
        elif stage_name == "review":
            self.review_score = result.output.get("score") if isinstance(result.output, dict) else None
