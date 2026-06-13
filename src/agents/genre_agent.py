"""Genre and style agent - establishes visual/audio identity."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class GenreStyleAgent(BaseReelAgent):
    """
    Establishes the complete aesthetic identity:
    - Color palette with grading parameters
    - Typography with animation style
    - Transition design
    - Pacing/rhythm rules with BPM
    - Music mood, genre, and reference tracks
    - Filter presets (grain, vignette, effects)
    - Camera movement style
    """

    @property
    def stage_name(self) -> str:
        return "genre_style"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        screenplay = state.screenplay or {}
        scenes = screenplay.get("scenes", [])
        characters = screenplay.get("characters", [])

        # Extract mood from scenes
        moods = [s.get("mood", "") for s in scenes if s.get("mood")]
        mood = ", ".join(moods[:3]) if moods else "to be determined"

        return {
            "title": screenplay.get("title", "Untitled"),
            "genre": state.genre or "auto-detect",
            "mood": mood,
            "duration": screenplay.get("total_duration_seconds", 30),
            "scene_count": len(scenes),
            "character_names": ", ".join(c.get("name", "") for c in characters),
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.6)

        try:
            style_guide = self.parse_json_response(response)

            if not style_guide.get("color_palette"):
                return StageResult(
                    stage_name=self.stage_name,
                    success=False,
                    output={"error": "Incomplete style guide", "raw": response.content},
                    confidence=0.0,
                )

            confidence = self._assess_confidence(style_guide)
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=style_guide,
                confidence=confidence,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse style guide: {e}", "raw": response.content},
                confidence=0.0,
            )

    def _assess_confidence(self, style_guide: dict) -> float:
        """Score based on completeness of style guide."""
        score = 0.4
        if style_guide.get("color_palette"):
            score += 0.15
        if style_guide.get("music"):
            score += 0.15
        if style_guide.get("typography"):
            score += 0.1
        if style_guide.get("pacing"):
            score += 0.1
        if style_guide.get("filters"):
            score += 0.1
        return min(score, 1.0)
