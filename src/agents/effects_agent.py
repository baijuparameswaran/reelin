"""Effects and filters agent - post-processing."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class EffectsAgent(BaseReelAgent):
    """
    Plans post-processing effects:
    - Color grading with curves and LUT selection
    - Film grain, vignette, and artistic effects
    - Text overlays with animation
    - Transition types and timing between clips
    - Speed ramping and time effects
    - FFmpeg-compatible filter parameters
    """

    @property
    def stage_name(self) -> str:
        return "effects_filters"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        style_guide = state.style_guide or {}
        scenes = state.screenplay.get("scenes", []) if state.screenplay else []

        return {
            "clips_count": len(state.visual_clips),
            "style_guide_json": json.dumps(style_guide, indent=2),
            "scenes_json": json.dumps(scenes, indent=2),
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.5)

        try:
            effects_plan = self.parse_json_response(response)

            if not effects_plan.get("global_effects") and not effects_plan.get("per_clip_effects"):
                return StageResult(
                    stage_name=self.stage_name,
                    success=False,
                    output={"error": "No effects plan generated", "raw": response.content},
                    confidence=0.0,
                )

            # Generate processed clip paths
            processed = []
            for clip_path in state.visual_clips:
                processed.append(clip_path.replace(".mp4", "_fx.mp4"))

            confidence = self._assess_confidence(effects_plan)
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=effects_plan,
                artifacts=processed,
                confidence=confidence,
                metadata={"effects_plan": effects_plan},
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse effects plan: {e}", "raw": response.content},
                confidence=0.0,
            )

    def _assess_confidence(self, effects_plan: dict) -> float:
        """Score based on completeness of effects plan."""
        score = 0.5
        if effects_plan.get("global_effects", {}).get("color_grading"):
            score += 0.2
        if effects_plan.get("per_clip_effects"):
            score += 0.2
        if effects_plan.get("global_effects", {}).get("film_grain"):
            score += 0.1
        return min(score, 1.0)
