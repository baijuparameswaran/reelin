"""Quality review agent - evaluates the final reel."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class ReviewAgent(BaseReelAgent):
    """
    Evaluates the assembled reel across multiple quality dimensions:
    - Visual quality and consistency
    - Audio quality and sync
    - Pacing and rhythm
    - Genre adherence
    - Hook effectiveness (first 2 seconds)
    - Engagement prediction
    
    Provides specific, actionable improvement suggestions
    that map back to pipeline stages for iterative refinement.
    """

    @property
    def stage_name(self) -> str:
        return "review"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        screenplay = state.screenplay or {}
        style_guide = state.style_guide or {}

        return {
            "screenplay_json": json.dumps(screenplay, indent=2),
            "style_guide_json": json.dumps(style_guide, indent=2),
            "genre": state.genre or "auto",
            "duration": screenplay.get("total_duration_seconds", 30),
            "iteration": state.iteration_count,
            "output_path": state.final_output or "not yet assembled",
            "clips_count": len(state.processed_clips),
            "audio_count": len(state.audio_tracks),
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.4)

        try:
            review = self.parse_json_response(response)

            overall_score = review.get("overall_score", 0.0)
            needs_iteration = review.get("needs_iteration", False)
            iteration_feedback = review.get("iteration_feedback", "")

            confidence = min(0.6 + (state.iteration_count * 0.1), 0.9)

            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=review,
                confidence=confidence,
                metadata={
                    "needs_iteration": needs_iteration,
                    "feedback": iteration_feedback,
                    "overall_score": overall_score,
                },
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse review: {e}", "raw": response.content},
                confidence=0.0,
            )
