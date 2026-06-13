"""Screenplay writing agent - converts raw script to structured screenplay."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class ScreenplayAgent(BaseReelAgent):
    """
    Takes a raw script/idea and produces a structured screenplay with:
    - Hook (first 2 seconds)
    - Scene breakdowns (setting, time, mood, dialogue, visual directions)
    - Character profiles with voice descriptions
    - Timing estimates per scene
    - Music and pacing suggestions
    """

    @property
    def stage_name(self) -> str:
        return "screenplay"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        return {
            "script": state.script,
            "genre": state.genre or "auto-detect based on content",
            "duration_range": "15-60",
            "iteration": state.iteration_count,
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.8)

        try:
            screenplay = self.parse_json_response(response)
            # Validate required fields
            if not screenplay.get("scenes"):
                return StageResult(
                    stage_name=self.stage_name,
                    success=False,
                    output={"error": "No scenes generated", "raw": response.content},
                    confidence=0.0,
                )

            confidence = self._assess_confidence(screenplay)
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=screenplay,
                confidence=confidence,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse screenplay: {e}", "raw": response.content},
                confidence=0.0,
            )

    def _assess_confidence(self, screenplay: dict) -> float:
        """Heuristic confidence scoring based on output completeness."""
        score = 0.5
        scenes = screenplay.get("scenes", [])
        if scenes:
            score += 0.1
        if len(scenes) >= 2:
            score += 0.1
        if all(s.get("dialogue") for s in scenes):
            score += 0.1
        if screenplay.get("characters"):
            score += 0.1
        if screenplay.get("hook"):
            score += 0.1
        return min(score, 1.0)
