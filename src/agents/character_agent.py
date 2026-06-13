"""Character design agent - creates visual references for characters."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class CharacterDesignAgent(BaseReelAgent):
    """
    Takes character descriptions from screenplay and produces:
    - Detailed visual prompts optimized for image generation models
    - Negative prompts for quality control
    - Voice casting descriptions for TTS
    - Consistency tags for maintaining look across frames
    - Color palettes per character
    """

    @property
    def stage_name(self) -> str:
        return "character_design"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        characters = state.screenplay.get("characters", []) if state.screenplay else []
        style_context = ""
        if state.style_guide:
            style_context = json.dumps(state.style_guide, indent=2)
        else:
            style_context = f"Genre: {state.genre or 'auto'}"

        return {
            "characters_json": json.dumps(characters, indent=2),
            "genre": state.genre or "auto",
            "rendering_mode": self.kwargs.get("rendering_mode", "animated"),
            "style_context": style_context,
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.7)

        try:
            result = self.parse_json_response(response)
            characters = result.get("characters", [])

            if not characters:
                return StageResult(
                    stage_name=self.stage_name,
                    success=False,
                    output={"error": "No characters designed", "raw": response.content},
                    confidence=0.0,
                )

            confidence = self._assess_confidence(characters)
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=characters,
                confidence=confidence,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse character designs: {e}", "raw": response.content},
                confidence=0.0,
            )

    def _assess_confidence(self, characters: list) -> float:
        """Score based on completeness of character descriptions."""
        if not characters:
            return 0.0
        score = 0.5
        for char in characters:
            if char.get("visual_prompt") and len(char["visual_prompt"]) > 30:
                score += 0.1
            if char.get("consistency_tags"):
                score += 0.05
            if char.get("voice_casting"):
                score += 0.05
        return min(score, 1.0)
