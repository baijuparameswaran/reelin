"""Visual rendering agent - generates video frames/clips."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class VisualRendererAgent(BaseReelAgent):
    """
    Generates visual content for each scene by creating a rendering plan:
    - Keyframe image prompts for image generation (Flux, DALL-E)
    - Frame interpolation instructions for animation
    - Camera movement specifications
    - Character placement and expression mapping
    
    The actual image/video generation is delegated to external APIs
    based on the rendering plan this agent produces.
    """

    @property
    def stage_name(self) -> str:
        return "visual_rendering"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        scenes = state.screenplay.get("scenes", []) if state.screenplay else []
        characters = state.characters if state.characters else []
        style_guide = state.style_guide or {}

        # Determine rendering mode from characters or kwargs
        rendering_mode = "animated"
        if characters and isinstance(characters[0], dict):
            rendering_mode = characters[0].get("rendering_mode", "animated")
        rendering_mode = self.kwargs.get("rendering_mode", rendering_mode)

        return {
            "scenes_json": json.dumps(scenes, indent=2),
            "characters_json": json.dumps(characters, indent=2),
            "style_guide_json": json.dumps(style_guide, indent=2),
            "rendering_mode": rendering_mode,
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.6)

        try:
            rendering_plan = self.parse_json_response(response)
            scenes = rendering_plan.get("scenes", [])

            if not scenes:
                return StageResult(
                    stage_name=self.stage_name,
                    success=False,
                    output={"error": "No rendering plan generated", "raw": response.content},
                    confidence=0.0,
                )

            # Generate clip paths from the rendering plan
            clips = []
            for scene in scenes:
                scene_num = scene.get("scene_number", 0)
                clip_path = f"/tmp/reel/clips/scene_{scene_num}.mp4"
                clips.append(clip_path)

            confidence = self._assess_confidence(scenes)
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=rendering_plan,
                artifacts=clips,
                confidence=confidence,
                metadata={"rendering_plan": rendering_plan},
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse rendering plan: {e}", "raw": response.content},
                confidence=0.0,
            )

    def _assess_confidence(self, scenes: list) -> float:
        """Score based on completeness of rendering plan."""
        if not scenes:
            return 0.0
        score = 0.4
        for scene in scenes:
            keyframes = scene.get("keyframes", [])
            if keyframes:
                score += 0.1
                if all(kf.get("image_prompt") for kf in keyframes):
                    score += 0.05
            if scene.get("interpolation"):
                score += 0.05
        return min(score, 1.0)
