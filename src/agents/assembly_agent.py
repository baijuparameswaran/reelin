"""Video assembly agent - combines all elements into final reel."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class AssemblyAgent(BaseReelAgent):
    """
    Plans final video assembly:
    - Clip sequencing according to screenplay order
    - Audio track mixing (music + voiceover + SFX)
    - Output codec and format specifications
    - Trim points and timeline positioning
    - Final adjustments (normalization, padding, looping)
    """

    @property
    def stage_name(self) -> str:
        return "assembly"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        screenplay = state.screenplay or {}
        scenes = screenplay.get("scenes", [])
        scene_order = [s.get("scene_number", i) for i, s in enumerate(scenes)]

        return {
            "clips_count": len(state.processed_clips),
            "audio_count": len(state.audio_tracks),
            "output_format": "vertical_9_16 (1080x1920)",
            "duration": screenplay.get("total_duration_seconds", 30),
            "scene_order": json.dumps(scene_order),
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.3)

        try:
            assembly_plan = self.parse_json_response(response)

            if not (assembly_plan.get("output_specs") or assembly_plan.get("output") or assembly_plan.get("timeline") or assembly_plan.get("sequence") or assembly_plan.get("scenes")):
                return StageResult(
                    stage_name=self.stage_name,
                    success=False,
                    output={"error": "No assembly plan generated", "raw": response.content},
                    confidence=0.0,
                )

            output_path = "/tmp/reel/output/final_reel.mp4"
            confidence = self._assess_confidence(assembly_plan)

            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=assembly_plan,
                artifacts=[output_path],
                confidence=confidence,
                metadata={"assembly_plan": assembly_plan, "output_path": output_path},
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse assembly plan: {e}", "raw": response.content},
                confidence=0.0,
            )

    def _assess_confidence(self, assembly_plan: dict) -> float:
        """Score based on completeness of assembly plan."""
        score = 0.5
        if assembly_plan.get("output_specs", {}).get("resolution"):
            score += 0.15
        if assembly_plan.get("clip_sequence"):
            score += 0.2
        if assembly_plan.get("audio_mix"):
            score += 0.15
        return min(score, 1.0)
