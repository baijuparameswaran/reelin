"""Audio composition agent - generates music, voiceover, SFX."""
import json
from typing import Optional

from src.agents.base_agent import BaseReelAgent
from src.pipeline.state import PipelineState, StageResult


class AudioComposerAgent(BaseReelAgent):
    """
    Plans the complete audio landscape:
    - Background music generation prompt (for Suno/Udio)
    - Voiceover direction with timing and emotion (for ElevenLabs/Azure TTS)
    - Sound effects placement and sourcing
    - Mix levels and mastering settings
    
    The actual audio generation is delegated to specialized APIs
    based on the audio plan this agent produces.
    """

    @property
    def stage_name(self) -> str:
        return "audio_music"

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        screenplay = state.screenplay or {}
        style_guide = state.style_guide or {}
        music_guide = style_guide.get("music", {})
        characters = state.characters if state.characters else []

        return {
            "screenplay_json": json.dumps(screenplay, indent=2),
            "music_guide_json": json.dumps(music_guide, indent=2),
            "duration": screenplay.get("total_duration_seconds", 30),
            "characters_json": json.dumps(characters, indent=2),
        }

    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        user_prompt = self.format_prompt(state, feedback)
        response = await self.call_llm(user_prompt, temperature=0.7)

        try:
            audio_plan = self.parse_json_response(response)

            if not audio_plan.get("music") and not audio_plan.get("voiceover"):
                return StageResult(
                    stage_name=self.stage_name,
                    success=False,
                    output={"error": "No audio plan generated", "raw": response.content},
                    confidence=0.0,
                )

            # Generate artifact paths from the plan
            tracks = []
            if audio_plan.get("music") or audio_plan.get("soundtrack") or audio_plan.get("background_music"):
                tracks.append("/tmp/reel/audio/background_music.mp3")
            for i, vo in enumerate(audio_plan.get("voiceover", audio_plan.get("voice_over", audio_plan.get("narration", [])))):
                char = vo.get("character", f"voice_{i}") if isinstance(vo, dict) else f"voice_{i}"
                tracks.append(f"/tmp/reel/audio/vo_{char}.mp3")
            for i, sfx in enumerate(audio_plan.get("sfx", audio_plan.get("sound_effects", audio_plan.get("effects", [])))):
                tracks.append(f"/tmp/reel/audio/sfx_{i}.mp3")
            # Fallback: if no tracks extracted, check for generic tracks/audio keys
            if not tracks:
                for key in ("tracks", "audio", "audio_tracks", "elements"):
                    items = audio_plan.get(key, [])
                    if isinstance(items, list) and items:
                        for i, item in enumerate(items):
                            tracks.append(f"/tmp/reel/audio/track_{i}.mp3")
                        break
            # Ensure at least one track so downstream stages don't fail
            if not tracks:
                tracks.append("/tmp/reel/audio/main_track.mp3")

            confidence = self._assess_confidence(audio_plan)
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                output=audio_plan,
                artifacts=tracks,
                confidence=confidence,
                metadata={"audio_plan": audio_plan},
            )
        except (json.JSONDecodeError, KeyError) as e:
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                output={"error": f"Failed to parse audio plan: {e}", "raw": response.content},
                confidence=0.0,
            )

    def _assess_confidence(self, audio_plan: dict) -> float:
        """Score based on completeness of audio plan."""
        score = 0.4
        if audio_plan.get("music", {}).get("prompt"):
            score += 0.2
        if audio_plan.get("voiceover"):
            score += 0.15
        if audio_plan.get("mix_settings"):
            score += 0.15
        if audio_plan.get("sfx"):
            score += 0.1
        return min(score, 1.0)
