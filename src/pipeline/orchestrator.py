"""
Reel Factory Pipeline Orchestrator

Manages the sequential execution of agents through the reel creation pipeline.
Features:
- User intervention with configurable timeout (auto-continues on timeout)
- Well-defined stage boundaries with input/output validation
- Complete token usage tracking across all stages
- Default settings for unattended operation
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.agents.base_agent import BaseReelAgent
from src.agents.screenplay_agent import ScreenplayAgent
from src.agents.character_agent import CharacterDesignAgent
from src.agents.genre_agent import GenreStyleAgent
from src.agents.renderer_agent import VisualRendererAgent
from src.agents.audio_agent import AudioComposerAgent
from src.agents.effects_agent import EffectsAgent
from src.agents.assembly_agent import AssemblyAgent
from src.agents.review_agent import ReviewAgent
from src.config.settings import PipelineConfig
from src.pipeline.stage_defaults import (
    STAGE_BOUNDARIES,
    STAGE_DEFAULTS,
    StageDefaults,
    validate_stage_inputs,
)
from src.pipeline.state import PipelineState, StageResult
from src.utils.token_tracker import PipelineTokenTracker
class InteractionMode(Enum):
    PROMPT = "prompt"      # Pause at each stage for user input (with timeout)
    AUTO = "auto"          # Run all stages without stopping
    MINIMAL = "minimal"    # Only pause on errors or low-confidence outputs
class StageAction(Enum):
    APPROVE = "approve"
    MODIFY = "modify"
    SKIP = "skip"
    OVERRIDE_MODEL = "override_model"
    ABORT = "abort"
@dataclass
class StageConfig:
    """Per-stage configuration allowing model and parameter overrides."""
    name: str
    agent_class: type
    default_model: str
    model_override: Optional[str] = None
    defaults: StageDefaults = field(default_factory=StageDefaults)
    custom_params: dict = field(default_factory=dict)

    @property
    def active_model(self) -> str:
        return self.model_override or self.default_model
@dataclass
class InterventionResult:
    """Result of a user intervention attempt."""
    action: StageAction
    timed_out: bool = False
    feedback: str = ""
    model_override: Optional[str] = None
class PipelineOrchestrator:
    """
    Orchestrates the full reel creation pipeline.

    Execution flow per stage:
    1. Validate inputs against stage boundary contract
    2. Execute agent (with token tracking)
    3. If confidence >= threshold in auto/minimal mode -> auto-approve
    4. If prompt mode -> wait for user input (with timeout)
    5. On timeout -> apply default action (approve/skip based on config)
    6. On failure -> retry with fallback model (up to max_retries)
    """

    def __init__(
        self,
        config: PipelineConfig,
        interaction_mode: InteractionMode = InteractionMode.PROMPT,
    ):
        self.config = config
        self.interaction_mode = interaction_mode
        self.state = PipelineState()
        self.token_tracker = PipelineTokenTracker()
        self.stages = self._build_stages()
        self.user_callback: Optional[Callable] = None

    def _build_stages(self) -> list[StageConfig]:
        return [
            StageConfig(
                "screenplay", ScreenplayAgent, self.config.screenplay_model,
                defaults=STAGE_DEFAULTS["screenplay"],
            ),
            StageConfig(
                "character_design", CharacterDesignAgent, self.config.character_model,
                defaults=STAGE_DEFAULTS["character_design"],
            ),
            StageConfig(
                "genre_style", GenreStyleAgent, self.config.screenplay_model,
                defaults=STAGE_DEFAULTS["genre_style"],
            ),
            StageConfig(
                "visual_rendering", VisualRendererAgent, self.config.video_renderer,
                defaults=STAGE_DEFAULTS["visual_rendering"],
            ),
            StageConfig(
                "audio_music", AudioComposerAgent, self.config.music_model,
                defaults=STAGE_DEFAULTS["audio_music"],
            ),
            StageConfig(
                "effects_filters", EffectsAgent, self.config.screenplay_model,
                defaults=STAGE_DEFAULTS["effects_filters"],
            ),
            StageConfig(
                "assembly", AssemblyAgent, self.config.screenplay_model,
                defaults=STAGE_DEFAULTS["assembly"],
            ),
            StageConfig(
                "review", ReviewAgent, self.config.review_model,
                defaults=STAGE_DEFAULTS["review"],
            ),
        ]

    def set_user_callback(self, callback: Callable[[str, StageResult, float], StageAction]):
        """
        Set callback for user intervention at stage boundaries.
        
        Callback signature: (stage_name, result, timeout_seconds) -> StageAction
        The callback itself is responsible for timing out.
        """
        self.user_callback = callback

    async def run(self, script: str, genre: Optional[str] = None) -> PipelineState:
        """Execute the full pipeline from script to final reel."""
        # Auto-refresh local models before running
        await self._refresh_models()
        self.state.script = script
        self.state.genre = genre

        for stage in self.stages:
            # 1. Validate stage boundary inputs (skip for first stage)
            if stage.name != "screenplay":
                valid, missing = validate_stage_inputs(stage.name, self.state)
                if not valid:
                    self.state.add_result(stage.name, StageResult(
                        stage_name=stage.name,
                        success=False,
                        output={"error": f"Missing required inputs: {missing}"},
                        confidence=0.0,
                    ))
                    continue

            # 2. Execute the stage agent
            result = await self._execute_stage_with_retry(stage)
            self.state.add_result(stage.name, result)

            if not result.success:
                continue

            # 3. Determine if user intervention is needed
            intervention = await self._handle_intervention(stage, result)

            if intervention.action == StageAction.ABORT:
                break
            elif intervention.action == StageAction.MODIFY:
                self.state.last_feedback = intervention.feedback
                result = await self._execute_stage_with_retry(stage)
                self.state.add_result(stage.name, result)
            elif intervention.action == StageAction.OVERRIDE_MODEL:
                stage.model_override = intervention.model_override
                result = await self._execute_stage_with_retry(stage)
                self.state.add_result(stage.name, result)

            # 4. Handle iterative review loop
            if stage.name == "review" and result.metadata.get("needs_iteration"):
                if self.state.iteration_count < self.config.max_iterations:
                    self.state.iteration_count += 1
                    self.state.last_feedback = result.metadata.get("feedback", "")
                    return await self._iterate()

        # Generate video output if pipeline completed successfully
        await self._generate_video()

        # Attach final token summary to state
        self.state.token_summary = self.token_tracker.full_summary()
        return self.state

    async def _generate_video(self):
        """Generate actual MP4 video from pipeline outputs."""
        try:
            if not self.state.screenplay:
                return

            from src.rendering.video_generator import VideoGenerator
            generator = VideoGenerator()
            output_path = generator.generate(
                screenplay=self.state.screenplay,
                style_guide=self.state.style_guide or {},
                characters=self.state.characters or [],
                genre=self.state.genre or "default",
            )
            self.state.final_output = output_path
        except Exception as e:
            # Video generation is best-effort; don't fail the pipeline
            self.state.final_output = f"Video generation failed: {e}"

    async def _execute_stage_with_retry(self, stage: StageConfig) -> StageResult:
        """Execute a stage with retry logic and fallback models."""
        models_to_try = [stage.active_model] + stage.defaults.fallback_models
        last_result = None

        for attempt, model in enumerate(models_to_try[:stage.defaults.max_retries + 1]):
            start_time = time.perf_counter()
            try:
                agent = stage.agent_class(
                    model=model,
                    token_tracker=self.token_tracker,
                    **stage.custom_params,
                )
                result = await agent.execute(self.state)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if result.success:
                    return result
                last_result = result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                last_result = StageResult(
                    stage_name=stage.name,
                    success=False,
                    output={"error": str(e)},
                    confidence=0.0,
                )

        return last_result or StageResult(
            stage_name=stage.name, success=False, output={"error": "All retries exhausted"}, confidence=0.0,
        )

    async def _handle_intervention(self, stage: StageConfig, result: StageResult) -> InterventionResult:
        """Handle user intervention with timeout-based auto-continue."""
        defaults = stage.defaults

        # Auto mode: never pause
        if self.interaction_mode == InteractionMode.AUTO:
            return InterventionResult(action=StageAction.APPROVE)

        # Minimal mode: only pause on low confidence
        if self.interaction_mode == InteractionMode.MINIMAL:
            if result.confidence >= defaults.min_confidence_for_auto:
                return InterventionResult(action=StageAction.APPROVE)

        # Prompt mode: wait for user with timeout
        if self.user_callback:
            try:
                action = await asyncio.wait_for(
                    self._async_user_callback(stage.name, result, defaults.intervention_timeout_seconds),
                    timeout=defaults.intervention_timeout_seconds,
                )
                return InterventionResult(action=action)
            except asyncio.TimeoutError:
                # Timeout: apply default action
                if defaults.auto_approve_on_timeout:
                    return InterventionResult(action=StageAction.APPROVE, timed_out=True)
                else:
                    return InterventionResult(action=StageAction.APPROVE, timed_out=True)

        # No callback set: auto-approve
        return InterventionResult(action=StageAction.APPROVE)

    async def _async_user_callback(
        self, stage_name: str, result: StageResult, timeout: float
    ) -> StageAction:
        """Wrap sync callback in async for timeout support."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.user_callback, stage_name, result, timeout
        )
    async def _refresh_models(self):
        """Auto-refresh local models: detect deprecated, pull upgrades, remove obsolete."""
        try:
            from src.llm.ollama_provider import OllamaProvider
            from src.llm.model_registry import ModelRefreshManager

            provider = OllamaProvider()
            if not await provider.is_available():
                return  # Ollama not running, skip silently

            manager = ModelRefreshManager(provider)
            result = await manager.auto_refresh()

            if result.get("status") == "refreshed":
                pulled = result.get("pulled", [])
                removed = result.get("removed", [])
                if pulled or removed:
                    # Reset cached model list so resolve_model picks up changes
                    provider._available_models = None
        except Exception:
            pass  # Never block pipeline on refresh failure

    async def _iterate(self) -> PipelineState:
        """Re-run pipeline from visual rendering onward with feedback."""
        restart_idx = next(
            (i for i, s in enumerate(self.stages) if s.name == "visual_rendering"), 0
        )
        for stage in self.stages[restart_idx:]:
            valid, missing = validate_stage_inputs(stage.name, self.state)
            if not valid:
                continue
            result = await self._execute_stage_with_retry(stage)
            self.state.add_result(stage.name, result)

        self.state.token_summary = self.token_tracker.full_summary()
        return self.state
