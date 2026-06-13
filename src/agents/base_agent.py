"""Base agent class for all pipeline stages."""
import json
import time
from abc import ABC, abstractmethod
from typing import Optional

from src.llm.router import LLMRouter, LLMResponse
from src.pipeline.state import PipelineState, StageResult
from src.prompts.templates import AGENT_PROMPTS
from src.utils.token_tracker import PipelineTokenTracker


class BaseReelAgent(ABC):
    """
    Abstract base for all reel pipeline agents.

    Each agent:
    - Receives the full pipeline state
    - Uses a configurable LLM model
    - Has a system prompt and user prompt template from AGENT_PROMPTS
    - Tracks token usage via the shared PipelineTokenTracker
    - Returns a StageResult with confidence score
    - Can incorporate user feedback on retry

    Boundary contract:
    - Agents MUST only read state fields listed in their stage boundary's required_inputs
    - Agents MUST only write to fields listed in their produced_outputs
    - Violations are logged but not enforced at runtime (trust boundary)
    """

    def __init__(self, model: str, token_tracker: Optional[PipelineTokenTracker] = None, **kwargs):
        self.model = model
        self.token_tracker = token_tracker or PipelineTokenTracker()
        self.router = LLMRouter()
        self.kwargs = kwargs

    @abstractmethod
    async def execute(self, state: PipelineState, feedback: Optional[str] = None) -> StageResult:
        """Execute this agent's task and return results."""
        ...

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Name of this pipeline stage."""
        ...

    @property
    def prompts(self) -> dict:
        """Get the prompt templates for this agent's stage."""
        return AGENT_PROMPTS.get(self.stage_name, {})

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return self.prompts.get("system_prompt", "You are a helpful assistant.")

    def track_tokens(self, input_tokens: int, output_tokens: int, latency_ms: float = 0.0, model: Optional[str] = None, provider: Optional[str] = None):
        """Record token usage for this agent call."""
        self.token_tracker.record(
            stage=self.stage_name,
            model=model or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            provider=provider or "unknown",
        )

    async def call_llm(
        self,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = "json",
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """
        Call the configured LLM via the router and track tokens.

        Args:
            user_prompt: The formatted user prompt
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum response length (defaults to stage config)
            response_format: "json" for structured output, None for plain text
            timeout: LLM call timeout in seconds (defaults to stage config)

        Returns:
            LLMResponse with content, tokens, and latency
        """
        from src.pipeline.stage_defaults import STAGE_DEFAULTS
        stage_defaults = STAGE_DEFAULTS.get(self.stage_name)
        effective_max_tokens = max_tokens or (stage_defaults.max_tokens if stage_defaults else 2048)
        effective_timeout = timeout or (stage_defaults.llm_timeout_seconds if stage_defaults else 600.0)

        response = await self.router.call(
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            response_format=response_format,
            stage_name=self.stage_name,
            timeout=effective_timeout,
        )
        self.track_tokens(response.input_tokens, response.output_tokens, response.latency_ms, model=response.model, provider=response.provider)
        return response

    def format_prompt(self, state: PipelineState, feedback: Optional[str] = None, **kwargs) -> str:
        """
        Format the user prompt template with state data and additional kwargs.

        Subclasses should override _get_prompt_vars() to provide stage-specific variables.
        """
        template = self.prompts.get("user_prompt_template", "")
        variables = self._get_prompt_vars(state, feedback)
        variables.update(kwargs)

        # Add feedback section
        if feedback:
            variables["feedback_section"] = (
                f"\n**User Feedback from Previous Attempt:**\n{feedback}\n"
                "Please address this feedback in your output."
            )
        else:
            variables["feedback_section"] = ""

        try:
            return template.format(**variables)
        except KeyError as e:
            # If a template variable is missing, return template with available vars
            for key, value in variables.items():
                template = template.replace("{" + key + "}", str(value))
            return template

    def _get_prompt_vars(self, state: PipelineState, feedback: Optional[str] = None) -> dict:
        """
        Get template variables from pipeline state.
        Override in subclasses for stage-specific variable extraction.
        """
        return {
            "script": state.script,
            "genre": state.genre or "auto-detect based on content",
            "feedback": feedback or "",
        }

    def parse_json_response(self, response: LLMResponse) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        return response.parse_json()
