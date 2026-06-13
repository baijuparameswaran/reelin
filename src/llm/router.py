"""
LLM Router - unified interface for calling different LLM providers.

Priority order:
1. Local/offline models via Ollama (default, no API key needed)
2. OpenAI (if OPENAI_API_KEY is set and model starts with gpt-/o1/o3)
3. Anthropic (if ANTHROPIC_API_KEY is set and model starts with claude-)

The router automatically detects Ollama availability and resolves
the best local model for each pipeline stage.
"""
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProviderMode(Enum):
    """LLM provider selection mode."""
    LOCAL_FIRST = "local_first"
    CLOUD_FIRST = "cloud_first"
    LOCAL_ONLY = "local_only"
    CLOUD_ONLY = "cloud_only"


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    provider: str = "unknown"
    raw_response: Optional[dict] = None

    def parse_json(self) -> dict:
        """Attempt to parse response content as JSON."""
        text = self.content.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            lines = [l for l in lines[1:] if not l.strip() == '```']
            text = '\n'.join(lines)
        return json.loads(text)


CLOUD_PROVIDER_MAP = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude-": "anthropic",
    "flux-": "replicate",
    "runway-": "runway",
    "suno-": "suno",
    "elevenlabs-": "elevenlabs",
    "stable-": "replicate",
}


def _detect_cloud_provider(model: str) -> str:
    """Detect cloud provider from model name prefix."""
    for prefix, provider in CLOUD_PROVIDER_MAP.items():
        if model.startswith(prefix):
            return provider
    return "openai"


def _is_local_model(model: str) -> bool:
    """Check if a model name refers to a local/Ollama model."""
    local_prefixes = (
        "llama", "mistral", "qwen", "phi", "gemma", "codellama",
        "deepseek", "starcoder", "vicuna", "neural-chat", "orca",
        "solar", "yi", "zephyr", "dolphin", "nous-hermes",
    )
    model_lower = model.lower().split(":")[0]
    return any(model_lower.startswith(p) for p in local_prefixes)


class LLMRouter:
    """
    Routes LLM calls to the appropriate provider.

    Default behavior (LOCAL_FIRST):
    - Checks if Ollama is available
    - Resolves the best local model for the requested stage
    - Falls back to cloud APIs if local is unavailable

    Environment variables:
    - REEL_LLM_MODE: local_first|cloud_first|local_only|cloud_only
    - OLLAMA_HOST: Ollama server URL (default http://localhost:11434)
    - OPENAI_API_KEY: OpenAI API key (for cloud fallback)
    - ANTHROPIC_API_KEY: Anthropic API key (for cloud fallback)
    """

    def __init__(self, mode: Optional[ProviderMode] = None):
        if mode:
            self.mode = mode
        elif os.environ.get("REEL_LLM_MODE"):
            self.mode = ProviderMode(os.environ["REEL_LLM_MODE"])
        else:
            # Default: local_only unless cloud API keys are available
            has_cloud_keys = bool(
                os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            )
            self.mode = ProviderMode.LOCAL_FIRST if has_cloud_keys else ProviderMode.LOCAL_ONLY
        self._openai_client = None
        self._anthropic_client = None
        self._ollama = None
        self._ollama_checked = False
        self._ollama_available = False

    @property
    def ollama(self):
        if self._ollama is None:
            from src.llm.ollama_provider import OllamaProvider
            self._ollama = OllamaProvider()
        return self._ollama

    @property
    def openai_client(self):
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._openai_client

    @property
    def anthropic_client(self):
        if self._anthropic_client is None:
            try:
                from anthropic import AsyncAnthropic
                self._anthropic_client = AsyncAnthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._anthropic_client

    async def _check_ollama(self) -> bool:
        """Check Ollama availability (cached after first check)."""
        if not self._ollama_checked:
            self._ollama_available = await self.ollama.is_available()
            self._ollama_checked = True
        return self._ollama_available

    async def call(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[str] = None,
        stage_name: Optional[str] = None,
        timeout: float = 600.0,
    ) -> LLMResponse:
        """
        Route an LLM call based on the configured mode.

        Args:
            model: Model identifier (cloud model name or local model name)
            system_prompt: System/role instruction
            user_prompt: User message content
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            response_format: "json" for structured output, None for plain text
            stage_name: Pipeline stage (for resolving best local model)

        Returns:
            LLMResponse with content, token usage, provider info, and latency
        """
        start = time.perf_counter()

        if self.mode == ProviderMode.LOCAL_ONLY:
            response = await self._call_local(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name, timeout
            )
        elif self.mode == ProviderMode.CLOUD_ONLY:
            response = await self._call_cloud(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format
            )
        elif self.mode == ProviderMode.LOCAL_FIRST:
            response = await self._call_local_first(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name, timeout
            )
        else:
            response = await self._call_cloud_first(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name, timeout
            )

        response.latency_ms = (time.perf_counter() - start) * 1000
        return response

    async def _call_local_first(
        self, model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name, timeout=600.0
    ) -> LLMResponse:
        """Try local model first, fall back to cloud on failure."""
        local_error = None
        if await self._check_ollama():
            try:
                return await self._call_local(
                    model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name, timeout
                )
            except Exception as e:
                local_error = e

        # Only fall back to cloud if the model is actually callable (openai/anthropic)
        provider = _detect_cloud_provider(model)
        if provider in ("openai", "anthropic"):
            return await self._call_cloud(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format
            )

        # No viable fallback - re-raise the local error or report
        if local_error:
            raise local_error
        raise RuntimeError(f"No callable provider for model '{model}' (provider: {provider})")

    async def _call_cloud_first(
        self, model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name, timeout=600.0
    ) -> LLMResponse:
        """Try cloud APIs first, fall back to local on failure."""
        try:
            return await self._call_cloud(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format
            )
        except Exception:
            pass

        if await self._check_ollama():
            return await self._call_local(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name
            )

        raise RuntimeError(
            "No LLM provider available. Set OPENAI_API_KEY/ANTHROPIC_API_KEY or start Ollama."
        )

    async def _call_local(
        self, model, system_prompt, user_prompt, temperature, max_tokens, response_format, stage_name, timeout=600.0
    ) -> LLMResponse:
        """Call via Ollama with automatic model resolution."""
        if _is_local_model(model):
            local_model = model
        elif stage_name:
            local_model = await self.ollama.resolve_model(stage_name)
            if not local_model:
                raise RuntimeError(f"No local model available for stage: {stage_name}")
        else:
            models = await self.ollama.list_models()
            if not models:
                raise RuntimeError("No models available in Ollama")
            local_model = models[0]

        response = await self.ollama.call(
            model=local_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
        )
        response.provider = "ollama"
        return response

    async def _call_cloud(
        self, model, system_prompt, user_prompt, temperature, max_tokens, response_format
    ) -> LLMResponse:
        """Call a cloud LLM provider."""
        provider = _detect_cloud_provider(model)

        if provider == "openai":
            response = await self._call_openai(
                model, system_prompt, user_prompt, temperature, max_tokens, response_format
            )
        elif provider == "anthropic":
            response = await self._call_anthropic(
                model, system_prompt, user_prompt, temperature, max_tokens
            )
        else:
            raise RuntimeError(
                f"Cloud provider '{provider}' for model '{model}' is not implemented. "
                f"Use local_first or local_only mode for this stage."
            )

        return response

    async def _call_openai(
        self, model, system_prompt, user_prompt, temperature, max_tokens, response_format
    ) -> LLMResponse:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self.openai_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=model,
            latency_ms=0.0,
            provider="openai",
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )

    async def _call_anthropic(
        self, model, system_prompt, user_prompt, temperature, max_tokens
    ) -> LLMResponse:
        resp = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            input_tokens=resp.usage.input_tokens if resp.usage else 0,
            output_tokens=resp.usage.output_tokens if resp.usage else 0,
            model=model,
            latency_ms=0.0,
            provider="anthropic",
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
