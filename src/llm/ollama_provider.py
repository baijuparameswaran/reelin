"""
Ollama provider - local/offline LLM inference via Ollama.

Ollama runs open-source models locally and exposes an OpenAI-compatible API.
Install: https://ollama.com
Pull models: ollama pull llama3:8b-instruct-q5_K_M
"""
import json
import os
from typing import Optional


def _get_httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for Ollama support. Install with: pip install httpx"
        )


# Avoid circular import - import LLMResponse at function level
def _make_response(content, input_tokens, output_tokens, model, latency_ms, raw_response=None):
    from src.llm.router import LLMResponse
    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        latency_ms=latency_ms,
        raw_response=raw_response,
    )


# Recommended local models per pipeline stage
STAGE_LOCAL_MODELS = {
    "screenplay": "llama3:8b-instruct-q5_K_M",
    "character_design": "llama3:8b-instruct-q5_K_M",
    "genre_style": "mistral:7b-instruct-q5_K_M",
    "visual_rendering": "mistral:7b-instruct-q5_K_M",
    "audio_music": "mistral:7b-instruct-q5_K_M",
    "effects_filters": "qwen2:7b-instruct-q5_K_M",
    "assembly": "phi3:mini",
    "review": "llama3:8b-instruct-q5_K_M",
}

# Fallback chain
MODEL_FALLBACKS = {
    "llama3:8b-instruct-q5_K_M": ["llama3:8b", "llama3:latest", "llama3.1:8b"],
    "mistral:7b-instruct-q5_K_M": ["mistral:7b-instruct", "mistral:latest", "mistral:7b"],
    "qwen2:7b-instruct-q5_K_M": ["qwen2:7b", "qwen2:latest", "qwen2.5:7b"],
    "phi3:mini": ["phi3:latest", "phi3:mini-4k"],
}


class OllamaProvider:
    """
    Local LLM provider using Ollama HTTP API.

    Ollama exposes an OpenAI-compatible endpoint at localhost:11434.
    Supports JSON mode, model management, and automatic model resolution.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._available_models: Optional[list[str]] = None

    async def call(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[str] = None,
        timeout: float = 600.0,
    ):
        """Call a local model via Ollama /api/chat endpoint."""
        httpx = _get_httpx()

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if response_format == "json":
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        total_duration_ns = data.get("total_duration", 0)
        latency_ms = total_duration_ns / 1_000_000

        return _make_response(content, input_tokens, output_tokens, model, latency_ms, data)

    async def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            httpx = _get_httpx()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List locally available models."""
        if self._available_models is not None:
            return self._available_models
        try:
            httpx = _get_httpx()
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                self._available_models = [m["name"] for m in data.get("models", [])]
                return self._available_models
        except Exception:
            return []

    async def resolve_model(self, stage_name: str) -> Optional[str]:
        """
        Resolve the best available local model for a given stage.
        Returns None if no suitable model is available locally.
        """
        preferred = STAGE_LOCAL_MODELS.get(stage_name)
        if not preferred:
            return None

        available = await self.list_models()
        if not available:
            return None

        if preferred in available:
            return preferred

        # Check without quantization suffix
        base_name = preferred.split(":")[0]
        for avail in available:
            if avail.startswith(base_name):
                return avail

        # Try fallbacks
        fallbacks = MODEL_FALLBACKS.get(preferred, [])
        for fb in fallbacks:
            if fb in available:
                return fb
            fb_base = fb.split(":")[0]
            for avail in available:
                if avail.startswith(fb_base):
                    return avail

        # Last resort: any available model
        return available[0] if available else None

    async def pull_model(self, model_name: str) -> bool:
        """Pull/download a model. Returns True on success."""
        try:
            httpx = _get_httpx()
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": False},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def pull_stage_models(self, stages: Optional[list[str]] = None) -> dict[str, bool]:
        """Pull recommended models for specified stages (or all)."""
        if stages is None:
            stages = list(STAGE_LOCAL_MODELS.keys())

        models_to_pull = set()
        for stage in stages:
            model = STAGE_LOCAL_MODELS.get(stage)
            if model:
                models_to_pull.add(model)

        results = {}
        for model in models_to_pull:
            results[model] = await self.pull_model(model)
        return results
