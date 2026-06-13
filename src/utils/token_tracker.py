"""
Token usage tracking across the entire pipeline.

Tracks per-call, per-stage, and pipeline-level token usage with:
- Model and provider attribution (ollama vs openai vs anthropic)
- Cost estimation for cloud providers
- Latency monitoring
- Budget limit warnings
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""  # "ollama", "openai", "anthropic"
    stage: str = ""
    timestamp: float = field(default_factory=time.time)
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Pricing per 1M tokens: (input_price, output_price)
# Local models (Ollama) have zero cost
MODEL_PRICING = {
    # Cloud - OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Cloud - Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-3-20240307": (0.25, 1.25),
    "claude-opus-4-20250514": (15.00, 75.00),
    # Cloud - Media (token-based pricing not applicable, placeholder)
    "flux-1.1-pro": (0.0, 0.0),
    "runway-gen3": (0.0, 0.0),
    "elevenlabs-v2": (0.0, 0.0),
    "suno-v4": (0.0, 0.0),
    # Local (always free)
    "_local_default": (0.0, 0.0),
}


def _is_local_provider(provider: str) -> bool:
    return provider in ("ollama", "local", "")


@dataclass
class PipelineTokenTracker:
    """
    Tracks all token usage across the entire pipeline run.

    Features:
    - Per-call recording with model + provider attribution
    - Per-stage aggregation
    - Cost estimation (zero for local models)
    - Latency tracking
    - Budget limit checking
    """
    usages: list[TokenUsage] = field(default_factory=list)
    _stage_totals: dict[str, dict] = field(default_factory=dict)

    def record(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        provider: str = "",
    ) -> TokenUsage:
        """Record a single LLM call's token usage."""
        cost = self._estimate_cost(model, input_tokens, output_tokens, provider)
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider=provider,
            stage=stage,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        self.usages.append(usage)
        self._update_stage_totals(stage, usage)
        return usage

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int, provider: str) -> float:
        """Estimate cost. Local models are always free."""
        if _is_local_provider(provider):
            return 0.0
        pricing = MODEL_PRICING.get(model, (0.0, 0.0))
        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    def _update_stage_totals(self, stage: str, usage: TokenUsage):
        if stage not in self._stage_totals:
            self._stage_totals[stage] = {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "cost_usd": 0.0, "call_count": 0, "total_latency_ms": 0.0,
                "model": "", "provider": "",
            }
        t = self._stage_totals[stage]
        t["input_tokens"] += usage.input_tokens
        t["output_tokens"] += usage.output_tokens
        t["total_tokens"] += usage.total_tokens
        t["cost_usd"] += usage.cost_usd
        t["call_count"] += 1
        t["total_latency_ms"] += usage.latency_ms
        # Track last model/provider used for this stage
        t["model"] = usage.model
        t["provider"] = usage.provider

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usages)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    @property
    def total_calls(self) -> int:
        return len(self.usages)

    @property
    def total_latency_ms(self) -> float:
        return sum(u.latency_ms for u in self.usages)

    def stage_summary(self, stage: str) -> Optional[dict]:
        """Get aggregated totals for a specific stage."""
        return self._stage_totals.get(stage)

    def check_budget(self, max_per_stage: int = 50000, max_total: int = 300000) -> dict:
        """Check token usage against budget limits."""
        warnings = []
        for stage, totals in self._stage_totals.items():
            if totals["total_tokens"] > max_per_stage:
                warnings.append(f"Stage {stage} exceeded budget: {totals['total_tokens']} > {max_per_stage}")
        if self.total_tokens > max_total:
            warnings.append(f"Pipeline total exceeded budget: {self.total_tokens} > {max_total}")
        return {"within_budget": len(warnings) == 0, "warnings": warnings}

    def full_summary(self) -> dict:
        """Complete summary of all token usage for display."""
        return {
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
                "call_count": self.total_calls,
                "total_latency_ms": round(self.total_latency_ms, 2),
            },
            "per_stage": dict(self._stage_totals),
            "per_call": [
                {
                    "stage": u.stage,
                    "model": u.model,
                    "provider": u.provider,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost_usd": round(u.cost_usd, 6),
                    "latency_ms": round(u.latency_ms, 2),
                }
                for u in self.usages
            ],
        }
