"""
Pipeline orchestration: state management, stage boundaries, and execution control.

The orchestrator runs agents sequentially with:
- Auto-refresh of local models before each run
- Boundary validation between stages
- Timeout-based user intervention with auto-continue
- Retry with fallback models on failure
- Iterative improvement loop via review feedback
"""
from src.pipeline.state import PipelineState, StageResult

__all__ = [
    "PipelineState",
    "StageResult",
]
