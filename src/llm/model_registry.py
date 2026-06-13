"""
Model registry - tracks model versions, deprecation, and upgrade paths.

Provides automatic refresh logic that:
1. Checks if currently installed models are deprecated
2. Identifies newer/better replacements
3. Pulls upgrades and removes obsolete models
4. Persists state in a local JSON file to track what was installed when
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Model lifecycle: recommended -> deprecated -> obsolete
# recommended: actively suggested for new installs
# deprecated: still works but a better option exists (auto-upgrade on refresh)
# obsolete: no longer functional or available (force-remove on refresh)

@dataclass
class ModelEntry:
    """A model in the registry with lifecycle metadata."""
    name: str
    status: str  # "recommended" | "deprecated" | "obsolete"
    successor: Optional[str] = None  # What replaces this model
    reason: str = ""  # Why deprecated/obsolete
    added_date: str = ""  # When this model was added to registry
    deprecated_date: str = ""  # When it was marked deprecated
    size_gb: float = 0.0
    capabilities: list = field(default_factory=list)  # ["json", "creative", "technical"]
    min_ram_gb: float = 4.0


# Central model registry with upgrade paths
# This is the source of truth for which models are current vs obsolete
MODEL_REGISTRY: dict[str, ModelEntry] = {
    # === Llama family ===
    "llama2:7b": ModelEntry(
        name="llama2:7b",
        status="obsolete",
        successor="llama3:8b-instruct-q5_K_M",
        reason="Llama 3 is significantly better at instruction following and JSON output",
        deprecated_date="2024-04-01",
        size_gb=3.8,
    ),
    "llama3:8b": ModelEntry(
        name="llama3:8b",
        status="deprecated",
        successor="llama3:8b-instruct-q5_K_M",
        reason="Instruct-tuned version produces better structured output",
        deprecated_date="2025-01-01",
        size_gb=4.7,
    ),
    "llama3:8b-instruct-q5_K_M": ModelEntry(
        name="llama3:8b-instruct-q5_K_M",
        status="recommended",
        added_date="2025-01-01",
        size_gb=4.7,
        capabilities=["json", "creative", "evaluation"],
        min_ram_gb=6.0,
    ),
    "llama3:latest": ModelEntry(
        name="llama3:latest",
        status="deprecated",
        successor="llama3:8b-instruct-q5_K_M",
        reason="Explicit quantization tag preferred for reproducibility",
        deprecated_date="2025-06-01",
        size_gb=4.7,
    ),
    "llama3.1:8b": ModelEntry(
        name="llama3.1:8b",
        status="recommended",
        added_date="2025-03-01",
        size_gb=4.9,
        capabilities=["json", "creative", "evaluation", "long_context"],
        min_ram_gb=6.0,
    ),
    # === Mistral family ===
    "mistral:7b": ModelEntry(
        name="mistral:7b",
        status="deprecated",
        successor="mistral:7b-instruct-q5_K_M",
        reason="Instruct-tuned version is better for pipeline tasks",
        deprecated_date="2025-01-01",
        size_gb=4.1,
    ),
    "mistral:latest": ModelEntry(
        name="mistral:latest",
        status="deprecated",
        successor="mistral:7b-instruct-q5_K_M",
        reason="Explicit version tag preferred for reproducibility",
        deprecated_date="2025-06-01",
        size_gb=4.1,
    ),
    "mistral:7b-instruct": ModelEntry(
        name="mistral:7b-instruct",
        status="deprecated",
        successor="mistral:7b-instruct-q5_K_M",
        reason="Quantized version offers same quality with less memory",
        deprecated_date="2025-03-01",
        size_gb=4.1,
    ),
    "mistral:7b-instruct-q5_K_M": ModelEntry(
        name="mistral:7b-instruct-q5_K_M",
        status="recommended",
        added_date="2025-03-01",
        size_gb=4.1,
        capabilities=["json", "technical", "style_reasoning"],
        min_ram_gb=6.0,
    ),
    # === Qwen family ===
    "qwen:7b": ModelEntry(
        name="qwen:7b",
        status="obsolete",
        successor="qwen2:7b-instruct-q5_K_M",
        reason="Qwen 2 has much better JSON output and instruction following",
        deprecated_date="2024-09-01",
        size_gb=4.4,
    ),
    "qwen2:7b": ModelEntry(
        name="qwen2:7b",
        status="deprecated",
        successor="qwen2:7b-instruct-q5_K_M",
        reason="Instruct variant preferred for structured tasks",
        deprecated_date="2025-03-01",
        size_gb=4.4,
    ),
    "qwen2:latest": ModelEntry(
        name="qwen2:latest",
        status="deprecated",
        successor="qwen2:7b-instruct-q5_K_M",
        reason="Explicit version preferred",
        deprecated_date="2025-06-01",
        size_gb=4.4,
    ),
    "qwen2:7b-instruct-q5_K_M": ModelEntry(
        name="qwen2:7b-instruct-q5_K_M",
        status="recommended",
        added_date="2025-03-01",
        size_gb=4.4,
        capabilities=["json", "technical", "multilingual"],
        min_ram_gb=6.0,
    ),
    "qwen2.5:7b": ModelEntry(
        name="qwen2.5:7b",
        status="recommended",
        added_date="2025-06-01",
        size_gb=4.5,
        capabilities=["json", "technical", "multilingual", "code"],
        min_ram_gb=6.0,
    ),
    # === Phi family ===
    "phi3:mini": ModelEntry(
        name="phi3:mini",
        status="recommended",
        added_date="2025-01-01",
        size_gb=2.3,
        capabilities=["json", "technical"],
        min_ram_gb=4.0,
    ),
    "phi3:mini-4k": ModelEntry(
        name="phi3:mini-4k",
        status="deprecated",
        successor="phi3:mini",
        reason="Default phi3:mini now supports adequate context",
        deprecated_date="2025-04-01",
        size_gb=2.3,
    ),
    # === Obsolete/legacy ===
    "codellama:7b": ModelEntry(
        name="codellama:7b",
        status="obsolete",
        successor="llama3:8b-instruct-q5_K_M",
        reason="Llama 3 instruct handles code and creative tasks better",
        deprecated_date="2024-06-01",
        size_gb=3.8,
    ),
    "vicuna:7b": ModelEntry(
        name="vicuna:7b",
        status="obsolete",
        successor="llama3:8b-instruct-q5_K_M",
        reason="Outdated model family, replaced by Llama 3",
        deprecated_date="2024-04-01",
        size_gb=3.8,
    ),
    "neural-chat:7b": ModelEntry(
        name="neural-chat:7b",
        status="obsolete",
        successor="mistral:7b-instruct-q5_K_M",
        reason="Intel model discontinued, Mistral is superior",
        deprecated_date="2024-09-01",
        size_gb=4.1,
    ),
}


# Upgrade path: maps deprecated/obsolete models to their recommended replacement chain
def get_upgrade_path(model_name: str) -> list[str]:
    """
    Get the full upgrade chain for a model.
    Returns list of models from current -> final recommended successor.
    """
    path = [model_name]
    visited = {model_name}
    current = model_name

    while current in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[current]
        if entry.status == "recommended" and current != model_name:
            break
        if entry.successor and entry.successor not in visited:
            path.append(entry.successor)
            visited.add(entry.successor)
            current = entry.successor
        else:
            break

    return path


def get_recommended_replacement(model_name: str) -> Optional[str]:
    """Get the final recommended model that replaces this one."""
    path = get_upgrade_path(model_name)
    if len(path) > 1:
        final = path[-1]
        entry = MODEL_REGISTRY.get(final)
        if entry and entry.status == "recommended":
            return final
    return None


STATE_FILE = Path.home() / ".reel-factory" / "model_state.json"


@dataclass
class ModelState:
    """Persisted state about installed models and refresh history."""
    installed_models: dict = field(default_factory=dict)  # {model_name: install_timestamp}
    last_refresh: Optional[str] = None
    refresh_count: int = 0
    removed_models: list = field(default_factory=list)  # history of removals

    @classmethod
    def load(cls) -> "ModelState":
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return cls(
                    installed_models=data.get("installed_models", {}),
                    last_refresh=data.get("last_refresh"),
                    refresh_count=data.get("refresh_count", 0),
                    removed_models=data.get("removed_models", []),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return cls()

    def save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "installed_models": self.installed_models,
            "last_refresh": self.last_refresh,
            "refresh_count": self.refresh_count,
            "removed_models": self.removed_models,
        }, indent=2))

    def record_install(self, model_name: str):
        self.installed_models[model_name] = datetime.now(timezone.utc).isoformat()

    def record_removal(self, model_name: str, reason: str):
        if model_name in self.installed_models:
            del self.installed_models[model_name]
        self.removed_models.append({
            "model": model_name,
            "reason": reason,
            "removed_at": datetime.now(timezone.utc).isoformat(),
        })


@dataclass
class RefreshAction:
    """A single action to take during model refresh."""
    action: str  # "upgrade" | "remove" | "pull"
    model: str
    reason: str
    replacement: Optional[str] = None


@dataclass
class RefreshPlan:
    """Plan of actions for a model refresh cycle."""
    actions: list = field(default_factory=list)
    models_to_pull: list = field(default_factory=list)
    models_to_remove: list = field(default_factory=list)
    already_current: list = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.models_to_pull or self.models_to_remove)


class ModelRefreshManager:
    """
    Manages model lifecycle: detects deprecated models, plans upgrades,
    executes refresh (pull new, remove old).

    Called at the start of each pipeline run to ensure models are current.
    """

    def __init__(self, ollama_provider):
        self.ollama = ollama_provider
        self.state = ModelState.load()

    async def plan_refresh(self, installed_models: list[str]) -> RefreshPlan:
        """
        Analyze installed models and create a refresh plan.

        Returns a plan with:
        - Models to upgrade (deprecated -> recommended replacement)
        - Models to remove (obsolete, no longer useful)
        - Models to pull (new recommendations not yet installed)
        """
        plan = RefreshPlan()

        for model in installed_models:
            entry = MODEL_REGISTRY.get(model)

            if entry is None:
                # Unknown model - not in registry, leave as-is
                plan.already_current.append(model)
                continue

            if entry.status == "recommended":
                plan.already_current.append(model)
                continue

            if entry.status == "deprecated":
                replacement = get_recommended_replacement(model)
                if replacement and replacement not in installed_models:
                    plan.actions.append(RefreshAction(
                        action="upgrade",
                        model=model,
                        reason=entry.reason,
                        replacement=replacement,
                    ))
                    plan.models_to_pull.append(replacement)
                    plan.models_to_remove.append(model)
                elif replacement and replacement in installed_models:
                    # Replacement already installed, just remove deprecated
                    plan.actions.append(RefreshAction(
                        action="remove",
                        model=model,
                        reason=f"Deprecated; replacement {replacement} already installed",
                    ))
                    plan.models_to_remove.append(model)
                else:
                    plan.already_current.append(model)

            elif entry.status == "obsolete":
                replacement = get_recommended_replacement(model)
                plan.actions.append(RefreshAction(
                    action="remove",
                    model=model,
                    reason=entry.reason,
                    replacement=replacement,
                ))
                plan.models_to_remove.append(model)
                if replacement and replacement not in installed_models:
                    plan.models_to_pull.append(replacement)

        # Also check if any STAGE_LOCAL_MODELS are missing
        from src.llm.ollama_provider import STAGE_LOCAL_MODELS
        for stage, needed_model in STAGE_LOCAL_MODELS.items():
            if needed_model not in installed_models and needed_model not in plan.models_to_pull:
                plan.actions.append(RefreshAction(
                    action="pull",
                    model=needed_model,
                    reason=f"Required for stage: {stage}",
                ))
                plan.models_to_pull.append(needed_model)

        # Deduplicate
        plan.models_to_pull = list(dict.fromkeys(plan.models_to_pull))
        plan.models_to_remove = list(dict.fromkeys(plan.models_to_remove))

        return plan

    async def execute_refresh(self, plan: RefreshPlan, remove_deprecated: bool = True) -> dict:
        """
        Execute the refresh plan: pull new models, optionally remove old ones.

        Args:
            plan: The refresh plan from plan_refresh()
            remove_deprecated: If True, delete deprecated/obsolete models from disk

        Returns:
            Summary dict with results of each action
        """
        results = {
            "pulled": [],
            "removed": [],
            "failed": [],
            "skipped": [],
        }

        # Pull new/upgraded models first
        for model in plan.models_to_pull:
            success = await self.ollama.pull_model(model)
            if success:
                results["pulled"].append(model)
                self.state.record_install(model)
            else:
                results["failed"].append(model)

        # Remove deprecated/obsolete models (only if replacement was pulled successfully)
        if remove_deprecated:
            for model in plan.models_to_remove:
                # Find what replaces it
                action = next((a for a in plan.actions if a.model == model), None)
                replacement = action.replacement if action else None

                # Only remove if replacement is confirmed available
                if replacement and replacement not in results["pulled"]:
                    # Check if replacement was already installed
                    available = await self.ollama.list_models()
                    if replacement not in available:
                        results["skipped"].append(model)
                        continue

                success = await self._delete_model(model)
                if success:
                    results["removed"].append(model)
                    reason = action.reason if action else "Obsolete"
                    self.state.record_removal(model, reason)
                else:
                    results["skipped"].append(model)

        # Update state
        self.state.last_refresh = datetime.now(timezone.utc).isoformat()
        self.state.refresh_count += 1
        self.state.save()

        return results

    async def auto_refresh(self) -> dict:
        """
        Full automatic refresh cycle: detect, plan, execute.
        Called at the start of each pipeline run.

        Returns summary of actions taken.
        """
        if not await self.ollama.is_available():
            return {"status": "skipped", "reason": "Ollama not available"}

        # Force fresh model list
        self.ollama._available_models = None
        installed = await self.ollama.list_models()

        if not installed:
            return {"status": "skipped", "reason": "No models installed"}

        plan = await self.plan_refresh(installed)

        if not plan.has_changes:
            return {"status": "current", "models": plan.already_current}

        results = await self.execute_refresh(plan)
        results["status"] = "refreshed"
        return results

    async def _delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama."""
        try:
            httpx = _get_httpx()
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(
                    f"{self.ollama.base_url}/api/delete",
                    json={"name": model_name},
                )
                return resp.status_code == 200
        except Exception:
            return False
