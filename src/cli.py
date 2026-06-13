"""CLI interface for Reel Factory."""
import asyncio
import threading
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

from src.config.settings import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator, InteractionMode, StageAction
from src.pipeline.state import StageResult

console = Console()


def user_intervention_callback(stage_name: str, result: StageResult, timeout: float) -> StageAction:
    """
    Interactive CLI callback for user decisions at stage boundaries.

    Displays a countdown timer. If no input within timeout, auto-approves.
    """
    console.print()
    console.print(Panel(
        f"[bold green]Stage Complete:[/] {stage_name}\n"
        f"[dim]Confidence: {result.confidence:.0%}[/]\n"
        f"[yellow]Auto-continuing in {timeout:.0f}s if no input...[/]",
        title="Pipeline Checkpoint",
    ))

    if isinstance(result.output, dict):
        console.print_json(data=result.output)
    elif result.output:
        console.print(str(result.output)[:500])

    console.print(
        "[dim]Options: [bold]approve[/] (default) | modify | skip | override | abort[/]"
    )

    user_input = {"value": "approve"}
    input_received = threading.Event()

    def get_input():
        try:
            choice = Prompt.ask("Action", default="approve")
            user_input["value"] = choice
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            input_received.set()

    thread = threading.Thread(target=get_input, daemon=True)
    thread.start()
    input_received.wait(timeout=timeout)

    if not input_received.is_set():
        console.print("[yellow]Timeout reached - auto-approving[/]")
        return StageAction.APPROVE

    action_map = {
        "approve": StageAction.APPROVE,
        "modify": StageAction.MODIFY,
        "skip": StageAction.SKIP,
        "override": StageAction.OVERRIDE_MODEL,
        "abort": StageAction.ABORT,
    }
    return action_map.get(user_input["value"], StageAction.APPROVE)


def display_token_summary(summary: dict):
    """Display token usage summary in a rich table."""
    console.print()
    console.print(Panel("[bold]Token Usage Summary[/]", style="blue"))

    totals = summary.get("totals", {})
    console.print(f"  Total tokens: [bold]{totals.get('total_tokens', 0):,}[/]")
    console.print(f"  Input: {totals.get('input_tokens', 0):,} | Output: {totals.get('output_tokens', 0):,}")
    console.print(f"  LLM calls: {totals.get('call_count', 0)}")
    console.print(f"  Total latency: {totals.get('total_latency_ms', 0):.0f}ms")

    per_stage = summary.get("per_stage", {})
    if per_stage:
        console.print()
        table = Table(title="Per-Stage Breakdown")
        table.add_column("Stage", style="bold")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Tokens", justify="right")
        table.add_column("Calls", justify="right")
        table.add_column("Latency", justify="right")

        for stage, data in per_stage.items():
            table.add_row(
                stage,
                data.get('model', 'unknown'),
                data.get('provider', 'unknown'),
                f"{data['total_tokens']:,}",
                str(data['call_count']),
                f"{data['total_latency_ms']:.0f}ms",
            )
        console.print(table)


@click.group()
def cli():
    """Reel Factory - AI-powered video reel generation."""
    pass


@cli.command()
@click.option("--script", "-s", required=True, help="The script or idea for the reel")
@click.option("--genre", "-g", default=None, help="Genre (comedy, drama, horror, action)")
@click.option("--config", "-c", default="config/pipeline.yaml", help="Config file path")
@click.option("--mode", "-m", default="prompt", type=click.Choice(["prompt", "auto", "minimal"]))
@click.option("--rendering", "-r", default=None, type=click.Choice(["animated", "talking_head", "stylized"]))
@click.option("--llm-mode", default=None, type=click.Choice(["local_first", "cloud_first", "local_only", "cloud_only"]),
              help="Override LLM provider mode")
def create(script: str, genre: Optional[str], config: str, mode: str, rendering: Optional[str], llm_mode: Optional[str]):
    """Create a new video reel from a script."""
    import os
    console.print(Panel("[bold]Reel Factory[/] - Creating your reel", style="blue"))

    pipeline_config = PipelineConfig.from_yaml(config)
    if rendering:
        pipeline_config.character_rendering_mode = rendering
    if llm_mode:
        os.environ["REEL_LLM_MODE"] = llm_mode
        pipeline_config.llm_mode = llm_mode

    interaction = InteractionMode(mode)
    orchestrator = PipelineOrchestrator(pipeline_config, interaction)

    if interaction == InteractionMode.PROMPT:
        orchestrator.set_user_callback(user_intervention_callback)

    console.print(f"  Script: [italic]{script}[/]")
    console.print(f"  Genre: [bold]{genre or 'auto-detect'}[/]")
    console.print(f"  LLM Mode: [cyan]{pipeline_config.llm_mode}[/]")
    console.print(f"  Interaction: {mode} | Rendering: {rendering or pipeline_config.character_rendering_mode}")
    console.print(f"  Max iterations: {pipeline_config.max_iterations}")
    console.print()

    state = asyncio.run(orchestrator.run(script, genre))

    if state.final_output:
        console.print(Panel(f"[bold green]Reel created![/]\nOutput: {state.final_output}", style="green"))
    else:
        console.print("[yellow]Pipeline did not produce a final output.[/]")

    if state.token_summary:
        display_token_summary(state.token_summary)


@cli.group()
def models():
    """Manage local LLM models for offline operation."""
    pass


@models.command("status")
def models_status():
    """Show status of local models (which are available/missing)."""
    from src.llm.ollama_provider import OllamaProvider, STAGE_LOCAL_MODELS

    async def check():
        provider = OllamaProvider()
        available = await provider.is_available()
        if not available:
            console.print("[red]Ollama is not running![/]")
            console.print("Install from: https://ollama.com")
            console.print("Start with: [bold]ollama serve[/]")
            return

        local_models = await provider.list_models()
        console.print(f"[green]Ollama is running[/] ({len(local_models)} models available)")
        console.print()

        table = Table(title="Stage Model Assignments")
        table.add_column("Stage", style="bold")
        table.add_column("Required Model", style="cyan")
        table.add_column("Status")
        table.add_column("Resolved To")

        for stage, model in STAGE_LOCAL_MODELS.items():
            resolved = await provider.resolve_model(stage)
            if resolved:
                status = "[green]Ready[/]"
            else:
                status = "[red]Missing[/]"
                resolved = "-"
            table.add_row(stage, model, status, resolved or "-")

        console.print(table)

    asyncio.run(check())


@models.command("pull")
@click.option("--stage", "-s", default=None, help="Pull model for specific stage only")
@click.option("--all", "pull_all", is_flag=True, help="Pull all recommended models")
def models_pull(stage: Optional[str], pull_all: bool):
    """Download recommended local models via Ollama."""
    from src.llm.ollama_provider import OllamaProvider, STAGE_LOCAL_MODELS

    async def pull():
        provider = OllamaProvider()
        available = await provider.is_available()
        if not available:
            console.print("[red]Ollama is not running! Start with: ollama serve[/]")
            return

        if stage:
            if stage not in STAGE_LOCAL_MODELS:
                console.print(f"[red]Unknown stage: {stage}[/]")
                console.print(f"Available: {list(STAGE_LOCAL_MODELS.keys())}")
                return
            stages = [stage]
        else:
            stages = list(STAGE_LOCAL_MODELS.keys())

        # Deduplicate models
        models_to_pull = {}
        for s in stages:
            model = STAGE_LOCAL_MODELS[s]
            if model not in models_to_pull:
                models_to_pull[model] = []
            models_to_pull[model].append(s)

        console.print(f"Pulling {len(models_to_pull)} unique models...")
        console.print()

        for model, used_by in models_to_pull.items():
            console.print(f"  Pulling [cyan]{model}[/] (used by: {', '.join(used_by)})...")
            success = await provider.pull_model(model)
            if success:
                console.print(f"    [green]Done[/]")
            else:
                console.print(f"    [red]Failed[/]")

        console.print()
        console.print("[green]Model pull complete![/]")

    asyncio.run(pull())


@models.command("list")
def models_list():
    """List all locally available Ollama models."""
    from src.llm.ollama_provider import OllamaProvider

    async def list_them():
        provider = OllamaProvider()
        available = await provider.is_available()
        if not available:
            console.print("[red]Ollama is not running![/]")
            return

        models = await provider.list_models()
        if not models:
            console.print("[yellow]No models installed. Run: reel-factory models pull[/]")
            return

        table = Table(title="Installed Ollama Models")
        table.add_column("Model", style="cyan")
        for m in sorted(models):
            table.add_row(m)
        console.print(table)

    asyncio.run(list_them())




@models.command("refresh")
@click.option("--dry-run", is_flag=True, help="Show what would change without doing it")
@click.option("--keep-deprecated", is_flag=True, help="Pull upgrades but keep old models")
def models_refresh(dry_run: bool, keep_deprecated: bool):
    """Refresh models: upgrade deprecated, remove obsolete, pull missing."""
    from src.llm.ollama_provider import OllamaProvider
    from src.llm.model_registry import ModelRefreshManager, MODEL_REGISTRY

    async def refresh():
        provider = OllamaProvider()
        available = await provider.is_available()
        if not available:
            console.print("[red]Ollama is not running! Start with: ollama serve[/]")
            return

        # Force fresh list
        provider._available_models = None
        installed = await provider.list_models()

        manager = ModelRefreshManager(provider)
        plan = await manager.plan_refresh(installed)

        if not plan.has_changes:
            console.print("[green]All models are current![/]")
            console.print(f"  Installed & up-to-date: {len(plan.already_current)}")
            return

        # Show the plan
        console.print(Panel("[bold]Model Refresh Plan[/]", style="blue"))

        if plan.models_to_pull:
            console.print("\n[bold green]Models to pull (new/upgraded):[/]")
            for action in plan.actions:
                if action.action in ("upgrade", "pull"):
                    if action.replacement:
                        console.print(f"  [cyan]{action.replacement}[/] (replaces {action.model})")
                        console.print(f"    Reason: {action.reason}")
                    else:
                        console.print(f"  [cyan]{action.model}[/]")
                        console.print(f"    Reason: {action.reason}")

        if plan.models_to_remove:
            console.print("\n[bold red]Models to remove (deprecated/obsolete):[/]")
            for action in plan.actions:
                if action.action == "remove" or (action.action == "upgrade" and action.model in plan.models_to_remove):
                    entry = MODEL_REGISTRY.get(action.model)
                    status = entry.status if entry else "unknown"
                    console.print(f"  [red]{action.model}[/] ({status})")
                    console.print(f"    Reason: {action.reason}")

        if dry_run:
            console.print("\n[yellow]Dry run - no changes made[/]")
            return

        # Execute
        console.print("\n[bold]Executing refresh...[/]")
        results = await manager.execute_refresh(plan, remove_deprecated=not keep_deprecated)

        console.print()
        if results.get("pulled"):
            console.print(f"[green]Pulled:[/] {', '.join(results['pulled'])}")
        if results.get("removed"):
            console.print(f"[red]Removed:[/] {', '.join(results['removed'])}")
        if results.get("failed"):
            console.print(f"[yellow]Failed:[/] {', '.join(results['failed'])}")
        if results.get("skipped"):
            console.print(f"[dim]Skipped:[/] {', '.join(results['skipped'])}")
        console.print("\n[green]Refresh complete![/]")

    asyncio.run(refresh())


@models.command("registry")
def models_registry():
    """Show the model registry with lifecycle status."""
    from src.llm.model_registry import MODEL_REGISTRY, get_recommended_replacement

    table = Table(title="Model Registry")
    table.add_column("Model", style="bold")
    table.add_column("Status")
    table.add_column("Successor", style="cyan")
    table.add_column("Reason")

    for name, entry in sorted(MODEL_REGISTRY.items(), key=lambda x: (x[1].status, x[0])):
        if entry.status == "recommended":
            status_str = "[green]recommended[/]"
        elif entry.status == "deprecated":
            status_str = "[yellow]deprecated[/]"
        else:
            status_str = "[red]obsolete[/]"

        successor = entry.successor or "-"
        reason = entry.reason[:60] + "..." if len(entry.reason) > 60 else entry.reason

        table.add_row(name, status_str, successor, reason or "-")

    console.print(table)


@cli.command()
def stages():
    """Show pipeline stages and their default configurations."""
    from src.pipeline.stage_defaults import STAGE_DEFAULTS, STAGE_BOUNDARIES
    from src.llm.ollama_provider import STAGE_LOCAL_MODELS

    table = Table(title="Pipeline Stages")
    table.add_column("#", style="cyan")
    table.add_column("Stage", style="bold")
    table.add_column("Local Model", style="green")
    table.add_column("Cloud Fallback", style="yellow")
    table.add_column("Timeout", justify="right")
    table.add_column("Auto-approve", justify="center")
    table.add_column("Retries", justify="right")

    cloud_models = {
        "screenplay": "gpt-4o",
        "character_design": "gpt-4o",
        "genre_style": "gpt-4o",
        "visual_rendering": "runway-gen3",
        "audio_music": "suno-v4",
        "effects_filters": "gpt-4o",
        "assembly": "gpt-4o",
        "review": "gpt-4o",
    }

    for i, (name, defaults) in enumerate(STAGE_DEFAULTS.items(), 1):
        local = STAGE_LOCAL_MODELS.get(name, "-")
        cloud = cloud_models.get(name, "-")
        table.add_row(
            str(i),
            name,
            local,
            cloud,
            f"{defaults.intervention_timeout_seconds:.0f}s",
            "Yes" if defaults.auto_approve_on_timeout else "No",
            str(defaults.max_retries),
        )

    console.print(table)


if __name__ == "__main__":
    cli()
