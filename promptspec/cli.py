"""CLI interface for PromptSpec."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box

from .runner import TestRunner, RunResults
from .gateway import LLMGateway

app = typer.Typer(
    name="promptspec",
    help="Unit testing for LLM prompts - Test your prompts locally before deploying",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    spec_file: str = typer.Argument(
        "promptspec.yaml",
        help="Path to the YAML spec file",
    ),
    max_concurrent: int = typer.Option(
        10,
        "--max-concurrent",
        "-c",
        help="Maximum number of concurrent test executions",
        min=1,
        max=50,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output for each test",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        "-j",
        help="Model to use for LLM-as-a-Judge assertions (default: gpt-3.5-turbo)",
    ),
):
    """Run prompt tests from a YAML spec file."""
    spec_path = Path(spec_file)

    if not spec_path.exists():
        console.print(f"[red]Error:[/red] Spec file not found: {spec_file}")
        raise typer.Exit(code=1)

    # Initialize gateway with optional judge model
    gateway = LLMGateway(default_judge_model=judge_model or "gpt-3.5-turbo")

    # Initialize runner
    runner = TestRunner(gateway=gateway, max_concurrent=max_concurrent)

    # Run tests with progress indicator
    console.print(f"[bold blue]Running tests from:[/bold blue] {spec_file}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Executing tests...", total=None)

        async def run_async():
            return await runner.run_spec(str(spec_path))

        import asyncio
        results = asyncio.run(run_async())

        progress.update(task, completed=100)

    # Display results
    _display_results(results, verbose)

    # Exit with appropriate code
    if results.all_passed:
        raise typer.Exit(code=0)
    else:
        raise typer.Exit(code=1)


def _display_results(results: RunResults, verbose: bool = False):
    """Display test results in a formatted table.

    Args:
        results: RunResults to display
        verbose: Whether to show detailed output
    """
    # Create results table
    table = Table(
        title="Prompt Eval Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Test Case", style="cyan", no_wrap=False)
    table.add_column("Status", style="bold", width=10)
    table.add_column("Latency", justify="right", width=12)
    table.add_column("Output Preview", style="dim", width=50)

    # Add rows
    for result in results.results:
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
        latency_str = f"{result.latency_ms:.0f}ms"
        output_preview = result.output[:47] + "..." if len(result.output) > 50 else result.output

        if not result.passed and result.error:
            output_preview = f"[red]{result.error}[/red]"

        table.add_row(
            result.description,
            status,
            latency_str,
            output_preview,
        )

    console.print("\n")
    console.print(table)
    console.print("\n")

    # Show summary
    avg_latency = results.total_latency_ms / results.total if results.total > 0 else 0.0
    summary_text = (
        f"[bold]Total:[/bold] {results.total} | "
        f"[green]Passed:[/green] {results.passed} | "
        f"[red]Failed:[/red] {results.failed} | "
        f"[cyan]Success Rate:[/cyan] {results.success_rate:.1f}% | "
        f"[yellow]Avg Latency:[/yellow] {avg_latency:.0f}ms"
    )

    if results.all_passed:
        panel = Panel(
            summary_text,
            title="[green]✓ All Tests Passed[/green]",
            border_style="green",
        )
    else:
        panel = Panel(
            summary_text,
            title="[red]✗ Some Tests Failed[/red]",
            border_style="red",
        )

    console.print(panel)

    # Show verbose details for failed tests
    if verbose and results.failed > 0:
        console.print("\n[bold red]Failed Test Details:[/bold red]\n")
        for result in results.results:
            if not result.passed:
                console.print(f"[bold]{result.description}[/bold]")
                if result.error:
                    console.print(f"  [red]Error:[/red] {result.error}")
                else:
                    console.print(f"  [yellow]Output:[/yellow] {result.output[:200]}")
                    for assertion in result.assertion_results:
                        if not assertion["passed"]:
                            console.print(
                                f"  [red]✗ {assertion['type']}:[/red] {assertion.get('error', 'Failed')}"
                            )
                console.print()


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"PromptSpec version {__version__}")


if __name__ == "__main__":
    app()

