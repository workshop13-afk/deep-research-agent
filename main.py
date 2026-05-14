import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import DeepResearchAgent
from prompts import DEFAULT_PROMPT, PROMPT_DESCRIPTIONS, SYSTEM_PROMPTS, VULNERABLE_PROMPTS
from report import save_report

load_dotenv()
console = Console()


def _check_env() -> None:
    missing = [k for k in ("LLM_BASE_URL", "LLM_MODEL") if not os.environ.get(k)]
    if missing:
        console.print(f"[bold red]Missing environment variables:[/bold red] {', '.join(missing)}")
        console.print("Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and fill in your settings.")
        sys.exit(1)


def _print_modes() -> None:
    table = Table(title="Research Modes", show_header=True, header_style="bold cyan")
    table.add_column("Mode", style="cyan")
    table.add_column("Security", justify="center")
    table.add_column("Description")
    for name, desc in PROMPT_DESCRIPTIONS.items():
        label = name + (" [bold green](default)[/bold green]" if name == DEFAULT_PROMPT else "")
        sec = "[bold red]VULN[/bold red]" if name in VULNERABLE_PROMPTS else "[green]OK[/green]"
        table.add_row(label, sec, desc)
    console.print(table)


def _interactive_prompt(default_mode: str) -> tuple[str, str]:
    console.print(
        Panel(
            "[bold]Deep Research Agent[/bold]\nPowered by [cyan]Claude[/cyan] + [green]Tavily Search[/green]",
            border_style="cyan",
        )
    )
    _print_modes()
    console.print()

    query = console.input("[bold cyan]Research query:[/bold cyan] ").strip()
    if not query:
        console.print("[red]No query provided. Exiting.[/red]")
        sys.exit(1)

    mode_input = (
        console.input(
            f"[bold cyan]Mode[/bold cyan] [{'/'.join(SYSTEM_PROMPTS)}] "
            f"(press Enter for '{default_mode}'): "
        ).strip()
        or default_mode
    )
    if mode_input not in SYSTEM_PROMPTS:
        console.print(f"[yellow]Unknown mode '{mode_input}', using '{default_mode}'.[/yellow]")
        mode_input = default_mode

    return query, mode_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Research Agent — internet-powered research reports via Claude + Tavily",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\nExamples:\n"
        "  python main.py 'latest advances in quantum computing'\n"
        "  python main.py --mode tech 'Rust vs Go for systems programming in 2025'\n"
        "  python main.py --mode market 'EV battery supply chain trends'\n"
        "  python main.py --list-modes\n",
    )
    parser.add_argument("query", nargs="?", help="Research topic or question")
    parser.add_argument(
        "--mode", "-m",
        default=DEFAULT_PROMPT,
        choices=list(SYSTEM_PROMPTS),
        metavar="MODE",
        help=f"Research mode — system prompt persona (default: {DEFAULT_PROMPT}). Use --list-modes to see all.",
    )
    parser.add_argument(
        "--output", "-o",
        default="reports",
        help="Directory to save the markdown report (default: reports/)",
    )
    parser.add_argument(
        "--list-modes", "-l",
        action="store_true",
        help="Print available research modes and exit",
    )
    args = parser.parse_args()

    if args.list_modes:
        _print_modes()
        return

    _check_env()

    if args.query:
        query, mode = args.query, args.mode
    else:
        query, mode = _interactive_prompt(args.mode)

    agent = DeepResearchAgent(system_prompt_name=mode)
    result = agent.research(query)

    if not result["report"]:
        console.print("[bold red]The agent did not produce a report. Try a more specific query.[/bold red]")
        sys.exit(1)

    save_report(result, output_dir=args.output)
    console.print(f"\n[bold]Searches performed:[/bold] {result['search_count']}  |  "
                  f"[bold]Sources collected:[/bold] {len(result['sources'])}")


if __name__ == "__main__":
    main()
