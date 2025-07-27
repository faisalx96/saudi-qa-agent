#!/usr/bin/env python3
"""
Interactive Saudi Arabia Q&A Agent

Run this script to ask questions to the Saudi Arabia agent directly.

Usage:
    python run_agent.py                    # Interactive mode
    python run_agent.py "Your question"    # Single question mode
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from agent folder
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config before other modules
from config import config
from agent.saudi_arabia_agent import run_saudi_agent_sync
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
import json


console = Console()


def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing = config.validate_required(required_vars)
    
    if missing:
        console.print(f"[red]❌ Missing environment variables: {', '.join(missing)}[/red]")
        console.print("\nPlease set:")
        for var in missing:
            console.print(f"export {var}='your_key_here'")
        return False
    return True


def display_result(result: dict):
    """Display the agent result in a nice format"""
    console.print("\n" + "="*80)
    
    # Question and classification
    console.print(f"[bold cyan]Question:[/bold cyan] {result['question']}")
    
    is_saudi = result.get('is_saudi_question', False)
    if is_saudi:
        console.print(f"[bold green]✓ Identified as Saudi Arabia question[/bold green]")
    else:
        console.print(f"[bold yellow]✗ Not a Saudi Arabia question[/bold yellow]")
    
    # Show step execution
    console.print("\n[bold]Steps Executed:[/bold]")
    steps_table = Table(show_header=True, header_style="bold magenta")
    steps_table.add_column("Step", style="cyan")
    steps_table.add_column("Status", style="green")
    steps_table.add_column("Details")
    
    for step_name, step_data in result.get('step_outputs', {}).items():
        output = step_data.get('output', 'No output')
        if isinstance(output, bool):
            output = "✓" if output else "✗"
        elif len(str(output)) > 50:
            output = str(output)[:50] + "..."
        steps_table.add_row(step_name.capitalize(), "Completed", str(output))
    
    console.print(steps_table)
    
    # Final answer
    console.print("\n[bold]Answer:[/bold]")
    answer_panel = Panel(
        result.get('final_answer', 'No answer generated'),
        style="green" if is_saudi else "yellow",
        title="Agent Response"
    )
    console.print(answer_panel)
    
    # Search results summary (if available)
    if result.get('search_results') and len(result['search_results']) > 10:
        console.print("\n[dim]Search performed and found relevant information[/dim]")


def run_single_question(question: str):
    """Run the agent with a single question"""
    console.print(f"\n[bold]Processing your question...[/bold]")
    
    try:
        result = run_saudi_agent_sync(question)
        display_result(result)
        return result
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


def interactive_mode():
    """Run the agent in interactive mode"""
    console.print(Panel.fit(
        "[bold cyan]Saudi Arabia Q&A Agent[/bold cyan]\n"
        "Ask questions about Saudi Arabia!\n"
        "Type 'exit' or 'quit' to stop.",
        title="Welcome"
    ))
    
    console.print("\n[dim]Examples:[/dim]")
    console.print("  • What is the capital of Saudi Arabia?")
    console.print("  • Tell me about Vision 2030")
    console.print("  • Who is the Crown Prince of Saudi Arabia?")
    console.print("  • What is the weather in Riyadh?")
    console.print()
    
    while True:
        try:
            question = console.input("[bold cyan]Your question:[/bold cyan] ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            
            if not question:
                console.print("[yellow]Please enter a question[/yellow]")
                continue
            
            run_single_question(question)
            console.print("\n" + "-"*80 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")


def main():
    """Main entry point"""
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check if question provided as argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_single_question(question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()