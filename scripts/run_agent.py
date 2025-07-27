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
    is_saudi = result.get('is_saudi_question', False)
    
    # Create a nice answer box
    answer_panel = Panel(
        result.get('final_answer', 'No answer generated'),
        style="green" if is_saudi else "yellow",
        title=f"💬 Answer - {'Saudi Arabia Question ✓' if is_saudi else 'Not a Saudi Question ✗'}",
        border_style="green" if is_saudi else "yellow",
        expand=False
    )
    console.print(answer_panel)


def run_single_question(question: str):
    """Run the agent with a single question with real-time progress"""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    import time
    
    console.print("")  # Just add some spacing
    
    # Create progress bars for each step
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=False,  # Keep progress visible
    ) as progress:
        
        # Add tasks
        verify_task = progress.add_task("[cyan]Verifying question type", total=100)
        search_task = progress.add_task("[dim]Searching for information", total=100)
        answer_task = progress.add_task("[dim]Generating answer", total=100)
        
        try:
            # Start verification
            progress.update(verify_task, advance=20)
            time.sleep(0.1)
            
            # Run the agent
            start_time = time.time()
            result = run_saudi_agent_sync(question)
            
            # Complete verification
            is_saudi = result.get('is_saudi_question', False)
            progress.update(verify_task, completed=100, description=f"[green]✓ Verified: {'Saudi Arabia question' if is_saudi else 'Not a Saudi question'}")
            
            if is_saudi:
                # Search progress
                progress.update(search_task, advance=50, description="[yellow]Searching for information...")
                time.sleep(0.2)
                progress.update(search_task, completed=100, description="[green]✓ Search completed")
                
                # Answer generation
                progress.update(answer_task, advance=50, description="[yellow]Generating answer...")
                time.sleep(0.2)
                progress.update(answer_task, completed=100, description="[green]✓ Answer generated")
            else:
                # Skip search for non-Saudi questions
                progress.update(search_task, completed=100, description="[dim]⏭ Search skipped")
                progress.update(answer_task, completed=100, description="[green]✓ Standard response provided")
            
            # Brief pause to show completion
            time.sleep(0.5)
            
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            return None
    
    # Display full results after progress completes
    console.print("")  # Add spacing
    display_result(result)
    return result


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
            question = console.input("\n[bold cyan]Your question:[/bold cyan] ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            
            if not question:
                console.print("[yellow]Please enter a question[/yellow]")
                continue
            
            run_single_question(question)
            
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