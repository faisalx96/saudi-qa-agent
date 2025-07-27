#!/usr/bin/env python3
"""
Simple runner for Saudi Arabia Q&A Agent

Usage:
    python simple_run.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from agent folder
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config
from config import config
from agent.saudi_arabia_agent import run_saudi_agent_sync


def main():
    # Check environment
    missing_vars = config.validate_required(["OPENAI_API_KEY", "TAVILY_API_KEY"])
    if missing_vars:
        for var in missing_vars:
            print(f"❌ Please set {var} environment variable")
        return
    
    print("=== Saudi Arabia Q&A Agent ===")
    print("Ask questions about Saudi Arabia!")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not question:
            continue
        
        print("\nProcessing...")
        
        try:
            # Run the agent
            result = run_saudi_agent_sync(question)
            
            # Display results
            print(f"\nIs Saudi Question: {result['is_saudi_question']}")
            print(f"\nAnswer:\n{result['final_answer']}")
            print("\n" + "-"*60 + "\n")
            
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()