"""
Complete Demo: Saudi Arabia Q&A Agent Evaluation

This script demonstrates the complete workflow:
1. Setup datasets in Langfuse
2. Test the agent functionality
3. Run comprehensive evaluations
4. Generate results report

Usage:
    python demo_saudi_agent_evaluation.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import config
from config import config

from setup_langfuse_datasets import main as setup_datasets, verify_datasets
from evaluate_saudi_agent import quick_test, run_comprehensive_evaluation
from agent.saudi_arabia_agent import run_saudi_agent_sync


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)


def check_prerequisites():
    """Check if all prerequisites are met"""
    
    print_header("CHECKING PREREQUISITES")
    
    # Check environment variables
    required_vars = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY", 
        "OPENAI_API_KEY",
        "TAVILY_API_KEY"
    ]
    
    missing_vars = config.validate_required(required_vars)
    
    if missing_vars:
        print(f"❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables and run again:")
        for var in missing_vars:
            print(f"export {var}='your_key_here'")
        return False
    
    print("✅ All required environment variables are set")
    
    # Check Python dependencies (basic check)
    try:
        import langfuse
        import openai
        import tavily
        import langgraph
        print("✅ All required Python packages are available")
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        print("Please install missing packages:")
        print("pip install langfuse openai tavily-python langgraph")
        return False
    
    return True


def demo_agent_functionality():
    """Demonstrate the agent's functionality with example questions"""
    
    print_header("AGENT FUNCTIONALITY DEMO")
    
    demo_questions = [
        {
            "question": "What is the capital of Saudi Arabia?",
            "expected": "Should identify as Saudi question and provide Riyadh as answer"
        },
        {
            "question": "Who is the current Crown Prince of Saudi Arabia?",
            "expected": "Should identify as Saudi question and search for current information"
        },
        {
            "question": "What is Vision 2030?",
            "expected": "Should identify as Saudi question and explain the economic reform program"
        },
        {
            "question": "What is the weather in London?",
            "expected": "Should identify as NON-Saudi question and politely decline"
        },
        {
            "question": "Tell me about machine learning",
            "expected": "Should identify as NON-Saudi question and politely decline"
        }
    ]
    
    print(f"Testing agent with {len(demo_questions)} sample questions...\n")
    
    for i, test_case in enumerate(demo_questions, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        
        print(f"🔍 Test {i}: {question}")
        print(f"📝 Expected: {expected}")
        
        try:
            result = run_saudi_agent_sync(question)
            
            print(f"✅ Is Saudi Question: {result.get('is_saudi_question', 'Unknown')}")
            print(f"🔍 Search Results: {'Found' if result.get('search_results') else 'None/Skipped'}")
            
            answer = result.get('final_answer', 'No answer')
            print(f"💬 Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
            
            # Show step execution
            steps = list(result.get('step_outputs', {}).keys())
            print(f"🔧 Steps Executed: {' → '.join(steps)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)
    
    print("✅ Agent functionality demo completed")


def setup_evaluation_infrastructure():
    """Setup the evaluation datasets in Langfuse"""
    
    print_header("SETTING UP EVALUATION INFRASTRUCTURE")
    
    print("📋 Creating evaluation datasets in Langfuse...")
    
    # Setup datasets
    success = setup_datasets()
    
    if success:
        print("✅ Datasets created successfully")
        
        # Verify datasets
        print("\n🔍 Verifying datasets...")
        verification_results = verify_datasets()
        
        if all(result.get("complete", False) for result in verification_results.values()):
            print("✅ All datasets verified successfully")
            return True
        else:
            print("⚠️ Some datasets may be incomplete")
            return False
    else:
        print("❌ Failed to create datasets")
        return False


def run_evaluations():
    """Run the comprehensive evaluation suite"""
    
    print_header("RUNNING COMPREHENSIVE EVALUATIONS")
    
    print("📊 This will run evaluations for:")
    print("   1. Question verification accuracy")
    print("   2. Web search quality and relevance") 
    print("   3. Answer generation quality")
    print("   4. End-to-end agent performance")
    print()
    
    success = run_comprehensive_evaluation()
    
    if success:
        print("\n🎉 All evaluations completed successfully!")
        print("📊 Check your Langfuse dashboard for detailed results:")
        print("   - Individual metric scores")
        print("   - Execution traces")
        print("   - Performance analytics")
        return True
    else:
        print("\n⚠️ Some evaluations encountered issues")
        print("📋 Check the detailed output above for specific errors")
        return False


def generate_final_report():
    """Generate a final summary report"""
    
    print_header("FINAL SUMMARY REPORT")
    
    print("🎯 SAUDI ARABIA Q&A AGENT EVALUATION COMPLETED")
    print()
    print("📋 What was evaluated:")
    print("   ✅ Question verification (identifies Saudi Arabia questions)")
    print("   ✅ Web search functionality (retrieves relevant information)")  
    print("   ✅ Answer generation (provides accurate, helpful responses)")
    print("   ✅ End-to-end performance (complete workflow)")
    print()
    print("📊 Evaluation datasets created:")
    print("   • saudi-qa-verification-v1 (18 test cases)")
    print("   • saudi-qa-search-quality-v1 (4 test cases)")
    print("   • saudi-qa-answer-quality-v1 (3 test cases)")
    print()
    print("🔧 Custom metrics implemented:")
    print("   • Exact match verification")
    print("   • Search relevance and keyword coverage")
    print("   • Factual accuracy and completeness")
    print("   • Overall task success rate")
    print()
    print("🎉 NEXT STEPS:")
    print("   1. Review detailed results in your Langfuse dashboard")
    print("   2. Analyze any failed test cases")
    print("   3. Iterate on agent improvements") 
    print("   4. Re-run evaluations to measure progress")
    print()
    print("💡 Framework Benefits Demonstrated:")
    print("   • Simple 3-line evaluation setup")
    print("   • Automatic Langfuse integration")
    print("   • Comprehensive metrics and tracing")
    print("   • Production-ready error handling")


def main():
    """Main demo workflow"""
    
    print("🎬 SAUDI ARABIA Q&A AGENT - COMPLETE EVALUATION DEMO")
    print("This demo showcases our LLM Evaluation Framework with a real LangGraph agent")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please resolve the issues above and try again.")
        return False
    
    # Step 2: Demo agent functionality  
    demo_agent_functionality()
    
    # Step 3: Setup evaluation infrastructure
    if not setup_evaluation_infrastructure():
        print("\n⚠️ Could not setup evaluation infrastructure completely.")
        print("You may still proceed, but some evaluations might fail.")
        
        user_input = input("\nContinue anyway? (y/N): ").lower()
        if user_input != 'y':
            print("Demo cancelled.")
            return False
    
    # Step 4: Run evaluations
    evaluation_success = run_evaluations()
    
    # Step 5: Generate final report
    generate_final_report()
    
    # Return overall success
    return evaluation_success


if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️ Demo completed with some issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)