"""
Complete evaluation of the Saudi Arabia Q&A Agent using our LLM Evaluation Framework

This script demonstrates the full integration of:
1. LangGraph agent (saudi_arabia_agent.py)
2. Evaluation datasets (evaluation_datasets.py) 
3. Custom metrics (saudi_agent_metrics.py)
4. Our evaluation framework (llm_eval)
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import config
from config import config

from langfuse import Langfuse, observe

langfuse = Langfuse(
            public_key="pk-lf-c542f0f6-77fb-4704-b114-006fa90f5c0c",
            secret_key="sk-lf-219a7622-1acd-4e3b-8fba-d31f4e0dfe87",
            host="https://cloud.langfuse.com"
        )

# Note: This requires the original llm_eval framework
# For standalone usage, you may need to implement a simple evaluator
try:
    from llm_eval import Evaluator
except ImportError:
    print("Warning: llm_eval framework not found. Install or implement simple evaluator.")
    Evaluator = None

# Import agent AFTER Langfuse is initialized
from agent.saudi_arabia_agent import run_saudi_agent
from saudi_agent_metrics import (
    VERIFICATION_METRICS,
    SEARCH_METRICS,
    ANSWER_METRICS,
    END_TO_END_METRICS,
    get_metrics_for_dataset
)


def setup_environment():
    """Setup required environment variables and check dependencies"""

    required_vars = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY"
    ]

    missing_vars = config.validate_required(required_vars)

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these variables:")
        print("export LANGFUSE_PUBLIC_KEY='your_key'")
        print("export LANGFUSE_SECRET_KEY='your_key'")
        print("export OPENAI_API_KEY='your_key'")
        print("export TAVILY_API_KEY='your_key'")
        return False

    print("✅ All required environment variables are set")
    return True


def evaluate_verification_node():
    """Evaluate the question verification node independently"""

    print("\n=== Evaluating Question Verification Node ===")

    # Create a wrapper that extracts only verification results
    async def verification_task(input_data):
        question = input_data.get("question", "")
        result = await run_saudi_agent(question)

        # Return only verification-relevant data
        return {
            "question": question,
            "is_saudi_question": result.get("is_saudi_question", False),
            "step_outputs": {
                "verify": result.get("step_outputs", {}).get("verify", {})
            }
        }

    # Run evaluation with clear run name
    run_name = config.get_evaluation_run_name("verification", "saudi-qa-verification-v1")
    print(f"🏃 Starting evaluation run: {run_name}")

    evaluator = Evaluator(
        client=langfuse,
        task=verification_task,
        dataset="saudi-qa-verification-v2",
        metrics=list(VERIFICATION_METRICS.values()),
        config={
            "max_concurrency": 5,
            "timeout": 60.0,
            "run_name": run_name,
            "run_metadata": {
                "model": config.OPENAI_MODEL,
                "temperature": config.MODEL_TEMPERATURE,
                "evaluation_type": "verification",
                "dataset": "saudi-qa-verification-v1"
            },
            "langfuse_public_key": config.LANGFUSE_PUBLIC_KEY,
            "langfuse_secret_key": config.LANGFUSE_SECRET_KEY,
            "langfuse_host": config.LANGFUSE_HOST
        }
    )

    try:
        results = evaluator.run()
        print("✅ Verification node evaluation completed")
        return results
    except Exception as e:
        print(f"❌ Verification evaluation failed: {e}")
        return None


def evaluate_search_node():
    """Evaluate the web search node independently"""

    print("\n=== Evaluating Web Search Node ===")

    async def search_task(input_data):
        question = input_data.get("question", "")
        result = await run_saudi_agent(question)

        return {
            "question": question,
            "is_saudi_question": result.get("is_saudi_question", False),
            "search_results": result.get("search_results", ""),
            "step_outputs": {
                "search": result.get("step_outputs", {}).get("search", {})
            }
        }

    run_name = config.get_evaluation_run_name("search", "saudi-qa-search-quality-v1")
    print(f"🏃 Starting evaluation run: {run_name}")

    evaluator = Evaluator(
        task=search_task,
        dataset="saudi-qa-search-quality-v1",
        metrics=list(SEARCH_METRICS.values()),
        config={
            "max_concurrency": 5,
            "timeout": 60.0,
            "run_name": run_name,
            "run_metadata": {
                "model": config.OPENAI_MODEL,
                "temperature": config.MODEL_TEMPERATURE,
                "evaluation_type": "search",
                "dataset": "saudi-qa-search-quality-v1"
            },
            "langfuse_public_key": config.LANGFUSE_PUBLIC_KEY,
            "langfuse_secret_key": config.LANGFUSE_SECRET_KEY,
            "langfuse_host": config.LANGFUSE_HOST
        }
    )

    try:
        results = evaluator.run()
        print("✅ Search node evaluation completed")
        return results
    except Exception as e:
        print(f"❌ Search evaluation failed: {e}")
        return None


def evaluate_answer_node():
    """Evaluate the answer generation node independently"""

    print("\n=== Evaluating Answer Generation Node ===")

    async def answer_task(input_data):
        question = input_data.get("question", "")
        result = await run_saudi_agent(question)

        return {
            "question": question,
            "final_answer": result.get("final_answer", ""),
            "is_saudi_question": result.get("is_saudi_question", False),
            "step_outputs": {
                "answer": result.get("step_outputs", {}).get("answer", {})
            }
        }

    run_name = config.get_evaluation_run_name("answer", "saudi-qa-answer-quality-v1")
    print(f"🏃 Starting evaluation run: {run_name}")

    evaluator = Evaluator(
        task=answer_task,
        dataset="saudi-qa-answer-quality-v1",
        metrics=list(ANSWER_METRICS.values()),
        config={
            "max_concurrency": 5,
            "timeout": 60.0,
            "run_name": run_name,
            "run_metadata": {
                "model": config.OPENAI_MODEL,
                "temperature": config.MODEL_TEMPERATURE,
                "evaluation_type": "answer",
                "dataset": "saudi-qa-answer-quality-v1"
            },
            "langfuse_public_key": config.LANGFUSE_PUBLIC_KEY,
            "langfuse_secret_key": config.LANGFUSE_SECRET_KEY,
            "langfuse_host": config.LANGFUSE_HOST
        }
    )

    try:
        results = evaluator.run()
        print("✅ Answer node evaluation completed")
        return results
    except Exception as e:
        print(f"❌ Answer evaluation failed: {e}")
        return None


def evaluate_end_to_end():
    """Evaluate the complete agent end-to-end"""

    print("\n=== Evaluating Complete Agent End-to-End ===")

    # Use the agent directly as the task
    async def complete_agent_task(input_data):
        question = input_data.get("question", "")
        return await run_saudi_agent(question)

    # For end-to-end, we can use any of the datasets or create a combined one
    # Here we'll use the verification dataset as it has the most comprehensive test cases
    run_name = config.get_evaluation_run_name("end_to_end", "saudi-qa-verification-v1")
    print(f"🏃 Starting evaluation run: {run_name}")

    evaluator = Evaluator(
        task=complete_agent_task,
        dataset="saudi-qa-verification-v1",
        metrics=list(END_TO_END_METRICS.values()),
        config={
            "max_concurrency": 3,  # Lower for end-to-end tests
            "timeout": 120.0,      # Longer timeout for full agent runs
            "run_name": run_name,
            "run_metadata": {
                "model": config.OPENAI_MODEL,
                "temperature": config.MODEL_TEMPERATURE,
                "evaluation_type": "end_to_end",
                "dataset": "saudi-qa-verification-v1"
            },
            "langfuse_public_key": config.LANGFUSE_PUBLIC_KEY,
            "langfuse_secret_key": config.LANGFUSE_SECRET_KEY,
            "langfuse_host": config.LANGFUSE_HOST
        }
    )

    try:
        results = evaluator.run()
        print("✅ End-to-end evaluation completed")
        return results
    except Exception as e:
        print(f"❌ End-to-end evaluation failed: {e}")
        return None


def run_comprehensive_evaluation():
    """Run all evaluations and provide a comprehensive report"""

    print("🚀 Starting Comprehensive Saudi Arabia Agent Evaluation")
    print("=" * 60)

    # Check setup
    if not setup_environment():
        return False

    # Store all results
    evaluation_results = {}

    # Run individual node evaluations
    evaluation_results["verification"] = evaluate_verification_node()
    evaluation_results["search"] = evaluate_search_node()
    evaluation_results["answer"] = evaluate_answer_node()
    evaluation_results["end_to_end"] = evaluate_end_to_end()

    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY REPORT")
    print("=" * 60)

    for eval_type, results in evaluation_results.items():
        if results:
            print(f"\n{eval_type.upper()} EVALUATION:")
            try:
                summary = results.summary()
                print(f"  ✅ Completed successfully")
                print(f"  📈 Results: {summary}")
            except Exception as e:
                print(f"  ⚠️  Results available but summary failed: {e}")
        else:
            print(f"\n{eval_type.upper()} EVALUATION:")
            print(f"  ❌ Failed to complete")

    # Overall assessment
    successful_evals = sum(1 for result in evaluation_results.values() if result is not None)
    total_evals = len(evaluation_results)

    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Successfully completed: {successful_evals}/{total_evals} evaluations")

    if successful_evals == total_evals:
        print("   🎉 All evaluations completed successfully!")
        print("   📊 Check your Langfuse dashboard for detailed results and traces")
    else:
        print("   ⚠️  Some evaluations failed. Check error messages above.")

    return successful_evals == total_evals


def quick_test():
    """Quick test to verify the agent works before full evaluation"""

    print("🧪 Running Quick Agent Test...")

    test_cases = [
        "What is the capital of Saudi Arabia?",  # Should be handled
        "What is the weather in London?",        # Should be rejected
        "Tell me about Vision 2030",            # Should be handled
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\nTest {i}: {question}")
        try:
            result = asyncio.run(run_saudi_agent(question))
            print(f"  ✅ Is Saudi Question: {result.get('is_saudi_question')}")
            print(f"  💬 Answer: {result.get('final_answer', '')[:100]}...")
            print(f"  🔧 Steps: {list(result.get('step_outputs', {}).keys())}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print("\n✅ Quick test completed")


def main():
    """Main function with command line interface"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            quick_test()
        elif command == "verify":
            evaluate_verification_node()
        elif command == "search":
            evaluate_search_node()
        elif command == "answer":
            evaluate_answer_node()
        elif command == "e2e":
            evaluate_end_to_end()
        elif command == "full":
            run_comprehensive_evaluation()
        else:
            print("Usage: python evaluate_saudi_agent.py [test|verify|search|answer|e2e|full]")
            print("  test   - Quick functionality test")
            print("  verify - Evaluate verification node only")
            print("  search - Evaluate search node only") 
            print("  answer - Evaluate answer node only")
            print("  e2e    - Evaluate end-to-end only")
            print("  full   - Run all evaluations (default)")
    else:
        # Default: run comprehensive evaluation
        run_comprehensive_evaluation()


if __name__ == "__main__":
    main()