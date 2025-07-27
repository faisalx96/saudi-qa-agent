"""
Custom metrics for evaluating the Saudi Arabia Q&A Agent

This module implements specific metrics for each node of the LangGraph agent:
1. Question verification metrics
2. Search quality metrics  
3. Answer quality metrics
4. End-to-end evaluation metrics
"""

import re
from typing import Dict, Any, List, Optional, Union
from difflib import SequenceMatcher
import asyncio
from datetime import datetime


# ============================================================================
# NODE 1: QUESTION VERIFICATION METRICS
# ============================================================================

def exact_match_verification(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Exact match for verification node - checks if is_saudi_question matches expected
    """
    actual = output.get("is_saudi_question", None)
    expected_val = expected.get("is_saudi_question", None)
    
    if actual is None or expected_val is None:
        return 0.0
    
    return 1.0 if actual == expected_val else 0.0


def verification_confidence(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Measures confidence in verification decision based on step output reasoning
    """
    step_output = output.get("step_outputs", {}).get("verify", {})
    reasoning = step_output.get("reasoning", "")

    # Simple confidence scoring based on reasoning quality
    confidence_indicators = [
        "verified", "analyzed", "determined", "confirmed",
        "Saudi Arabia", "clear", "explicit"
    ]

    matches = sum(1 for indicator in confidence_indicators if indicator.lower() in reasoning.lower())
    return min(1.0, matches / 3.0)  # Normalize to 0-1


# ============================================================================
# NODE 2: SEARCH QUALITY METRICS  
# ============================================================================

def search_relevance_score(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Evaluates relevance of search results to the Saudi Arabia question
    """
    search_results = output.get("search_results", "")
    expected_keywords = expected.get("expected_keywords", [])
    
    if not search_results or not expected_keywords:
        return 0.0
    
    # Count how many expected keywords appear in search results
    found_keywords = 0
    for keyword in expected_keywords:
        if keyword.lower() in search_results.lower():
            found_keywords += 1
    
    return found_keywords / len(expected_keywords)


def keyword_coverage(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Measures percentage of expected keywords found in search results
    """
    return search_relevance_score(output, expected)  # Same implementation


def source_authority_score(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Evaluates the authority/quality of search result sources
    """
    search_results = output.get("search_results", "")
    
    # Define authoritative domains
    authoritative_sources = [
        "wikipedia.org", "britannica.com", "bbc.com", "reuters.com",
        "cnn.com", "gov.sa", "spa.gov.sa", "aljazeera.com"
    ]
    
    authority_score = 0.0
    for source in authoritative_sources:
        if source in search_results.lower():
            authority_score += 0.2  # Each authoritative source adds 0.2
    
    return min(1.0, authority_score)


def search_execution_success(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Checks if search was executed successfully when it should have been
    """
    should_search = expected.get("should_search", True)
    step_output = output.get("step_outputs", {}).get("search", {})
    search_output = step_output.get("output", "")
    
    if not should_search:
        # Should have been skipped
        return 1.0 if "skipped" in search_output.lower() else 0.0
    else:
        # Should have executed successfully
        return 0.0 if "error" in search_output.lower() else 1.0


# ============================================================================
# NODE 3: ANSWER QUALITY METRICS
# ============================================================================

def factual_accuracy(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Evaluates factual accuracy of the final answer
    """
    answer = output.get("final_answer", "")
    should_contain = expected.get("answer_should_contain", [])
    should_not_contain = expected.get("answer_should_not_contain", [])
    
    if not answer:
        return 0.0
    
    score = 0.0
    
    # Check required content
    if should_contain:
        found_required = sum(1 for item in should_contain if item.lower() in answer.lower())
        score += (found_required / len(should_contain)) * 0.7
    
    # Check prohibited content (negative scoring)
    if should_not_contain:
        found_prohibited = sum(1 for item in should_not_contain if item.lower() in answer.lower())
        penalty = (found_prohibited / len(should_not_contain)) * 0.3
        score = max(0.0, score - penalty)
    
    return min(1.0, score)


def answer_completeness(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Measures how complete the answer is based on expected completeness score
    """
    answer = output.get("final_answer", "")
    expected_completeness = expected.get("completeness_score", 0.5)
    
    if not answer:
        return 0.0
    
    # Simple completeness heuristics
    answer_length = len(answer.split())
    
    # Score based on length and content richness
    if answer_length < 10:
        actual_completeness = 0.2
    elif answer_length < 50:
        actual_completeness = 0.5
    elif answer_length < 150:
        actual_completeness = 0.8
    else:
        actual_completeness = 1.0
    
    # Compare with expected
    difference = abs(actual_completeness - expected_completeness)
    return max(0.0, 1.0 - difference)


def answer_relevance(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Measures relevance of answer to the original question
    """
    answer = output.get("final_answer", "")
    question = output.get("question", "")
    
    if not answer or not question:
        return 0.0
    
    # Extract key terms from question
    question_terms = set(question.lower().split())
    answer_terms = set(answer.lower().split())
    
    # Calculate overlap
    common_terms = question_terms.intersection(answer_terms)
    if len(question_terms) == 0:
        return 0.0
    
    relevance = len(common_terms) / len(question_terms)
    return min(1.0, relevance * 2)  # Boost the score


def rejection_appropriateness(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    For non-Saudi questions, checks if the agent appropriately rejected the question
    """
    answer = output.get("final_answer", "")
    should_reject = expected.get("rejection_response", False)
    
    if not should_reject:
        return 1.0  # Not applicable
    
    rejection_indicators = [
        "only answer questions about saudi arabia",
        "saudi arabia",
        "cannot answer",
        "not about saudi arabia"
    ]
    
    found_rejection = any(indicator in answer.lower() for indicator in rejection_indicators)
    return 1.0 if found_rejection else 0.0


# ============================================================================
# END-TO-END EVALUATION METRICS
# ============================================================================

def overall_task_success(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Measures overall success of the complete agent workflow
    """
    # Check if all expected steps were completed
    step_outputs = output.get("step_outputs", {})
    expected_steps = ["verify", "search", "answer"]
    
    completed_steps = sum(1 for step in expected_steps if step in step_outputs)
    step_completion = completed_steps / len(expected_steps)
    
    # Check final answer quality
    answer = output.get("final_answer", "")
    has_answer = 1.0 if answer and len(answer) > 10 else 0.0
    
    # Combine scores
    return (step_completion * 0.6) + (has_answer * 0.4)


def response_time_score(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Evaluates response time performance (assumes timing data in metadata)
    """
    # This would need actual timing data from the agent execution
    # For now, return a placeholder score
    return 1.0  # Placeholder - would implement with real timing data


def error_handling_quality(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Evaluates how well the agent handles errors and edge cases
    """
    step_outputs = output.get("step_outputs", {})
    
    # Check for graceful error handling in step outputs
    error_handling_score = 0.0
    
    for step, step_data in step_outputs.items():
        step_output = step_data.get("output", "")
        reasoning = step_data.get("reasoning", "")
        
        # Look for error indicators and how they're handled
        if "error" in step_output.lower():
            if "gracefully" in reasoning.lower() or "handled" in reasoning.lower():
                error_handling_score += 0.5
        else:
            error_handling_score += 0.3
    
    return min(1.0, error_handling_score)


# ============================================================================
# METRIC REGISTRY FOR EACH EVALUATION TYPE
# ============================================================================

VERIFICATION_METRICS = {
    "exact_match": exact_match_verification,
    "verification_confidence": verification_confidence,
}

SEARCH_METRICS = {
    "search_relevance": search_relevance_score,
    "keyword_coverage": keyword_coverage,
    "source_authority": source_authority_score,
    "search_execution": search_execution_success,
}

ANSWER_METRICS = {
    "factual_accuracy": factual_accuracy,
    "completeness": answer_completeness,
    "relevance": answer_relevance,
    "rejection_appropriateness": rejection_appropriateness,
}

END_TO_END_METRICS = {
    "overall_success": overall_task_success,
    "response_time": response_time_score,
    "error_handling": error_handling_quality,
}

# Combined registry for easy access
ALL_SAUDI_AGENT_METRICS = {
    **VERIFICATION_METRICS,
    **SEARCH_METRICS,
    **ANSWER_METRICS,
    **END_TO_END_METRICS,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_metrics_for_dataset(dataset_name: str) -> Dict[str, callable]:
    """
    Returns appropriate metrics for a given dataset
    """
    if "verification" in dataset_name:
        return VERIFICATION_METRICS
    elif "search" in dataset_name:
        return SEARCH_METRICS
    elif "answer" in dataset_name:
        return ANSWER_METRICS
    else:
        return END_TO_END_METRICS


def calculate_composite_score(scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate weighted composite score from multiple metrics
    """
    if not scores:
        return 0.0
    
    if weights is None:
        # Equal weights
        return sum(scores.values()) / len(scores)
    
    total_weight = sum(weights.get(metric, 1.0) for metric in scores.keys())
    weighted_sum = sum(score * weights.get(metric, 1.0) for metric, score in scores.items())
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def validate_agent_output(output: Dict[str, Any]) -> List[str]:
    """
    Validate that agent output has expected structure
    """
    errors = []
    
    required_fields = ["question", "is_saudi_question", "final_answer", "step_outputs"]
    for field in required_fields:
        if field not in output:
            errors.append(f"Missing required field: {field}")
    
    if "step_outputs" in output:
        expected_steps = ["verify", "search", "answer"]
        actual_steps = list(output["step_outputs"].keys())
        
        for step in expected_steps:
            if step not in actual_steps:
                errors.append(f"Missing step output: {step}")
    
    return errors


if __name__ == "__main__":
    # Test the metrics with sample data
    sample_output = {
        "question": "What is the capital of Saudi Arabia?",
        "is_saudi_question": True,
        "search_results": "Riyadh is the capital of Saudi Arabia. Located in the center of the country.",
        "final_answer": "The capital of Saudi Arabia is Riyadh.",
        "step_outputs": {
            "verify": {"output": True, "reasoning": "Question clearly asks about Saudi Arabia"},
            "search": {"output": "Found 3 results", "reasoning": "Web search completed successfully"}, 
            "answer": {"output": "Generated answer", "reasoning": "Answer based on search results"}
        }
    }
    
    sample_expected = {
        "is_saudi_question": True,
        "answer_should_contain": ["Riyadh", "capital", "Saudi Arabia"],
        "answer_should_not_contain": ["uncertain"],
        "completeness_score": 0.9
    }
    
    print("=== Testing Saudi Agent Metrics ===\n")
    
    # Test verification metrics
    print("Verification Metrics:")
    for name, metric in VERIFICATION_METRICS.items():
        score = metric(sample_output, sample_expected)
        print(f"  {name}: {score:.2f}")
    
    print("\nAnswer Quality Metrics:")
    for name, metric in ANSWER_METRICS.items():
        score = metric(sample_output, sample_expected)
        print(f"  {name}: {score:.2f}")
    
    print(f"\nComposite Score: {calculate_composite_score({name: metric(sample_output, sample_expected) for name, metric in ANSWER_METRICS.items()}):.2f}")