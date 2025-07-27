"""
Evaluation datasets for Saudi Arabia Q&A Agent

This module defines comprehensive evaluation datasets for each node of the agent:
1. Question verification dataset
2. Web search quality dataset  
3. Final answer accuracy dataset
"""

from typing import List, Dict, Any
import json

# Dataset 1: Question Verification
# Tests the agent's ability to correctly identify Saudi Arabia-related questions
VERIFICATION_DATASET = [
    # Positive cases - Should return True
    {
        "input": {"question": "What is the capital of Saudi Arabia?"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "geography", "difficulty": "easy", "type": "positive"}
    },
    {
        "input": {"question": "Who is the current Crown Prince of Saudi Arabia?"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "politics", "difficulty": "medium", "type": "positive"}
    },
    {
        "input": {"question": "Tell me about the history of Mecca"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "history_religion", "difficulty": "medium", "type": "positive"}
    },
    {
        "input": {"question": "What is Vision 2030 in Saudi Arabia?"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "economy", "difficulty": "medium", "type": "positive"}
    },
    {
        "input": {"question": "How is the weather in Riyadh?"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "geography", "difficulty": "easy", "type": "positive"}
    },
    {
        "input": {"question": "What are the main industries in KSA?"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "economy", "difficulty": "medium", "type": "positive"}
    },
    {
        "input": {"question": "Describe Saudi Arabian culture and traditions"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "culture", "difficulty": "medium", "type": "positive"}
    },
    {
        "input": {"question": "What is the population of Jeddah?"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "geography", "difficulty": "easy", "type": "positive"}
    },
    
    # Negative cases - Should return False
    {
        "input": {"question": "What is the capital of Egypt?"},
        "expected_output": {"is_saudi_question": False},
        "metadata": {"category": "geography", "difficulty": "easy", "type": "negative"}
    },
    {
        "input": {"question": "Tell me about the weather in London"},
        "expected_output": {"is_saudi_question": False},
        "metadata": {"category": "geography", "difficulty": "easy", "type": "negative"}
    },
    {
        "input": {"question": "Who is the President of the United States?"},
        "expected_output": {"is_saudi_question": False},
        "metadata": {"category": "politics", "difficulty": "easy", "type": "negative"}
    },
    {
        "input": {"question": "What is machine learning?"},
        "expected_output": {"is_saudi_question": False},
        "metadata": {"category": "technology", "difficulty": "easy", "type": "negative"}
    },
    {
        "input": {"question": "How do I cook pasta?"},
        "expected_output": {"is_saudi_question": False},
        "metadata": {"category": "cooking", "difficulty": "easy", "type": "negative"}
    },
    
    # Edge cases - Tricky questions
    {
        "input": {"question": "Compare Saudi Arabia with UAE economy"},
        "expected_output": {"is_saudi_question": True},
        "metadata": {"category": "economy", "difficulty": "hard", "type": "edge_case"}
    },
    {
        "input": {"question": "What are the differences between Sunni and Shia Islam?"},
        "expected_output": {"is_saudi_question": False},
        "metadata": {"category": "religion", "difficulty": "hard", "type": "edge_case"}
    },
    {
        "input": {"question": "Arab world oil production statistics"},
        "expected_output": {"is_saudi_question": False},
        "metadata": {"category": "economy", "difficulty": "hard", "type": "edge_case"}
    }
]

# Dataset 2: Web Search Quality
# Tests the quality and relevance of web search results
SEARCH_QUALITY_DATASET = [
    {
        "input": {
            "question": "What is the capital of Saudi Arabia?",
            "is_saudi_question": True
        },
        "expected_output": {
            "should_search": True,
            "expected_keywords": ["Riyadh", "capital", "Saudi Arabia"],
            "quality_criteria": ["accurate", "recent", "authoritative_source"]
        },
        "metadata": {"category": "geography", "search_complexity": "simple"}
    },
    {
        "input": {
            "question": "What is Vision 2030 Saudi Arabia?",
            "is_saudi_question": True
        },
        "expected_output": {
            "should_search": True,
            "expected_keywords": ["Vision 2030", "economic reform", "Mohammed bin Salman", "diversification"],
            "quality_criteria": ["comprehensive", "recent", "official_sources"]
        },
        "metadata": {"category": "economy", "search_complexity": "complex"}
    },
    {
        "input": {
            "question": "Who is the current king of Saudi Arabia?",
            "is_saudi_question": True
        },
        "expected_output": {
            "should_search": True,
            "expected_keywords": ["King Salman", "Saudi Arabia", "monarch", "royal family"],
            "quality_criteria": ["current", "accurate", "authoritative"]
        },
        "metadata": {"category": "politics", "search_complexity": "simple"}
    },
    {
        "input": {
            "question": "What is the weather in London?",
            "is_saudi_question": False
        },
        "expected_output": {
            "should_search": False,
            "search_results": "skipped",
            "reason": "not_saudi_question"
        },
        "metadata": {"category": "geography", "search_complexity": "skip"}
    }
]

# Dataset 3: Final Answer Quality
# Tests the accuracy, completeness, and helpfulness of final answers
ANSWER_QUALITY_DATASET = [
    {
        "input": {
            "question": "What is the capital of Saudi Arabia?",
            "search_results": "Riyadh is the capital and largest city of Saudi Arabia. Located in the center of the country, it serves as the political and administrative center.",
            "is_saudi_question": True
        },
        "expected_output": {
            "answer_should_contain": ["Riyadh", "capital", "Saudi Arabia"],
            "answer_should_not_contain": ["uncertain", "I don't know"],
            "factual_accuracy": True,
            "completeness_score": 0.9
        },
        "metadata": {"category": "geography", "answer_type": "factual"}
    },
    {
        "input": {
            "question": "Tell me about Vision 2030",
            "search_results": "Vision 2030 is Saudi Arabia's strategic framework to reduce dependence on oil, diversify the economy, and develop public service sectors such as health, education, infrastructure, recreation, and tourism.",
            "is_saudi_question": True
        },
        "expected_output": {
            "answer_should_contain": ["Vision 2030", "economic diversification", "oil dependence", "development"],
            "answer_should_not_contain": ["unclear", "insufficient information"],
            "factual_accuracy": True,
            "completeness_score": 0.8
        },
        "metadata": {"category": "economy", "answer_type": "explanatory"}
    },
    {
        "input": {
            "question": "What is the weather in Paris?",
            "search_results": "",
            "is_saudi_question": False
        },
        "expected_output": {
            "answer_should_contain": ["Saudi Arabia", "only answer questions about Saudi Arabia"],
            "rejection_response": True,
            "helpful_redirect": True
        },
        "metadata": {"category": "rejection", "answer_type": "redirect"}
    }
]

# Comprehensive evaluation metrics for each dataset
EVALUATION_METRICS = {
    "verification": {
        "primary_metrics": [
            "exact_match",          # Binary accuracy for True/False
            "precision",            # True positives / (True positives + False positives)
            "recall",               # True positives / (True positives + False negatives)
            "f1_score",            # Harmonic mean of precision and recall
        ],
        "secondary_metrics": [
            "response_time",        # Time taken for verification
            "confidence_score",     # If available from model
        ]
    },
    "search_quality": {
        "primary_metrics": [
            "search_relevance",     # How relevant are the search results
            "keyword_coverage",     # Percentage of expected keywords found
            "source_authority",     # Quality of sources (Wikipedia, BBC, etc.)
            "information_completeness", # How complete is the information
        ],
        "secondary_metrics": [
            "search_time",          # Time taken for search
            "num_results",          # Number of search results returned
            "error_rate",           # Percentage of failed searches
        ]
    },
    "answer_quality": {
        "primary_metrics": [
            "factual_accuracy",     # Accuracy of factual claims
            "completeness",         # How complete is the answer
            "relevance",            # How relevant to the question
            "helpfulness",          # Overall helpfulness score
        ],
        "secondary_metrics": [
            "answer_length",        # Length appropriateness
            "clarity",              # Clarity of expression
            "response_time",        # Time to generate answer
            "contains_expected",    # Contains expected key information
        ]
    },
    "end_to_end": {
        "primary_metrics": [
            "overall_accuracy",     # End-to-end accuracy
            "user_satisfaction",    # Simulated user satisfaction
            "task_completion",      # Whether task was completed successfully
        ],
        "secondary_metrics": [
            "total_response_time",  # Total time for all steps
            "step_success_rate",    # Percentage of successful steps
            "error_recovery",       # How well errors are handled
        ]
    }
}

def create_langfuse_datasets():
    """
    Create datasets in Langfuse for evaluation
    Returns dictionary with dataset names and their purposes
    """
    return {
        "saudi-qa-verification-v1": {
            "data": VERIFICATION_DATASET,
            "description": "Dataset for testing Saudi Arabia question verification accuracy",
            "metrics": EVALUATION_METRICS["verification"]
        },
        "saudi-qa-search-quality-v1": {
            "data": SEARCH_QUALITY_DATASET,
            "description": "Dataset for evaluating web search quality and relevance",
            "metrics": EVALUATION_METRICS["search_quality"]
        },
        "saudi-qa-answer-quality-v1": {
            "data": ANSWER_QUALITY_DATASET,
            "description": "Dataset for assessing final answer quality and accuracy",
            "metrics": EVALUATION_METRICS["answer_quality"]
        }
    }

def get_evaluation_plan():
    """
    Return comprehensive evaluation plan for the Saudi Arabia agent
    """
    return {
        "agent_components": [
            {
                "component": "Question Verification Node",
                "dataset": "saudi-qa-verification-v1",
                "key_metrics": ["exact_match", "precision", "recall", "f1_score"],
                "success_criteria": {
                    "exact_match": "> 0.90",
                    "f1_score": "> 0.85"
                }
            },
            {
                "component": "Web Search Node", 
                "dataset": "saudi-qa-search-quality-v1",
                "key_metrics": ["search_relevance", "keyword_coverage", "source_authority"],
                "success_criteria": {
                    "search_relevance": "> 0.80",
                    "keyword_coverage": "> 0.70"
                }
            },
            {
                "component": "Answer Generation Node",
                "dataset": "saudi-qa-answer-quality-v1", 
                "key_metrics": ["factual_accuracy", "completeness", "relevance"],
                "success_criteria": {
                    "factual_accuracy": "> 0.85",
                    "completeness": "> 0.75"
                }
            }
        ],
        "evaluation_approach": {
            "individual_nodes": "Test each node in isolation with specific datasets",
            "end_to_end": "Test complete agent flow with integrated scenarios",
            "error_scenarios": "Test error handling and edge cases",
            "performance": "Measure response times and resource usage"
        }
    }

if __name__ == "__main__":
    # Display evaluation plan
    plan = get_evaluation_plan()
    datasets = create_langfuse_datasets()
    
    print("=== Saudi Arabia Agent Evaluation Plan ===\n")
    
    for component in plan["agent_components"]:
        print(f"Component: {component['component']}")
        print(f"Dataset: {component['dataset']}")
        print(f"Key Metrics: {', '.join(component['key_metrics'])}")
        print(f"Success Criteria: {component['success_criteria']}")
        print()
    
    print("=== Dataset Summary ===")
    for name, info in datasets.items():
        print(f"{name}: {len(info['data'])} items - {info['description']}")