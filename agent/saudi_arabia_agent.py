"""
Saudi Arabia Q&A Agent using LangGraph

This agent answers questions about Saudi Arabia through a 3-step process:
1. Verify the question is about Saudi Arabia
2. Search the web for relevant information
3. Generate a comprehensive answer
"""

import os
from typing import Dict, Any, Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json
from tavily import TavilyClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

# Langfuse tracing imports
try:
    from langfuse import Langfuse, observe, get_client
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"Langfuse import error: {e}")
    LANGFUSE_AVAILABLE = False
    # Create dummy decorators if Langfuse is not available
    def observe(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    def get_client():
        return None

class SaudiArabiaAgentState:
    """State for the Saudi Arabia Q&A agent"""
    def __init__(self):
        self.question: str = ""
        self.is_saudi_question: bool = False
        self.search_results: str = ""
        self.final_answer: str = ""
        self.step_outputs: Dict[str, Any] = {}


@observe(name="verify_saudi_question")
def verify_saudi_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Verify if the question is about Saudi Arabia
    Returns: Updated state with is_saudi_question boolean
    """
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=config.MODEL_TEMPERATURE)
    
    verification_prompt = f"""
    Analyze the following question and determine if it is asking about Saudi Arabia.
    
    Question: "{state['question']}"
    
    Consider these criteria:
    - Does it explicitly mention Saudi Arabia, KSA, or Kingdom of Saudi Arabia?
    - Does it ask about Saudi cities (Riyadh, Jeddah, Mecca, Medina, etc.)?
    - Does it ask about Saudi culture, history, economy, politics, or geography?
    - Does it ask about Saudi rulers, royal family, or government?
    - Does it ask about Saudi landmarks, traditions, or events?
    
    Respond with ONLY "true" or "false" - nothing else.
    """
    
    response = llm.invoke([HumanMessage(content=verification_prompt)])
    is_saudi = response.content.strip().lower() == "true"
    
    # Store step output
    step_output = {
        "node": "verify_saudi_question",
        "input": state['question'],
        "output": is_saudi,
        "reasoning": "Verified if question relates to Saudi Arabia"
    }
    
    return {
        **state,
        "is_saudi_question": is_saudi,
        "step_outputs": {**state.get("step_outputs", {}), "verify": step_output}
    }


@observe(name="search_web")
def search_web(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Search the web for information about Saudi Arabia
    Only runs if is_saudi_question is True
    """
    if not state["is_saudi_question"]:
        step_output = {
            "node": "search_web",
            "input": state['question'],
            "output": "Skipped - not a Saudi Arabia question",
            "reasoning": "Question verification failed"
        }
        return {
            **state,
            "search_results": "",
            "step_outputs": {**state.get("step_outputs", {}), "search": step_output}
        }
    
    # Initialize Tavily client for web search
    tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
    
    # Enhance search query with Saudi Arabia context
    search_query = f"Saudi Arabia {state['question']}"
    
    try:
        # Perform web search
        search_response = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=5,
            include_domains=["wikipedia.org", "britannica.com", "bbc.com", "reuters.com"]
        )
        
        # Extract and format search results
        search_content = ""
        for result in search_response.get("results", []):
            search_content += f"Source: {result.get('url', 'Unknown')}\n"
            search_content += f"Title: {result.get('title', 'No title')}\n"
            search_content += f"Content: {result.get('content', 'No content')}\n\n"
        
        step_output = {
            "node": "search_web",
            "input": search_query,
            "output": f"Found {len(search_response.get('results', []))} results",
            "reasoning": f"Web search completed with {len(search_response.get('results', []))} sources"
        }
        
    except Exception as e:
        search_content = f"Search failed: {str(e)}"
        step_output = {
            "node": "search_web",
            "input": search_query,
            "output": f"Search error: {str(e)}",
            "reasoning": "Web search encountered an error"
        }
    
    return {
        **state,
        "search_results": search_content,
        "step_outputs": {**state.get("step_outputs", {}), "search": step_output}
    }


@observe(name="generate_answer")
def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Generate final answer based on search results
    """
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=config.MODEL_TEMPERATURE)
    
    if not state["is_saudi_question"]:
        answer = "I can only answer questions about Saudi Arabia. Please ask a question related to Saudi Arabia's geography, culture, history, economy, or current affairs."
        step_output = {
            "node": "generate_answer",
            "input": "Non-Saudi question",
            "output": answer,
            "reasoning": "Provided standard response for non-Saudi questions"
        }
    else:
        # Generate answer using search results
        answer_prompt = f"""
        You are an expert on Saudi Arabia. Based on the search results below, provide a comprehensive and accurate answer to the user's question.
        
        Question: {state['question']}
        
        Search Results:
        {state['search_results']}
        
        Instructions:
        - Provide a clear, informative answer based on the search results
        - If search results are insufficient, acknowledge limitations
        - Include relevant details about Saudi Arabia
        - Maintain factual accuracy
        - Structure your response clearly
        """
        
        response = llm.invoke([HumanMessage(content=answer_prompt)])
        answer = response.content
        
        step_output = {
            "node": "generate_answer",
            "input": state['question'],
            "output": answer[:100] + "..." if len(answer) > 100 else answer,
            "reasoning": "Generated answer based on search results"
        }
    
    return {
        **state,
        "final_answer": answer,
        "step_outputs": {**state.get("step_outputs", {}), "answer": step_output}
    }


def should_search(state: Dict[str, Any]) -> Literal["search", "answer"]:
    """Conditional edge: decide whether to search or skip to answer"""
    return "search" if state["is_saudi_question"] else "answer"


def create_saudi_arabia_agent():
    """Create and return the Saudi Arabia Q&A agent"""
    
    # Create the graph
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("verify", verify_saudi_question)
    workflow.add_node("search", search_web)
    workflow.add_node("answer", generate_answer)
    
    # Add edges
    workflow.set_entry_point("verify")
    workflow.add_conditional_edges(
        "verify",
        should_search,
        {
            "search": "search",
            "answer": "answer"
        }
    )
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    # Compile the graph
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


async def run_saudi_agent(question: str, trace_name: Optional[str] = None, user_id: Optional[str] = None, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the Saudi Arabia agent with a question
    
    Args:
        question: The question to ask about Saudi Arabia
        trace_name: Optional name for the Langfuse trace
        user_id: Optional user ID for tracing
        session_id: Optional session ID for tracing
        metadata: Optional metadata for the trace
        
    Returns:
        Complete state with all step outputs and final answer
    """
    agent = create_saudi_arabia_agent()
    
    # Initialize state
    initial_state = {
        "question": question,
        "is_saudi_question": False,
        "search_results": "",
        "final_answer": "",
        "step_outputs": {}
    }
    
    # Run the agent
    config = {"configurable": {"thread_id": "saudi_qa_thread"}}
    final_state = await agent.ainvoke(initial_state, config)
    
    return final_state


# Synchronous wrapper for compatibility
def run_saudi_agent_sync(question: str, trace_name: Optional[str] = None, user_id: Optional[str] = None, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Synchronous wrapper for the Saudi Arabia agent with Langfuse tracing"""
    import asyncio
    result = asyncio.run(run_saudi_agent(question, trace_name, user_id, session_id, metadata))
    return result

if __name__ == "__main__":
    # Test the agent
    test_question = "What is the capital of Saudi Arabia?"
    result = run_saudi_agent_sync(test_question)
    
    print("=== Saudi Arabia Agent Results ===")
    print(f"Question: {result['question']}")
    print(f"Is Saudi Question: {result['is_saudi_question']}")
    print(f"Final Answer: {result['final_answer']}")
    print("\nStep-by-step outputs:")
    for step, output in result['step_outputs'].items():
        print(f"{step}: {output}")