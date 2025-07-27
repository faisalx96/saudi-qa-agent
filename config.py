"""
Configuration module for Saudi QA Agent

Centralizes all environment variable loading and provides
default values where appropriate.
"""

import os
from pathlib import Path
from typing import Optional

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, will use system environment variables


class Config:
    """Central configuration class for all environment variables"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # Model Configuration
    # Using gpt-3.5-turbo for higher rate limits and lower cost
    # Can be overridden via environment variable
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    
    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    @classmethod
    def get_evaluation_run_name(cls, evaluation_type: str, dataset_name: str = None) -> str:
        """
        Generate a clear, readable name for evaluation runs
        
        Args:
            evaluation_type: Type of evaluation (verification, search, answer, end-to-end)
            dataset_name: Optional dataset name
            
        Returns:
            Formatted run name with timestamp and context
        """
        from datetime import datetime
        import socket
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = cls.OPENAI_MODEL.replace("-", "_")
        
        # Get current user or hostname for context
        try:
            context = os.getenv("USER", socket.gethostname().split('.')[0])
        except:
            context = "unknown"
        
        base_name = f"SaudiQA_{evaluation_type}_{model_name}_{timestamp}_{context}"
        
        if dataset_name:
            # Clean dataset name
            clean_dataset = dataset_name.replace("saudi-qa-", "").replace("-v1", "").replace("-", "_")
            base_name = f"SaudiQA_{evaluation_type}_{clean_dataset}_{model_name}_{timestamp}_{context}"
            
        return base_name
    
    @classmethod
    def validate_required(cls, required_vars: list[str]) -> list[str]:
        """
        Validate that required environment variables are set
        
        Args:
            required_vars: List of required variable names
            
        Returns:
            List of missing variable names
        """
        missing = []
        for var in required_vars:
            if not getattr(cls, var, None):
                missing.append(var)
        return missing
    
    @classmethod
    def get_all(cls) -> dict:
        """Get all configuration values as a dictionary"""
        return {
            key: value 
            for key, value in cls.__dict__.items() 
            if not key.startswith('_') and not callable(value)
        }


# Create a singleton instance for easy access
config = Config()