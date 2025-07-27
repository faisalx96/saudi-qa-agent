"""
Setup Langfuse datasets for Saudi Arabia Q&A Agent evaluation

This script creates the evaluation datasets in Langfuse for comprehensive agent testing.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from langfuse import Langfuse
from evaluation_datasets import create_langfuse_datasets
from datetime import datetime
import json


def setup_langfuse_client():
    """Initialize Langfuse client with environment variables"""
    client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST
    )
    return client


def create_dataset_in_langfuse(client: Langfuse, dataset_name: str, dataset_info: dict):
    """Create a single dataset in Langfuse with its items"""
    
    print(f"Creating dataset: {dataset_name}")
    
    # Create the dataset
    try:
        dataset = client.create_dataset(
            name=dataset_name,
            description=dataset_info["description"],
            metadata={
                "created_date": datetime.now().isoformat(),
                "metrics": dataset_info["metrics"],
                "item_count": len(dataset_info["data"]),
                "purpose": "saudi_arabia_agent_evaluation"
            }
        )
        print(f"✓ Dataset '{dataset_name}' created successfully")
    except Exception as e:
        print(f"⚠ Dataset '{dataset_name}' might already exist: {e}")
        # Try to get existing dataset
        try:
            dataset = client.get_dataset(dataset_name)
            print(f"✓ Using existing dataset '{dataset_name}'")
        except:
            print(f"✗ Failed to create or retrieve dataset '{dataset_name}'")
            return False
    
    # Add items to the dataset
    success_count = 0
    for i, item in enumerate(dataset_info["data"]):
        try:
            client.create_dataset_item(
                dataset_name=dataset_name,
                input=item["input"],
                expected_output=item["expected_output"],
                metadata=item.get("metadata", {})
            )
            success_count += 1
        except Exception as e:
            print(f"⚠ Failed to add item {i+1}: {e}")
    
    print(f"✓ Added {success_count}/{len(dataset_info['data'])} items to '{dataset_name}'")
    return True


def main():
    """Main function to setup all datasets in Langfuse"""
    
    print("=== Setting up Saudi Arabia Agent Evaluation Datasets in Langfuse ===\n")
    
    # Check environment variables
    required_env_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing_vars = config.validate_required(required_env_vars)
    
    if missing_vars:
        print(f"✗ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the script.")
        return False
    
    # Initialize Langfuse client
    try:
        client = setup_langfuse_client()
        print("✓ Langfuse client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Langfuse client: {e}")
        return False
    
    # Get datasets configuration
    datasets = create_langfuse_datasets()
    
    # Create each dataset
    created_datasets = []
    for dataset_name, dataset_info in datasets.items():
        success = create_dataset_in_langfuse(client, dataset_name, dataset_info)
        if success:
            created_datasets.append(dataset_name)
        print()  # Add spacing
    
    # Summary
    print("=== Setup Summary ===")
    print(f"Successfully created {len(created_datasets)}/{len(datasets)} datasets:")
    for dataset_name in created_datasets:
        print(f"  ✓ {dataset_name}")
    
    if len(created_datasets) == len(datasets):
        print("\n🎉 All datasets created successfully!")
        print("You can now run evaluations using these datasets.")
    else:
        failed_count = len(datasets) - len(created_datasets)
        print(f"\n⚠ {failed_count} datasets failed to create. Please check the errors above.")
    
    return len(created_datasets) == len(datasets)


def verify_datasets():
    """Verify that all datasets were created correctly in Langfuse"""
    
    print("=== Verifying Datasets in Langfuse ===\n")
    
    try:
        client = setup_langfuse_client()
    except Exception as e:
        print(f"✗ Failed to connect to Langfuse: {e}")
        return False
    
    datasets = create_langfuse_datasets()
    verification_results = {}
    
    for dataset_name in datasets.keys():
        try:
            dataset = client.get_dataset(dataset_name)
            item_count = len(list(dataset.items))
            expected_count = len(datasets[dataset_name]["data"])
            
            verification_results[dataset_name] = {
                "exists": True,
                "item_count": item_count,
                "expected_count": expected_count,
                "complete": item_count == expected_count
            }
            
            status = "✓" if item_count == expected_count else "⚠"
            print(f"{status} {dataset_name}: {item_count}/{expected_count} items")
            
        except Exception as e:
            verification_results[dataset_name] = {
                "exists": False,
                "error": str(e)
            }
            print(f"✗ {dataset_name}: Not found or error - {e}")
    
    # Summary
    all_complete = all(
        result.get("complete", False) 
        for result in verification_results.values()
    )
    
    if all_complete:
        print("\n✓ All datasets verified successfully!")
    else:
        print("\n⚠ Some datasets are incomplete or missing.")
    
    return verification_results


def print_dataset_info():
    """Print detailed information about the datasets"""
    
    datasets = create_langfuse_datasets()
    
    print("=== Dataset Information ===\n")
    
    for dataset_name, dataset_info in datasets.items():
        print(f"Dataset: {dataset_name}")
        print(f"Description: {dataset_info['description']}")
        print(f"Items: {len(dataset_info['data'])}")
        print(f"Primary Metrics: {', '.join(dataset_info['metrics']['primary_metrics'])}")
        
        # Show sample items
        print("Sample items:")
        for i, item in enumerate(dataset_info['data'][:2]):  # Show first 2 items
            print(f"  {i+1}. Input: {item['input']}")
            print(f"     Expected: {item['expected_output']}")
            print(f"     Category: {item['metadata'].get('category', 'N/A')}")
        
        if len(dataset_info['data']) > 2:
            print(f"  ... and {len(dataset_info['data']) - 2} more items")
        
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "verify":
            verify_datasets()
        elif command == "info":
            print_dataset_info()
        else:
            print("Usage: python setup_langfuse_datasets.py [verify|info]")
    else:
        # Default: create datasets
        main()