#!/usr/bin/env python3
"""
Setup script for Saudi Arabia Q&A Agent
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is supported"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        return False
    print("✅ Python version check passed")
    return True

def install_dependencies():
    """Install project dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_env_file():
    """Check if .env file exists and help create it"""
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file found")
        return True
    
    if example_file.exists():
        print("⚠️  No .env file found. Creating from .env.example...")
        try:
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env file created from example")
            print("📝 Please edit .env file and add your API keys:")
            print("   - OPENAI_API_KEY")
            print("   - TAVILY_API_KEY")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("❌ .env.example file not found")
        return False

def test_imports():
    """Test if core modules can be imported"""
    print("🧪 Testing imports...")
    try:
        sys.path.insert(0, str(Path.cwd()))
        from agent.saudi_arabia_agent import run_saudi_agent_sync
        print("✅ Core agent import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Saudi Arabia Q&A Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check environment file
    if not check_env_file():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Run the agent:")
    print("   python run.py                    # Interactive mode")
    print("   python run.py simple            # Simple mode")
    print('   python run.py "Your question"   # Single question')
    print("\n📚 Documentation: README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)