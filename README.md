# Saudi Arabia Q&A Agent

A sophisticated LangGraph-based agent that answers questions about Saudi Arabia through a structured 3-step workflow.

## 🤖 How It Works

The agent processes questions through three nodes:

1. **🔍 Question Verification** - Determines if the question is about Saudi Arabia
2. **🌐 Web Search** - Searches for relevant, up-to-date information (if applicable)
3. **💬 Answer Generation** - Provides comprehensive answers or politely declines

## 📁 Project Structure

```
saudi-qa-agent/
├── agent/                    # Core agent implementation
│   └── saudi_arabia_agent.py
├── scripts/                  # Interactive runners
│   ├── run_agent.py         # Rich interactive interface
│   └── simple_run.py        # Basic interactive interface
├── evaluation/               # Evaluation framework
│   ├── evaluation_datasets.py
│   ├── saudi_agent_metrics.py
│   ├── setup_langfuse_datasets.py
│   ├── evaluate_saudi_agent.py
│   └── demo_saudi_agent_evaluation.py
├── docs/                     # Documentation
│   └── README.md            # Detailed documentation
├── tests/                    # Unit tests (to be added)
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="your_openai_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
```

### 3. Run the Agent
```bash
# Interactive mode with rich formatting
python scripts/run_agent.py

# Simple interactive mode
python scripts/simple_run.py

# Single question
python scripts/run_agent.py "What is the capital of Saudi Arabia?"
```

## 🔧 Features

- ✅ **Smart Question Detection** - Accurately identifies Saudi Arabia-related questions
- 🌐 **Real-time Web Search** - Fetches current information from authoritative sources
- 🎯 **Contextual Answers** - Provides comprehensive, relevant responses
- 🚫 **Graceful Declining** - Politely handles non-Saudi questions
- 📊 **Comprehensive Evaluation** - Full testing framework with custom metrics
- 🔍 **Langfuse Integration** - Complete observability and tracing

## 💬 Example Questions

**✅ Saudi Arabia Questions (will be answered):**
- "What is the capital of Saudi Arabia?"
- "Who is the Crown Prince of Saudi Arabia?"
- "Tell me about Vision 2030"
- "What are the main cities in KSA?"
- "Describe Saudi Arabian culture"

**❌ Non-Saudi Questions (will be declined):**
- "What is the capital of France?"
- "Tell me about machine learning"
- "What's the weather in London?"

## 📊 Evaluation

The project includes a comprehensive evaluation framework:

```bash
# Run complete evaluation demo
python evaluation/demo_saudi_agent_evaluation.py

# Setup evaluation datasets in Langfuse
python evaluation/setup_langfuse_datasets.py

# Run specific evaluations
python evaluation/evaluate_saudi_agent.py [test|verify|search|answer|e2e|full]
```

## 🛠️ Development

### Architecture

The agent uses LangGraph for workflow management:
- **State Management**: Tracks question, verification result, search results, and final answer
- **Conditional Routing**: Skips search for non-Saudi questions
- **Error Handling**: Graceful degradation on failures
- **Tracing**: Complete observability through Langfuse

### Adding New Features

1. **Extend Agent Logic**: Modify `agent/saudi_arabia_agent.py`
2. **Add Evaluation Cases**: Update `evaluation/evaluation_datasets.py`
3. **Create Custom Metrics**: Add to `evaluation/saudi_agent_metrics.py`
4. **Update Documentation**: Keep docs current

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

MIT License - feel free to use this project for your own applications.

---

**Built with:** LangGraph • OpenAI • Tavily • Langfuse • Rich