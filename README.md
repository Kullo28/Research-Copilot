# Research Copilot

An agentic Retrieval-Augmented Generation (RAG) system that uses multiple specialized agents to answer research questions by decomposing them into sub-questions and retrieving relevant information from academic PDFs.

## Overview

Research Copilot implements a multi-agent architecture for intelligent document retrieval and synthesis. It breaks down complex research queries into manageable sub-questions, retrieves relevant passages from a corpus of academic papers, and synthesizes comprehensive answers with iterative refinement.

## Architecture

The system uses a graph-based agent orchestration with the following components:

### Agents
- **Planner**: Decomposes user queries into 3-5 focused sub-questions
- **Retriever**: Searches the vector store for relevant passages based on sub-questions
- **Writer**: Generates draft answers from retrieved passages
- **Reflector**: Evaluates answer quality and determines if revision is needed
- **Server**: Manages agent communication and state

### Key Components
- **graph.py**: Orchestrates the agent workflow using LangGraph
- **memory.py**: Defines the shared state structure (`AgentState`)
- **tools/vector_search.py**: Vector database search functionality using FAISS
- **ui.py**: User interface components

## Project Structure

```
research-copilot/
├── agents/
│   ├── planner.py       # Query decomposition agent
│   ├── retriever.py     # Document retrieval agent
│   ├── writer.py        # Answer generation agent
│   ├── reflector.py     # Quality evaluation agent
│   └── server.py        # Agent coordination
├── tools/
│   └── vector_search.py # Vector database operations
├── data/                # Academic PDFs and research documents
├── vectorstore/         # FAISS vector index
├── main.py             # Entry point
├── graph.py            # Workflow orchestration
├── memory.py           # State management
├── ui.py               # User interface
└── Reasearch_Copilot.ipynb  # Jupyter notebook with examples
```

## Features

- **Multi-agent Architecture**: Specialized agents for planning, retrieval, writing, and reflection
- **Vector Search**: FAISS-based semantic search over academic documents
- **Iterative Refinement**: Automatic evaluation and revision of generated answers
- **State Management**: Shared state across all agents for context preservation
- **Jupyter Integration**: Interactive notebook for experimentation

## Requirements

- Python 3.8+
- OpenAI API (local or remote)
- FAISS for vector search
- LangGraph for agent orchestration
- Ollama (for local LLM support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kullo28/Research-Copilot.git
cd Research-Copilot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1
LOCAL_OPENAI_API_KEY=ollama
LOCAL_MODEL_NAME=llama3.1
```

## Usage

### Command Line
```bash
python main.py
```
Then enter your research question when prompted.

### Jupyter Notebook
```bash
jupyter notebook Reasearch_Copilot.ipynb
```

## Example

```python
from main import run_query

result = run_query("How do multi-agent systems improve RAG reliability?")

print("Sub-questions:", result['sub_questions'])
print("Answer:", result['final_answer'])
```

## How It Works

1. **Planning**: The Planner agent decomposes the research question into focused sub-questions
2. **Retrieval**: The Retriever searches the vector store for passages matching each sub-question
3. **Writing**: The Writer generates a comprehensive answer from retrieved passages
4. **Reflection**: The Reflector evaluates the answer quality and determines if revision is needed
5. **Iteration**: If needed, the cycle repeats with refined questions/answers

## Data

The `data/` directory contains academic PDFs including:
- AI Agents vs Agentic AI
- Building Agentic AI Systems
- ReAct Framework
- Attention Is All You Need

## Contributing

Feel free to open issues and pull requests for improvements!

## License

This project is open source and available under the MIT License.
