# LLM Council Enhanced

> **Forked from [karpathy/llm-council](https://github.com/karpathy/llm-council)**
> 
> This is an enhanced version with multi-turn chat, RAG-powered context retrieval, chain of thought display, and additional AI models.

[![Fork of karpathy/llm-council](https://img.shields.io/badge/fork-karpathy%2Fllm--council-blue)](https://github.com/karpathy/llm-council)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![llmcouncil](header.jpg)

## ✨ What's New in This Fork

| Feature | Original | Enhanced |
|---------|----------|----------|
| Multi-turn chat | ❌ | ✅ |
| RAG context retrieval | ❌ | ✅ (ChromaDB) |
| Chain of thought display | ❌ | ✅ |
| Models supported | 4 | 6 (added Kimi-K2, DeepSeek-v3.2) |
| Follow-up questions | ❌ | ✅ |
| Context-aware responses | ❌ | ✅ (Smart RAG) |

### Key Enhancements

- **💬 Multi-turn Chat Mode**: Ask follow-up questions to the Chairman without re-running the full Council
- **🧠 RAG System**: ChromaDB-powered context retrieval for efficient follow-ups (no extra LLM calls!)
- **💭 Chain of Thought Display**: See the model's reasoning process (collapsible UI)
- **🤖 Additional Models**: Support for `moonshotai/kimi-k2-thinking` and `deepseek/deepseek-v3.2-exp`
- **⚡ Smart Context Management**: Similarity-based retrieval (threshold: 0.2, max tokens: 1000)
- **📊 Structured Metadata**: Council sessions indexed with turn, stage, and model information

## Original Project

This project is based on Andrej Karpathy's [llm-council](https://github.com/karpathy/llm-council), which implements a fascinating multi-LLM deliberation system. The original concept of having LLMs review and rank each other's work before synthesis is entirely from the original project.

### How It Works (Original Concept)

1. **Stage 1: First opinions**. The user query is given to all LLMs individually, and the responses are collected.
2. **Stage 2: Review**. Each individual LLM is given the responses of the other LLMs (anonymized) and asked to rank them.
3. **Stage 3: Final response**. The designated Chairman takes all responses and compiles them into a single final answer.

### What This Fork Adds

After Stage 3, you can now:
- Ask follow-up questions in "Chat Mode"
- The Chairman retrieves only relevant context via RAG (not the full history)
- See the model's reasoning process if the model provides it
- Continue the conversation naturally without re-running the Council

---

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for Python package management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key

Create a `.env` file in the project root (copy from `.env.example`):

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Get your API key at [openrouter.ai](https://openrouter.ai/). Make sure to purchase credits or enable automatic top-up.

### 3. Configure Models (Optional)

Edit `backend/config.py` to customize the council:

```python
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
    "moonshotai/kimi-k2-thinking",  # New!
    "deepseek/deepseek-v3.2-exp",   # New!
]

CHAIRMAN_MODEL = "google/gemini-3-pro-preview"
```

### 4. (Optional) Adjust RAG Settings

Edit `backend/rag.py` to tune retrieval:

```python
RAG_SIM_THRESHOLD = 0.2  # Similarity threshold for context retrieval
RAG_MAX_TOKENS = 1000    # Maximum tokens to include from RAG
```

## Running the Application

**Option 1: Use the start script**
```bash
# Windows
./start.ps1

# Linux/Mac
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 (or the port shown) in your browser.

## Usage

1. **Start a conversation** by clicking "New Conversation"
2. **Ask a question** - the full Council will deliberate (Stages 1-3)
3. **Ask follow-up questions** - the Chairman will respond using RAG context
4. **Expand reasoning** - click "💭 Chain of Thought" to see the model's thinking

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, OpenRouter API
- **Frontend:** React + Vite, react-markdown for rendering
- **RAG:** ChromaDB with `all-MiniLM-L6-v2` embeddings
- **Storage:** JSON files in `data/conversations/`, ChromaDB in `data/chroma_db/`
- **Package Management:** uv for Python, npm for JavaScript

## Architecture

### RAG System

The RAG system indexes Council deliberations with structured metadata:

- **Indexing**: After Stage 3, all responses are embedded with the original question prepended
- **Retrieval**: Follow-up questions query ChromaDB with cosine similarity
- **Gating**: Only chunks above the similarity threshold are retrieved
- **Formatting**: Context is formatted with `[Turn | Stage | Model]` headers for LLM consumption

See `backend/rag.py` for implementation details.

## Contributing

This is a fork for personal enhancement. If you'd like to contribute:
- For core functionality changes, consider contributing to the [original repo](https://github.com/karpathy/llm-council)
- For enhancements specific to RAG/chat features, feel free to open issues or PRs here

## Credits

**Original Author:** [Andrej Karpathy](https://github.com/karpathy) - [llm-council](https://github.com/karpathy/llm-council)

**Enhancements:** Multi-turn chat, RAG integration, chain of thought display

See [CREDITS.md](CREDITS.md) for detailed attribution.

## License

MIT License - See [LICENSE](LICENSE) for details.

This fork maintains the spirit of the original "vibe code" project while adding production-ready features for extended conversations.
