# Changelog

All notable changes to this fork are documented here.

## [Enhanced] - 2025-11-24

### Added
- **Multi-turn chat functionality** - Ask follow-up questions to the Chairman without re-running the full Council
- **RAG system** using ChromaDB for context-aware follow-ups
  - `all-MiniLM-L6-v2` embeddings for semantic search
  - Cosine similarity with configurable threshold (default 0.2)
  - Token budget management (max 1000 tokens per RAG context)
  - Idempotent indexing with deterministic IDs
- **Chain of thought display** - See the model's reasoning process in a collapsible UI
- **Additional AI models**:
  - `moonshotai/kimi-k2-thinking`
  - `deepseek/deepseek-v3.2-exp`
- **Smart context retrieval** - Threshold-based similarity search (no extra LLM calls)
- **Structured metadata** for indexed Council sessions:
  - `conversation_id`
  - `turn_index`
  - `stage` (opinion, review, synthesis)
  - `model`
- **Enhanced UI**:
  - Collapsible reasoning sections
  - Improved chat interface
  - Loading indicators for chat mode

### Changed
- Modified `chat_with_chairman` to return dict with `content` and optional `reasoning`
- Updated frontend to handle both Council and Chat modes
- Simplified conversation history in chat mode (only Stage 3 synthesis included in immediate context)
- Updated API timeout from 300s back to 120s (RAG makes this possible)
- Enhanced CORS configuration to support multiple frontend ports

### Technical Implementation
- **RAG Indexing**: Prepends original user question to each chunk (`Q: {question}\n\nA: {answer}`)
- **Retrieval Strategy**: 
  - Query ChromaDB with user's follow-up question
  - Calculate similarity from cosine distance
  - Filter by threshold (0.2)
  - Sort by similarity descending
  - Accumulate chunks up to token budget
- **Context Format**: `[Turn {n} | Stage {opinion|review|synthesis} | Model: {name}]`

### Dependencies Added
- `chromadb>=0.5.0`
- `sentence-transformers>=3.0.0`

## [Original] - 2025-11-23

Base fork from [karpathy/llm-council](https://github.com/karpathy/llm-council)

Original features:
- 3-stage Council process (Stage 1: Opinions, Stage 2: Rankings, Stage 3: Synthesis)
- Multi-LLM deliberation via OpenRouter
- Anonymous peer review and ranking
- Chairman-synthesized final response
