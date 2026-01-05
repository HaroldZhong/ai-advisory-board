# Credits

## Original Author

This project is the **AI Advisory Board**, a significantly enhanced evolution of the original **[llm-council](https://github.com/karpathy/llm-council)** created by **[Andrej Karpathy](https://github.com/karpathy)**.

The core concept of using multiple LLMs in a deliberative process with:

- Anonymous peer ranking
- Multi-stage synthesis
- Chairman-led final response

...is entirely from the original work.

## Original Project Quote

> "The idea of this repo is that instead of asking a question to your favorite LLM provider, you can group them into your 'LLM Council'. This repo is a simple, local web app that essentially looks like ChatGPT except it uses OpenRouter to send your query to multiple LLMs, it then asks them to review and rank each other's work, and finally a Chairman LLM produces the final response."
>
> — Andrej Karpathy

## Enhancements

The following features were added in this fork:

### Core Features

- **Multi-turn chat mode** with the Chairman
  - Allows follow-up questions without re-running the full Council
  - Maintains conversation context efficiently
  
- **RAG (Retrieval-Augmented Generation) system**
  - ChromaDB integration for vector storage
  - Semantic search with `all-MiniLM-L6-v2` embeddings
  - Threshold-based context retrieval
  - Token budget management

- **Chain of thought display**
  - Extracts reasoning from model responses
  - Collapsible UI for better UX
  - Works with models that provide `reasoning_details`

- **Additional model support**
  - `moonshotai/kimi-k2-thinking`
  - `deepseek/deepseek-v3.2-exp`

### Technical Enhancements

- Structured metadata for indexed sessions
- Idempotent indexing with deterministic IDs
- Cosine similarity-based retrieval
- Smart context formatting for LLM consumption
- Enhanced error handling and logging

## Dependencies

### Original Dependencies

- FastAPI
- httpx
- OpenRouter API
- React + Vite
- react-markdown

### Added Dependencies

- ChromaDB (`chromadb>=0.5.0`)
- Sentence Transformers (`sentence-transformers>=3.0.0`)

See `pyproject.toml` and `frontend/package.json` for complete dependency lists.

## Acknowledgments

Special thanks to:

- **Andrej Karpathy** for the original LLM Council concept and implementation
- The **OpenRouter** team for providing unified LLM access
- The **ChromaDB** and **Sentence Transformers** communities for excellent embedding tools

## License

This fork maintains the same permissive spirit as the original project. See [LICENSE](LICENSE) for details.
