# LLM Council Enhanced

An enhanced multi-turn AI chat system featuring a **3-stage deliberative council process** where multiple LLMs debate and synthesize answers, combined with **advanced RAG (Retrieval-Augmented Generation)** for context-aware conversations.

> **Forked from [karpathy/llm-council](https://github.com/karpathy/llm-council)**

[![Fork of karpathy/llm-council](https://img.shields.io/badge/fork-karpathy%2Fllm--council-blue)](https://github.com/karpathy/llm-council)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![llmcouncil](header.jpg)

---

## 🎯 Key Features

### ✨ Phase 1: Advanced RAG System (NEW - Dec 2024)
- **Hybrid Retrieval**: BM25 (keyword) + Dense (semantic) search with Reciprocal Rank Fusion
- **Query Rewriting**: Automatic coreference resolution for natural follow-up questions
- **Confidence Scoring**: HIGH/MEDIUM/LOW trust indicators based on council consensus
- **Enhanced Metadata**: Topic extraction and quality metrics for every conversation turn

### Core Council Process
- **Stage 1 (Collect)**: Multiple LLMs provide independent responses to your question
- **Stage 2 (Rank)**: Council members evaluate and rank each other's answers anonymously
- **Stage 3 (Synthesize)**: Chairman LLM creates final answer based on rankings and deliberation

### Additional Features
- **Multi-turn Conversations**: Context-aware dialogue with RAG-powered memory
- **Chain of Thought**: See reasoning steps from models that support it
- **Cost Tracking**: Real-time usage and cost analytics per conversation
- **Model Selection**: Choose your council members and chairman dynamically
- **File Upload**: Process PDFs, images, and text files with AI analysis
- **Analytics Dashboard**: Usage statistics, model performance, and cost breakdowns

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenRouter API key ([get one here](https://openrouter.ai/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HaroldZhong/llm-council-enhanced.git
   cd llm-council-enhanced
   ```

2. **Backend Setup**
   ```bash
   # Install uv (fast Python package manager)
   pip install uv
   
   # Install Python dependencies
   uv sync
   
   # Create .env file
   echo "OPENROUTER_API_KEY=your_key_here" > .env
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Start the Application**
   ```bash
   # Option 1: Use the start script (recommended)
   ./start.ps1  # Windows
   ./start.sh   # Linux/Mac
   
   # Option 2: Start manually
   # Terminal 1 - Backend
   uv run uvicorn backend.main:app --reload --port 8001
   
   # Terminal 2 - Frontend  
   cd frontend && npm run dev
   ```

5. **Open in Browser**
   ```
   http://localhost:5173
   ```

---

## 📖 How It Works

### The 3-Stage Council Process

When you ask a question in **Council Mode**:

1. **Stage 1: Independent Responses**
   - 5+ council models each provide their own answer
   - No knowledge of what others are saying
   - Responses shown side-by-side for comparison

2. **Stage 2: Anonymous Ranking**
   - Each model ranks ALL responses (including their own)
   - Responses are anonymized during ranking
   - Creates aggregate rankings showing consensus

3. **Stage 3: Final Synthesis**
   - Chairman model reviews all responses and rankings
   - Creates authoritative final answer
   - **NEW**: Includes confidence score (HIGH/MEDIUM/LOW)

### Advanced RAG Architecture

**Before Phase 1:**
- Simple dense-only retrieval
- No query preprocessing
- Basic metadata

**After Phase 1:**
- **Hybrid Retrieval**: Combines BM25 (keyword matching) + Dense (semantic understanding)
- **Query Rewriting**: Resolves pronouns like "it", "its", "that" in follow-ups
- **Rich Metadata**: Topics, quality scores, consensus metrics for every turn
- **Confidence Scoring**: Trust indicators based on model agreement

**Example:**
```
You: "Explain RAG systems"
Council: [Detailed explanation with HIGH confidence]

You: "What are its limitations?"
↓ Rewritten to: "What are the limitations of RAG systems?"
↓ Hybrid search finds relevant context
↓ Chairman answers using retrieved context
```

---

## 📊 Phase 1 RAG Features

### 1. Query Rewriting
Automatically expands abbreviated follow-up questions:
- **Original**: "What about its limitations?"
- **Rewritten**: "What are the limitations of RAG systems?"
- **Config**: `ENABLE_QUERY_REWRITE = True` in `backend/config.py`

### 2. Hybrid Retrieval
Combines keyword and semantic search:
- **BM25**: Exact keyword matching (e.g., "BM25", "ChromaDB")
- **Dense**: Semantic similarity (understands concepts)
- **Fusion**: Reciprocal Rank Fusion merges results
- **Filtering**: Conversation-scoped to prevent cross-talk

### 3. Confidence Scoring
Trust indicators based on council agreement:
- **HIGH** (>0.75): Strong consensus, factual questions
- **MEDIUM** (>0.5): Some disagreement, nuanced topics  
- **LOW** (≤0.5): Significant disagreement, subjective questions

### 4. Enhanced Metadata
Every council turn stores:
- **Topics**: Extracted keywords (e.g., "RAG", "ChromaDB")
- **Quality Metrics**: Per-model average rank and consensus score
- **Timestamps**: For temporal queries and analysis

---

## 🛠️ Architecture

### Backend (FastAPI + Python)
```
backend/
├── main.py                  # API endpoints and routing
├── council.py               # 3-stage council orchestration
├── rag.py                   # RAG system with ChromaDB
├── hybrid_retrieval.py      # BM25 + Dense fusion (NEW)
├── openrouter.py            # OpenRouter API client
├── storage.py               # JSON-based conversation storage
├── config.py                # Model configuration
├── analytics.py             # Usage analytics (NEW)
└── file_processing.py       # File upload handling (NEW)
```

### Frontend (React + Vite)
```
frontend/src/
├── App.jsx                  # Main application
├── components/
│   ├── ChatInterface.jsx    # Chat UI with council stages
│   ├── ModelSelector.jsx    # Dynamic model selection (NEW)
│   ├── AnalyticsDashboard.jsx  # Stats and metrics (NEW)
│   └── Sidebar.jsx          # Conversation management
└── api.js                   # Backend API client
```

### Data Flow
```
User Query
    ↓
Query Rewriting (resolve coreferences)
    ↓
RAG Retrieval (hybrid BM25 + dense)
    ↓
Stage 1: Council responses
    ↓
Stage 2: Peer ranking
    ↓
Stage 3: Chairman synthesis + confidence
    ↓
Index session (topics, quality, consensus)
    ↓
Display to user
```

---

## 🔧 Configuration

### Environment Variables (.env)
```bash
OPENROUTER_API_KEY=sk-or-...  # Required
```

### Model Configuration (backend/config.py)
```python
# Council members (5-7 recommended)
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    # ... add more
]

# Chairman (usually most capable model)
CHAIRMAN_MODEL = "google/gemini-2.5-pro"

# Phase 1 features
ENABLE_QUERY_REWRITE = True  # Enable/disable query rewriting
```

### RAG Tuning (backend/rag.py, backend/hybrid_retrieval.py)
```python
# Retrieval settings
RAG_MAX_TOKENS = 3000        # Max context size
threshold = 0.01             # RRF score threshold

# Hybrid weights
bm25_weight = 0.5           # Keyword importance
dense_weight = 0.5          # Semantic importance
```

---

## 💡 Usage Examples

### Council Mode (Full Deliberation)
Best for: Complex questions, important decisions, diverse perspectives
```
Ask: "Should I use microservices or monolithic architecture?"
→ 5 models debate
→ See rankings and reasoning
→ Get synthesized answer with MEDIUM confidence
```

### Chat Mode (Quick Responses)
Best for: Follow-ups, clarifications, quick answers
```
Ask: "What did you say about databases?"
→ Query rewritten automatically
→ Retrieves relevant past context
→ Quick answer from Chairman
```

---

## 📈 Analytics

Access the analytics dashboard to view:
- **Total Conversations**: Count and cost
- **Model Usage**: Which models are used most
- **Average Costs**: Per conversation and per model
- **Confidence Distribution**: HIGH/MEDIUM/LOW breakdown

---

## 🧪 Testing

### Manual Testing
```bash
# Test query rewriting
python test_query_rewrite.py

# Test metadata extraction
python test_metadata_smoke.py

# Phase 1 smoke tests (requires running backend)
python -m backend.eval_phase1
```

### Log Monitoring
Phase 1 features use `[PHASE1]` prefix for easy filtering:
```bash
# Watch logs for Phase 1 activity
tail -f backend.log | grep PHASE1

# You should see:
[PHASE1] Query rewrite: original='...' rewritten='...'
[PHASE1] Topics extracted: ['RAG', 'BM25']
[PHASE1] Confidence computed: label=HIGH avg_consensus=0.85
[PHASE1] Hybrid retrieval, candidates=6, returned=4
```

---

## 📁 Data Storage

### Conversations
- **Location**: `data/conversations/`
- **Format**: JSON files per conversation
- **Content**: All messages, council stages, costs, metadata

### RAG Index
- **Location**: `data/chroma_db/`
- **Engine**: ChromaDB vector database
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Metadata**: Topics, quality scores, timestamps

---

## 🚦 Roadmap

### Phase 1 ✅ COMPLETE (Dec 2024)
- [x] Query rewriting
- [x] Hybrid retrieval (BM25 + Dense)
- [x] Confidence scoring
- [x] Enhanced metadata

### Phase 1.5 (Future)
- [ ] Reranker for improved precision
- [ ] Web search integration for latest info
- [ ] Knowledge graph for complex relationships
- [ ] Contradiction detection

### Phase 2 (Planned)
- [ ] Multi-modal support (images, audio)
- [ ] Custom embedding models
- [ ] Advanced analytics and insights
- [ ] Storage migration (JSON → SQLite)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Original Concept**: [llm-council](https://github.com/karpathy/llm-council) by Andrej Karpathy
- **Enhancements**: RAG integration, hybrid retrieval, confidence scoring
- **APIs**: [OpenRouter](https://openrouter.ai/) for unified LLM access
- **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- **Retrieval**: [rank-bm25](https://github.com/dorianbrown/rank_bm25)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/HaroldZhong/llm-council-enhanced/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HaroldZhong/llm-council-enhanced/discussions)

---

**Built with ❤️ for transparent, deliberative AI conversations**
