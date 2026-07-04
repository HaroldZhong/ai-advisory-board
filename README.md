# AI Advisory Board

A desktop AI assistant powered by a **multi-LLM deliberative council** — multiple AI models debate, rank, and synthesize answers to deliver higher-quality responses than any single model alone. Features **persistent memory (PageIndex RAG)**, **web search**, **document processing**, and **custom personas**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![ai-advisory-board](header.png)

---

## 🎯 Key Features

### Council Process (3-Stage Deliberation)

- **Stage 1 — Collect**: 5+ LLMs independently answer your question
- **Stage 2 — Rank**: Models anonymously evaluate and rank all responses
- **Stage 3 — Synthesize**: Chairman model creates a final answer based on rankings and deliberation, with a confidence score (HIGH/MEDIUM/LOW)

### Memory & Retrieval (PageIndex RAG)

- **Persistent Memory**: Conversations are indexed and retrievable across sessions
- **Document Indexing**: Uploaded files are automatically indexed for future retrieval
- **Query Rewriting**: Automatic coreference resolution for natural follow-ups

### Web Search

- **Perplexity Integration**: Native web search via OpenRouter (Sonar models)
- **Fast/Deep Modes**: Quick web lookups or thorough research-grade searches
- **Inline Toggle**: One-click enable from the chat input bar

### File Processing

- **Drag & Drop**: Drop files directly into the chat
- **Supported Formats**: PDF, DOCX, PPTX, XLSX, CSV, TXT, Markdown, HTML, JSON, and images
- **Vision**: Images are analyzed using vision models
- **PageIndex Integration**: Extracted text is automatically indexed for cross-chat retrieval

### Advanced Features

- **Custom Personas**: Set persistent instructions that shape every response
- **Edit & Regenerate**: Click any previous message to edit and regenerate from that point
- **Chat Export**: Export conversations to Markdown
- **Session Budgets**: Set spending limits ($1/$2/$5/unlimited) with graceful degradation
- **Cost Tracking**: Real-time per-conversation and per-model cost analytics
- **40+ Models**: Curated registry from OpenAI, Anthropic, Google, xAI, DeepSeek, Mistral, and more
- **Folder Organization**: Group conversations into color-coded folders

---

## 🚀 Getting Started

### Option A: Desktop App (Recommended)

The simplest way to use the AI Advisory Board — a single executable that runs everything locally.

1. **Download** `AI Advisory Board.exe` from [Releases](https://github.com/HaroldZhong/ai-advisory-board/releases)
2. **Run** the exe — it launches a local server and opens the app in a native window
3. **Complete first-run setup** in the app: connect your OpenRouter key, choose privacy defaults, and set a starting session budget

> **Get an API key**: Sign up at [OpenRouter](https://openrouter.ai/) (free tier available)
>
> Windows may show an unsigned-app SmartScreen warning. Choose **More info → Run anyway** if you trust the downloaded release. Windows also needs the Microsoft Edge WebView2 Runtime; most Windows 11 machines already include it.

See [Installation & Packaging](docs/installation.md) for data locations, uninstall cleanup, WebView2 notes, and packaging details.

### Option B: Development Setup

For developers who want to modify the code:

#### Prerequisites

- Python 3.10+
- Node.js 18+
- [OpenRouter API key](https://openrouter.ai/)

#### Installation

```bash
# Clone
git clone https://github.com/HaroldZhong/ai-advisory-board.git
cd ai-advisory-board

# Backend
pip install uv
uv sync

# Frontend
cd frontend && npm install && cd ..

# Environment
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

#### Running

```bash
# Option 1: Start script (recommended)
./start.ps1     # Windows
./start.sh       # Linux/Mac

# Option 2: Manual
# Terminal 1 — Backend
uv run uvicorn backend.main:app --reload --port 8001

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Then open **http://localhost:5173** in your browser.

#### Building the Desktop App

```bash
# 1. Build the frontend
cd frontend && npm run build && cd ..

# 2. Build the exe
uv run --group packaging python build_exe.py

# Output: dist/AI Advisory Board.exe
```

---

## 📖 How It Works

### Data Flow

```
User Query
    ↓
[Optional] Web Search (Perplexity Sonar)
    ↓
Query Rewriting (resolve coreferences)
    ↓
PageIndex RAG Retrieval (persistent memory)
    ↓
Custom Instructions (persona prefix)
    ↓
Stage 1: Council responses (5+ models)
    ↓
Stage 2: Anonymous peer ranking
    ↓
Stage 3: Chairman synthesis + confidence
    ↓
Index session into PageIndex
    ↓
Display to user
```

### Conversation Modes

| Mode | When | What Happens |
|------|------|--------------|
| **Council** | First message | Full 3-stage deliberation with all council models |
| **Chat** | Follow-ups | Quick response from Chairman with RAG context |

---

## 🔧 Configuration

### Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-...  # Required — get at openrouter.ai
OPENROUTER_BASE_URL=...       # Optional relay base (see docs/installation.md → Network access)
HTTPS_PROXY=...               # Honored by the backend; Windows system proxy is NOT read
```

### Model Configuration (`backend/config.py`)

```python
# Default council members (sent in Stage 1)
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4.6",
    "x-ai/grok-4-fast",
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2-exp",
]

# Chairman (synthesizes in Stage 3)
CHAIRMAN_MODEL = "google/gemini-2.5-flash"
```

Models can be changed at runtime via the Model Selector in the UI. The curated registry (`CURATED_MODELS` in `config.py`) includes 40+ models across tiers:

- **Chairman Tier**: GPT-5.2, Gemini 3.1 Pro, Claude Opus 4.6, Kimi K2.5
- **Workhorse Tier**: GPT-5.1, GPT-5.3 Chat, Claude Sonnet 4.6, DeepSeek V3.2
- **Free Tier**: GPT-OSS 120B/20B, Devstral 2512

### In-App Settings

| Setting | Description |
|---------|-------------|
| **Session Budget** | Spending limit per conversation ($1/$2/$5/unlimited) |
| **Web Search** | Enable Perplexity web search (fast/deep) |
| **Custom Instructions** | Persistent persona/system prompt |
| **RAG Preset** | Memory retrieval depth (auto/low/medium/high/max) |
| **Zero Data Retention** | ZDR mode for privacy-sensitive queries |

---

## 🛠️ Architecture

### Backend (FastAPI + Python)

```
backend/
├── main.py                  # API endpoints, streaming, routing
├── council.py               # 3-stage council orchestration
├── openrouter.py            # OpenRouter API client
├── rag.py                   # PageIndex RAG system
├── storage.py               # JSON-based conversation storage
├── config.py                # Models, RAG, and budget configuration
├── web_search.py            # Perplexity web search integration
├── file_processing.py       # PDF/DOCX/PPTX/XLSX/image extraction
├── attachment_storage.py    # Attachment lifecycle management
├── analytics.py             # Usage and cost analytics
├── rag_utils.py             # Query rewriting utilities
├── budget_policy.py         # Budget-aware routing
├── logger.py                # Structured logging
├── pageindex/               # PageIndex reasoning RAG engine
└── tools/                   # Tool calling infrastructure
```

### Frontend (React + Vite + Tailwind + shadcn/ui)

```
frontend/src/
├── App.jsx                  # Main app + streaming handler
├── api.js                   # Backend API client
├── components/
│   ├── ChatInterface.jsx    # Chat UI, drag-drop, edit & regenerate
│   ├── Sidebar.jsx          # Conversation list + folder management
│   ├── ModelSelector.jsx    # Dynamic model picker
│   ├── AdvancedSettingsPanel.jsx  # Settings dialog
│   ├── AnalyticsDashboard.jsx    # Usage stats
│   ├── MarkdownRenderer.jsx      # LaTeX + Markdown rendering
│   └── ui/                  # shadcn/ui primitives
├── contexts/
│   └── SettingsContext.jsx  # Global settings state
└── landing/
    └── LandingPage.jsx      # Landing page
```

### Desktop Wrapper

```
desktop.py        # PyWebView window + FastAPI server launcher
build_exe.py      # PyInstaller build script
```

---

## 📁 Data Storage

Installed desktop builds store user data in the per-user app data directory. On Windows this resolves to:

```text
%LOCALAPPDATA%\HaroldZhong\AI Advisory Board\
```

Development checkouts keep data project-local by default. Set `AAB_DATA_DIR` to override either mode for testing or portable setups.

| What | Where | Format |
|------|-------|--------|
| API key | `.env` under the app data root | dotenv |
| Conversations | `data/conversations/` under the app data root | JSON per conversation |
| PageIndex Memory | `data/pageindex_memory.json` under the app data root | JSON reasoning index |
| Attachments | `data/conversations/attachments/` under the app data root | Binary files + metadata |
| Logs | `logs/app.log` and `logs/desktop.log` under the app data root | Rotating log files |

---

## 💰 Cost Governance

### Session Budget

Set a spending limit per conversation. The system automatically adjusts:

| Budget Spent | RAG Context | Behavior |
|-------------|-------------|----------|
| ≤75% | Auto | Full quality |
| 75–85% | Medium (8k) | Standard |
| 85–100% | Low (4k) | Quick mode |
| ≥100% | Low (4k) | New turns are blocked when enforcement is enabled |

New first-run and preset-created budgets are enforced at 100%. Raise the cap from the budget modal to continue sending new turns.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Original Concept**: [llm-council](https://github.com/karpathy/llm-council) by Andrej Karpathy
- **APIs**: [OpenRouter](https://openrouter.ai/) for unified LLM access
- **RAG**: [PageIndex](https://github.com/VectifyAI/PageIndex) reasoning-based retrieval
- **UI**: [shadcn/ui](https://ui.shadcn.com/) + [Tailwind CSS](https://tailwindcss.com/)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/HaroldZhong/ai-advisory-board/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HaroldZhong/ai-advisory-board/discussions)

---

**Built with ❤️ for transparent, deliberative AI conversations**
