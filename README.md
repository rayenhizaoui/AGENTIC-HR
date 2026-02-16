<p align="center">
  <img src="https://img.shields.io/badge/Platform-Intelligent%20HR-blue?style=for-the-badge" alt="Intelligent HR" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Frontend-React%2019-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/AI-LangGraph%20Multi--Agent-FF6F00?style=for-the-badge" alt="LangGraph" />
  <img src="https://img.shields.io/badge/LLM-Mistral%20(Ollama)-7C3AED?style=for-the-badge" alt="Mistral" />
</p>

# 🧠 Intelligent HR — Agentic AI Recruitment Platform

**Intelligent HR** is a full-stack, AI-powered recruitment platform built on a **multi-agent architecture** using **LangGraph**. It automates and streamlines the entire hiring pipeline — from job search and CV analysis to candidate ranking and offer letter generation — through specialized AI agents that collaborate autonomously.

> **Key Innovation:** Instead of a monolithic LLM chatbot, Intelligent HR uses a **Supervisor → Agent hierarchy** where a central router delegates tasks to domain-specific agents equipped with specialized tools, enabling accurate and context-aware HR operations.

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Multi-Agent System](#-multi-agent-system)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Frontend Pages](#-frontend-pages)
- [Job Sources](#-job-sources)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🤖 AI Chat Assistant
- Real-time conversational interface via **WebSocket** with streamed reasoning logs
- Multilingual support: **English**, **French**, and **Arabic**
- Intent detection + entity extraction (skills, location, level, salary) powered by Mistral LLM with rule-based fallback
- Transparent agent routing visible to the user in real-time

### 🔍 Multi-Source Job Search
- Aggregates jobs from **7+ sources** simultaneously (APIs, RSS feeds, HTML scraping)
- Advanced filters: location, remote-only, experience level (junior/mid/senior)
- **CV-based job recommendations** using 60% semantic embedding similarity + 40% skill overlap scoring

### 📄 CV Analysis & Processing
- **PDF parsing** with OCR fallback for scanned documents (EasyOCR + pdf2image)
- **Semantic skill enhancement** using `all-MiniLM-L6-v2` embeddings + synonym expansion
- Regex-based + NLP skill extraction across 25+ technology categories
- **PII anonymization** (names, emails, phones, addresses) — privacy-first pipeline
- Skills by category, certifications, education timeline, project count extraction
- In-memory CV library with persistent cache for multi-CV operations

### 📊 Candidate Ranking
- **Hybrid scoring**: 60% semantic similarity (SentenceTransformer) + 40% skill match (Mistral LLM feature extraction)
- Rank multiple CVs against any job description
- Matched/missing skills breakdown with progress bars and score badges

### 📝 Offer Letter Generation
- 3-step wizard: Position Details → Compensation → Review & Generate
- **RAG-powered templates** via ChromaDB with company knowledge base
- Salary market competitiveness checker
- Copy, download, or regenerate offers with live preview

### 🌐 Markdown Rendering
- Assistant messages render full **GitHub Flavored Markdown**: tables, links, headings, code blocks
- Job search results display as styled, clickable tables with external link icons

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (React 19)               │
│                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │   Chat   │ │  Search  │ │ Analyze  │ │ Hiring │ │
│  │(WebSocket)│ │(HTTP API)│ │(HTTP API)│ │(HTTP)  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└───────┼─────────────┼───────────┼────────────┼──────┘
        │ ws://       │ REST      │ REST       │ REST
┌───────┼─────────────┼───────────┼────────────┼──────┐
│       ▼             ▼           ▼            ▼      │
│              FastAPI Backend (Port 8001)             │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │          🧠 Supervisor (Router)               │   │
│  │   Intent Detection + Entity Extraction        │   │
│  │   Mistral LLM + Rule-based Fallback           │   │
│  └──────────────┬──────────────┬─────────────────┘   │
│                 │              │                      │
│    ┌────────────▼──┐    ┌─────▼────────────┐        │
│    │ Lead Recruiter│    │  Hiring Manager  │        │
│    │  (14 Tools)   │    │    (4 Tools)     │        │
│    └───────────────┘    └──────────────────┘        │
│                                                     │
│    ┌─────────────────────────────────────────┐      │
│    │  Shared State (AgentState - TypedDict)  │      │
│    │  messages, job_context, filters, intent │      │
│    │  search_results, ranking_results        │      │
│    └─────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
        │                    │
   ┌────▼────┐        ┌─────▼──────┐
   │ Ollama  │        │  ChromaDB  │
   │ Mistral │        │ (Templates)│
   └─────────┘        └────────────┘
```

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** | Async REST API + WebSocket server |
| **LangGraph** | Multi-agent graph orchestration |
| **LangChain** | LLM abstractions, message types, tool interfaces |
| **Mistral (Ollama)** | Local LLM for intent detection, entity extraction, ranking, offer generation |
| **SentenceTransformers** | `all-MiniLM-L6-v2` embeddings for semantic similarity |
| **ChromaDB** | Vector store for RAG template retrieval |
| **EasyOCR** | OCR for scanned PDF documents |
| **spaCy** | NLP entity recognition & text processing |
| **pdfplumber / PyMuPDF** | PDF text extraction |
| **BeautifulSoup4** | HTML scraping for Tunisian job boards |
| **feedparser** | RSS feed parsing (We Work Remotely) |

### Frontend
| Technology | Purpose |
|---|---|
| **React 19** | UI framework |
| **TypeScript** | Type safety |
| **Vite 7** | Build tool & dev server |
| **TailwindCSS 3** | Utility-first CSS styling |
| **Framer Motion** | Animations & transitions |
| **React Router 7** | Client-side routing |
| **Axios** | HTTP client |
| **react-markdown + remark-gfm** | Markdown rendering in chat |
| **Lucide React** | Icon library |

---

## 🤖 Multi-Agent System

### Supervisor (Router)
The central orchestrator that:
1. **Detects intent** from user messages using Mistral LLM (with rule-based fallback)
2. **Extracts entities**: skills, location, experience level, salary, candidate name
3. **Routes** to the appropriate specialized agent
4. **Maintains state** across the conversation

**Supported intents:** `job_search`, `cv_analysis`, `cv_ranking`, `offer_generation`, `salary_check`, `email_draft`, `greeting`, `clarification_needed`

### Lead Recruiter Agent — 14 Tools

| # | Tool | Description |
|---|---|---|
| 1 | `cv_parser_tool` | Parse PDF/DOCX CVs with text extraction |
| 2 | `batch_cv_parser` | Parse multiple CVs in batch |
| 3 | `text_cleaner_pipeline` | Clean and normalize CV text |
| 4 | `ocr_cv_tool` | OCR for scanned/image-based PDFs |
| 5 | `anonymizer_tool` | Redact PII (names, emails, phones, addresses) |
| 6 | `skill_extractor_tool` | Regex + word-boundary skill extraction (25+ categories) |
| 7 | `semantic_skill_enhancer` | Embedding-based skill discovery + synonym expansion |
| 8 | `candidate_summarizer` | Generate markdown-formatted candidate summaries |
| 9 | `similarity_matcher_tool` | Semantic similarity matching (SentenceTransformer) |
| 10 | `match_explainer` | Explain match scores between CV and job |
| 11 | `cv_ranker` | Rule-based candidate ranking |
| 12 | `llm_rank_candidates` | Hybrid ranking (60% embedding + 40% LLM skill match) |
| 13 | `job_scraper_tool` | Web scraping for job postings |
| 14 | `job_search_tool` | Multi-source job aggregation with dedup + caching |

### Hiring Manager Agent — 4 Tools

| # | Tool | Description |
|---|---|---|
| 1 | `template_retriever_tool` | RAG-based template retrieval from ChromaDB |
| 2 | `job_offer_generator` | Generate professional offer letters |
| 3 | `offer_validator_tool` | Validate offer completeness and terms |
| 4 | `market_salary_check` | Check salary competitiveness for a role |

---

## 📁 Project Structure

```
intelligent-hr/
├── README.md
├── backend/
│   ├── requirements.txt              # Python dependencies
│   ├── app/
│   │   ├── main.py                   # FastAPI app + WebSocket endpoint
│   │   ├── __init__.py
│   │   ├── api/                      # REST API routes
│   │   │   ├── router.py             # Central API router
│   │   │   ├── chat.py               # POST /chat — HTTP chat fallback
│   │   │   ├── search.py             # POST /search — Job search + /recommend
│   │   │   ├── candidates.py         # POST /candidates/analyze, /rank, /cached
│   │   │   └── hiring.py             # POST /hiring/offer, /salary-check
│   │   ├── agents/                   # Multi-agent system
│   │   │   ├── supervisor.py         # Supervisor router (intent + routing)
│   │   │   ├── shared/
│   │   │   │   ├── state.py          # AgentState (TypedDict) — shared state
│   │   │   │   └── utils.py          # Logger, helpers
│   │   │   ├── recruiter_agent/
│   │   │   │   ├── graph.py          # LangGraph graph for recruiter
│   │   │   │   └── tools/            # 14 recruiter tools
│   │   │   │       ├── parsers.py           # CV parser, batch parser, cleaner
│   │   │   │       ├── extraction.py        # Skill extraction + summarizer
│   │   │   │       ├── ranking.py           # Rule-based ranking
│   │   │   │       ├── llm_ranker.py        # Hybrid LLM+embedding ranking
│   │   │   │       ├── scraping.py          # Job scraper
│   │   │   │       ├── job_fetcher.py       # Multi-source job aggregation
│   │   │   │       ├── anonymizer_tool.py   # PII redaction
│   │   │   │       ├── similarity_matcher_tool.py  # Semantic matching
│   │   │   │       ├── match_explainer.py   # Score explanation
│   │   │   │       ├── job_cache.py         # Job result caching
│   │   │   │       ├── ocr_tool.py          # OCR for scanned PDFs
│   │   │   │       └── semantic_extractor.py # Semantic skill enhancer
│   │   │   └── manager_agent/
│   │   │       ├── graph.py          # LangGraph graph for hiring manager
│   │   │       └── tools/
│   │   │           ├── retrieval.py  # ChromaDB template retrieval (RAG)
│   │   │           └── generation.py # Offer generation + salary check
│   │   └── data/
│   │       └── company_knowledge/    # RAG knowledge base
│   │           ├── benefit_packages/standard.txt
│   │           ├── company_values/values.txt
│   │           └── offer_templates/  # Template documents
│   │               ├── data_scientist.txt
│   │               ├── junior_dev.txt
│   │               └── senior_tech.txt
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx                  # React entry point
│       ├── App.tsx                   # Router + layout
│       ├── index.css                 # Global styles (Tailwind)
│       ├── components/
│       │   └── Sidebar.tsx           # Navigation sidebar
│       ├── pages/
│       │   ├── Chat.tsx              # AI Chat (WebSocket + markdown)
│       │   ├── Search.tsx            # Job search + CV recommendations
│       │   ├── Analyze.tsx           # CV upload, library, ranking
│       │   └── Hiring.tsx            # Offer letter wizard
│       ├── services/
│       │   └── api.ts               # API client (Axios + WebSocket)
│       └── lib/
│           └── utils.ts             # Utility functions (cn)
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.10+ |
| **Node.js** | 18+ |
| **Ollama** | Latest |

### 1. Clone the Repository

```bash
git clone https://github.com/rayenhizaoui/AGENTIC-HR.git
cd AGENTIC-HR
```

### 2. Setup Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for NLP processing)
python -m spacy download en_core_web_sm
```

### 3. Setup Ollama (Local LLM)

```bash
# Install Ollama from https://ollama.com
# Pull the Mistral model
ollama pull mistral
```

> Ollama must be running on `http://localhost:11434` before starting the backend.

### 4. Start the Backend

```bash
cd backend
.\venv\Scripts\Activate.ps1  # or source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

The API will be available at `http://localhost:8001`.

### 5. Setup & Start Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### 6. Verify

- Open `http://localhost:5173` in your browser
- The sidebar should show **Intelligent HR** with 4 navigation links
- Try sending a message in the Chat page to verify the full pipeline

---

## 📡 API Reference

### Health Check
```
GET /health
→ { "status": "ok" }
```

### Chat
```
POST /chat
Body: { "message": "string", "context": {} }
→ { "response": "...", "reasoning_log": [...], "context": {} }
```

### WebSocket Chat (Real-time)
```
WS /ws/chat
Send: { "message": "string", "context": {} }
Receive:
  → { "type": "log", "step": "📥 Input: ..." }          // Streamed reasoning
  → { "type": "log", "step": "🧠 Intent: job_search" }
  → { "type": "log", "step": "🔀 Route → Lead_Recruiter" }
  → { "type": "response", "response": "...", "reasoning_log": [...] }
```

### Job Search
```
POST /search
Body: { "query": "python developer", "sources": [...], "max_results": 20,
        "location": "remote", "remote_only": true, "experience_level": "senior" }
→ { "total_found": 15, "jobs": [...] }
```

### CV-Based Job Recommendations
```
POST /search/recommend
Body: { "sources": [...], "max_results": 20, "location": "...", "remote_only": false }
→ { "cv_filename": "...", "cv_skills": [...], "jobs": [{ ..., "compatibility": 78.5 }] }
```

### CV Analysis
```
POST /candidates/analyze
Body: FormData { file: <PDF/DOCX> }
→ { "filename": "...", "text": "...", "skills": [...], "skills_data": { "skill_categories": {...} },
     "summary": "...", "pages": 2, "extraction_method": "pdfplumber" }
```

### Candidate Ranking
```
POST /candidates/rank
Body: { "job_description": "...", "filenames": ["cv1.pdf", "cv2.pdf"] }
→ { "total_candidates": 2, "rankings": [{ "candidate": "...", "score": 85.3, ... }] }
```

### Cached CVs
```
GET /candidates/cached
→ { "cached_cvs": [{ "filename": "...", "skills_count": 12, "summary_preview": "..." }], "total": 3 }
```

### Generate Offer Letter
```
POST /hiring/offer
Body: { "role": "...", "department": "...", "salary": "...", "start_date": "...",
        "candidate_name": "...", "location": "...", "contract_type": "..." }
→ { "offer_letter": "...", "context": {} }
```

### Salary Check
```
POST /hiring/salary-check
Body: { "role": "Data Scientist", "offered_salary": 75000 }
→ { "competitive": true, "market_range": { "min": 60000, "max": 95000 }, ... }
```

---

## 🖥 Frontend Pages

### 1. Chat Assistant (`/`)
- Real-time AI conversation via WebSocket
- Streamed reasoning logs showing intent detection, entity extraction, and agent routing
- CV upload directly from chat with drag & drop
- Full markdown rendering (tables, links, code blocks)
- Connection status indicator with auto-reconnect

### 2. Job Search (`/search`)
- **Search Jobs** tab: Multi-source search with filters (location, remote, experience level)
- **CV Match** tab: Upload CV → get personalized job recommendations with compatibility scores
- Source badges with color coding for each job board
- Direct apply links to original job postings

### 3. Analyze CVs (`/analyze`)
- **Upload & Analyze** tab: Drag & drop PDF upload, document info panel, full analysis result
- **CV Library** tab: Grid view of all analyzed CVs with skill counts and summaries
- **Rank for Job** tab: Paste job description → rank selected CVs → see scores with matched/missing skills
- Analysis includes: skill categories, certifications, career timeline, education with year, project count

### 4. Hiring Ops (`/hiring`)
- 3-step wizard: Position Details → Compensation (with salary gauge) → Review & Generate
- Live preview sidebar showing form data in real-time
- Generated offer with copy-to-clipboard, download as text, and regenerate actions

---

## 🌍 Job Sources

| Source | Type | Region | Method |
|---|---|---|---|
| **Remote OK** | API | Global (Remote) | JSON API |
| **We Work Remotely** | RSS | Global (Remote) | RSS Feed (feedparser) |
| **Arbeitnow** | API | Europe | JSON API |
| **The Muse** | API | Global | JSON API |
| **Remotive** | API | Global (Remote) | JSON API |
| **Himalayas** | API | Global (Remote) | JSON API |
| **Emploi.tn / TanitJobs** | Scraping | Tunisia 🇹🇳 | HTML Scraping (BeautifulSoup) |

All sources are queried in parallel with automatic deduplication (by title + company) and result caching.

---

## 📸 Screenshots

> Screenshots can be added here showing:
> - Chat interface with real-time reasoning logs
> - Job search results with source badges
> - CV analysis with skill categories and stats dashboard
> - Candidate ranking with score breakdown
> - Hiring wizard with offer preview

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is developed as part of an academic/professional project. All rights reserved.

---

<p align="center">
  Built with ❤️ using <strong>LangGraph</strong>, <strong>FastAPI</strong>, and <strong>React</strong>
</p>
