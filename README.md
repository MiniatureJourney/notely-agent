<div align="center">

# 📚 Notely — AI-Powered Academic Notes Platform

**The intelligent study companion for Indian college students.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Google_Cloud_Run-4285F4?style=for-the-badge)](https://notely-287509159734.us-central1.run.app)
[![Full Technical Walkthrough](https://img.shields.io/badge/🎥_Full_Technical_Walkthrough-YouTube-FF0000?style=for-the-badge)](https://youtu.be/J0GMxIIXiWc)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas_Vector_Search-00ED64?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![Vertex AI](https://img.shields.io/badge/Google-Vertex_AI_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Cloud Run](https://img.shields.io/badge/Deployed_on-Cloud_Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)

> **Google Cloud Rapid Agent Hackathon 2026 — MongoDB Partner Track**

</div>

---

## 🎯 The Problem

India has **40 million+ college students** with no centralized platform to share academic notes. Notes get lost in WhatsApp groups and private drives — with zero quality control, no semantic search, and no personalized study assistance.

## ✅ The Solution

Notely is a **full-stack agentic platform** where:
- Students upload notes (text or photo) and they are **automatically scored, embedded, and searchable**
- **NoteBot** — a Gemini 2.5 Flash AI agent — searches notes semantically, generates flashcards, quizzes, and summaries
- The agent **remembers** each student's academic strengths and weaknesses and adapts over time
- A **gamified leaderboard** drives community contribution

---

## 🌐 Live Demo

**→ [https://notely-287509159734.us-central1.run.app](https://notely-287509159734.us-central1.run.app)**

No login required — create a quick profile and start exploring.

---

## 🏗️ Architecture

```
Student Message
      │
      ▼
┌─────────────────────────────────────────────────────┐
│          FastAPI Backend  (Google Cloud Run)         │
│                                                     │
│  POST /api/chat                                     │
│      │                                              │
│      ├─► Gemini 2.5 Flash (Vertex AI)               │
│      │     Function Calling Loop (up to 8 rounds)   │
│      │          │                                   │
│      │          ├─► search_notes ──► MCP Server ──► MongoDB Atlas $vectorSearch
│      │          ├─► generate_summary ──────────────► fetch full_content
│      │          ├─► generate_flashcards ──────────► Gemini Q&A pairs
│      │          ├─► generate_quiz ────────────────► Gemini MCQ + answers
│      │          ├─► record_quiz_score ────────────► user_memory collection
│      │          ├─► get_leaderboard ──────────────► users collection
│      │          └─► get_forum_posts / get_requests ► forum / requests
│      │                                              │
│      └─► Response + Activity Feed (live tool log)  │
│                                                     │
└─────────────────────────────────────────────────────┘
      │
      ▼
MongoDB Atlas (notely_db)
  ├── notes          (embeddings + full content)
  ├── users          (profiles + points + memory)
  ├── user_memory    (strong / weak topics per student)
  ├── forum_posts
  ├── requests
  └── comments
```

---

## ✨ Features

### 🤖 NoteBot AI Agent
| Feature | Description |
|---|---|
| Semantic Search | MongoDB Atlas `$vectorSearch` via official MCP Server — finds notes by meaning, not just keywords |
| Summarization | Exam-ready concise summary from any note |
| Flashcards | Generates N Q&A pairs in structured format |
| Practice Quizzes | Multiple-choice questions with answers and explanations |
| Academic Memory | Tracks each student's weak/strong topics, updated after every quiz |
| Learning Patterns | AI observes and records how a student learns over time |
| Emergency Exam Mode | Pulls weak topics → searches best notes → generates a timed 24-hour study plan |
| MCP Activity Feed | Live display of every tool call (name, args, timestamp, whether it used MCP) |

### 📝 Note Management
| Feature | Description |
|---|---|
| Manual Upload | Rich text form — title, subject, college, dept, semester, chapter, teacher |
| OCR Image Upload | Photo of handwritten notes → Gemini Vision auto-extracts text, classifies subject/chapter/semester |
| Image Compression | Client-side Canvas compression before OCR (10MB → ~1MB, 85% quality — non-destructive) |
| Quality Scoring | Gemini evaluates every note on 4 criteria — Completeness, Clarity, Examples, Exam Usefulness (0–100) |
| Strengths & Weaknesses | Per-note AI feedback stored in MongoDB and displayed on cards |
| Vector Embedding | `text-embedding-005` (768-dim) embedded at upload for semantic search |

### 🏘️ Community
| Feature | Description |
|---|---|
| Home Feed | Notes sorted by quality score, with filter chips (All / Trending / Newest / My College) |
| Browse Notes | Full semantic search + filters for college, department, semester |
| Leaderboard | Global + college-filtered rankings by points, with medal badges |
| College Forum | Discussion board with upvotes and "Ask NoteBot" integration |
| Note Requests | Request missing notes; fulfill requests for bonus points |
| Comments | Comment on any note |

### 🎮 Gamification
| Action | Points |
|---|---|
| Upload a note (text) | +50 pts |
| Upload via OCR (image) | +100 pts |
| Post a comment | +5 pts |
| Upvote a note | +1 pt |
| Fulfill a note request | +100 pts |

### ⚡ Performance (Production Optimizations)
- **MongoDB Compound Index** on `(college, department, semester)` — query latency from ~50ms → ~2ms
- **Exact-match queries** replace regex scans across all collections
- **Strict data projections** — `full_content` and `embedding` excluded from feed endpoints (~90% payload reduction)
- **Independent async fetches** — all dashboard API calls fire concurrently, each updates the DOM independently
- **Optimistic UI** — upvotes (notes + forum) update instantly, revert silently on failure
- **Cold-start auto-retry** — home feed shows friendly message and retries automatically after 4 seconds

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **AI Agent** | Gemini 2.5 Flash (Vertex AI) — function calling loop |
| **Embeddings** | `text-embedding-005` (Vertex AI) — 768 dimensions |
| **OCR** | Gemini 2.5 Flash Vision API |
| **Quality Scoring** | Gemini 2.5 Flash — 4-criterion academic rubric |
| **Database** | MongoDB Atlas — operational + vector search |
| **MCP** | `@mongodb-js/mongodb-mcp-server` (official, stdio transport) |
| **Backend** | FastAPI (Python 3.11) + Pydantic |
| **Frontend** | Vanilla HTML / CSS / JavaScript (single-file SPA) |
| **Hosting** | Google Cloud Run (us-central1) |
| **Container** | Docker + uvicorn |

---

## 📡 API Reference

### AI & Agent
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | NoteBot agentic chat (Gemini + MCP tool loop) |
| `POST` | `/api/exam-mode` | Generate a personalized 24-hour study plan |

### Notes
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/notes` | Browse notes — filter by `college`, `department`, `semester`, `sort` |
| `GET` | `/api/notes/search?q=` | Semantic vector search with optional filters |
| `GET` | `/api/notes/{id}` | Fetch full note by ID |
| `POST` | `/api/upload-note` | Upload text note (JSON body) |
| `POST` | `/api/ocr-note` | Upload image note (multipart/form-data) |
| `POST` | `/api/notes/{id}/upvote` | Upvote a note |
| `GET` | `/api/notes/{id}/comments` | Get comments on a note |
| `POST` | `/api/comments` | Post a comment |

### Community
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/forum` | Get forum posts (filter by `college`) |
| `POST` | `/api/forum` | Create a forum post |
| `POST` | `/api/forum/{id}/upvote` | Upvote a post |
| `GET` | `/api/requests` | Get open note requests |
| `POST` | `/api/requests` | Post a note request |
| `POST` | `/api/requests/{id}/fulfill` | Mark a request as fulfilled |
| `GET` | `/api/leaderboard` | Get top contributors |

### Users
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/users` | Create / update user profile |
| `GET` | `/api/users/{user_id}` | Get user profile + points |
| `GET` | `/api/user/{user_id}/notes` | Get all notes by a user |

### Platform & Infrastructure
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/platform-stats` | Live counts of notes, users, open requests |
| `GET` | `/api/colleges` | Distinct college names (for dynamic dropdowns) |
| `GET` | `/api/departments` | Distinct department names (for dynamic dropdowns) |
| `GET` | `/health` | Service health check |
| `POST` | `/mcp` | MCP JSON-RPC transport (Google AI Agent Builder) |
| `GET` | `/mcp/health` | MCP server connection health |
| `GET` | `/api/mcp/tools` | List all MongoDB MCP tools |
| `POST` | `/api/mcp/execute` | Execute any MCP tool via REST |

---

## 🗄️ MongoDB Atlas Setup

### Vector Search Index
Create this index in Atlas under `notely_db.notes`:

```json
{
  "name": "notes_vector_index",
  "type": "vectorSearch",
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "college" },
    { "type": "filter", "path": "department" },
    { "type": "filter", "path": "semester" },
    { "type": "filter", "path": "subject" }
  ]
}
```

### Compound Index (Performance)
```javascript
db.notes.createIndex({ college: 1, department: 1, semester: 1 })
```

---

## 💻 Local Development

```bash
# 1. Clone the repository
git clone https://github.com/MiniatureJourney/notely-agent.git
cd notely-agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
# Create a .env file with:
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
VERTEX_REGION=us-central1

# 5. Run the development server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# 6. Open the app
# Visit http://localhost:8080 in your browser

# 7. (Optional) Seed the database with sample notes
python seed_data.py
```

---

## 🚀 Deploy to Google Cloud Run

```bash
gcloud run deploy notely \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,VERTEX_REGION=us-central1,MONGODB_URI=YOUR_MONGODB_URI"
```

---

## 🔌 Google AI Agent Builder — MCP Connection

1. Open **Google AI Studio → Agent Platform → Your Agent → Tools**
2. Under **MCP**, click **+ Add MCP Server**
3. Enter the server URL:
   ```
   https://notely-287509159734.us-central1.run.app/mcp
   ```
4. The agent automatically discovers all 28 MongoDB Atlas tools

---

## 📁 Project Structure

```
notely-agent/
├── main.py              # FastAPI app — all REST endpoints + Gemini agentic loop
├── database.py          # All MongoDB Atlas operations (27+ functions)
├── agent_tools.py       # 11 Gemini tool implementations + async dispatcher
├── embeddings.py        # Vertex AI text-embedding-005 wrapper with caching
├── rules_engine.py      # Gemini-powered 4-criterion note quality scorer
├── mcp_bridge.py        # Official MongoDB MCP Server bridge (stdio transport)
├── seed_data.py         # Database seeding script for sample notes
├── requirements.txt     # Python dependencies
├── Dockerfile           # Production container definition
├── docs/
│   └── index.html       # Entire frontend (single-file SPA)
└── tests/
    ├── test_all.py
    ├── test_chat_memory.py
    ├── test_gemini.py
    ├── test_ocr.py
    ├── test_ocr_api.py
    ├── test_regex.py
    ├── test_rules.py
    └── test_system_rigorous.py
```

---

## 🧪 Running Tests

```bash
# Run the full test suite
pytest tests/

# Run a specific test file
pytest tests/test_system_rigorous.py -v

# Run OCR tests
pytest tests/test_ocr.py -v
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ for the **Google Cloud Rapid Agent Hackathon 2026 — MongoDB Partner Track**

**[Live Demo](https://notely-287509159734.us-central1.run.app)** · **[Report an Issue](https://github.com/MiniatureJourney/notely-agent/issues)**

</div>
