# Notely — AI-Powered Academic Notes Platform

> **Google Cloud Rapid Agent Hackathon 2026 | MongoDB Partner Track**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Cloud Run](https://img.shields.io/badge/Deployed%20on-Google%20Cloud%20Run-blue)](https://notely-287509159734.us-central1.run.app)
[![MongoDB Atlas](https://img.shields.io/badge/Database-MongoDB%20Atlas-green)](https://mongodb.com/atlas)

**Live Demo:** https://notely-287509159734.us-central1.run.app

---

## What is Notely?

Notely is an AI-powered academic notes sharing platform built for 40M+ Indian college students. Students upload, discover, and study notes together — powered by a Gemini AI agent (NoteBot) that reasons through MongoDB Atlas using the official MongoDB MCP Server.

### The Core Problem
Indian college students have no centralized, AI-assisted platform to share academic notes. Notes are scattered across WhatsApp groups and private drives, with no quality control, no semantic search, and no personalized study assistance.

### The Solution
A full-stack agentic platform where:
- Students upload notes (with AI-powered OCR and quality scoring)
- NoteBot AI agent searches notes semantically using MongoDB Atlas Vector Search
- The agent remembers each student's academic strengths/weaknesses and adapts
- A gamified points and leaderboard system drives community contribution

---

## Agentic Architecture

```
Student Message
      │
      ▼
  FastAPI Backend (Google Cloud Run)
      │
      ├─► Gemini 1.5 Flash (Vertex AI)
      │      │ Function Calling Loop (up to 8 rounds)
      │      │
      │      ├─► search_notes ──────► MongoDB MCP Server ──► $vectorSearch
      │      ├─► generate_summary ──► MongoDB MCP Server ──► find note
      │      ├─► generate_flashcards ─► Gemini generates Q&A pairs
      │      ├─► generate_quiz ──────► Gemini generates MCQ quiz
      │      ├─► record_quiz_score ──► MongoDB user_memory (strong/weak topics)
      │      ├─► get_leaderboard ───► MongoDB users collection
      │      └─► get_requests ──────► MongoDB requests collection
      │
      ▼
  NoteBot Response + Activity Feed (shows live MCP tool calls)
```

---

## MongoDB Atlas Integration (Partner Track)

MongoDB Atlas serves as the **unified operational + vector memory layer**:

| Collection | Purpose |
|------------|---------|
| `notes` | All uploaded notes with 768-dim vector embeddings (`text-embedding-005`) |
| `users` | Student profiles, points, badges, strong/weak topics |
| `user_memory` | Persistent academic memory (updated by NoteBot after every quiz) |
| `quiz_results` | Full quiz history for learning analytics |
| `study_history` | Timestamped study sessions |
| `learning_patterns` | AI-observed behavioral patterns per student |
| `requests` | Open note requests from the community |
| `forum_posts` | College discussion threads |

### MongoDB MCP Server Integration
The official `@mongodb-js/mongodb-mcp-server` is pre-baked into the Docker image and runs as a subprocess. NoteBot routes all database queries through it via the MCP protocol, enabling Google AI Studio Agent Builder to connect directly:

```
MCP Endpoint: https://notely-287509159734.us-central1.run.app/mcp
```

### Vector Search Index
```json
{
  "name": "notes_vector_index",
  "type": "vectorSearch",
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine"},
    {"type": "filter", "path": "college"},
    {"type": "filter", "path": "department"},
    {"type": "filter", "path": "semester"},
    {"type": "filter", "path": "subject"}
  ]
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Agent Brain | Gemini 1.5 Flash via Vertex AI |
| Vector Embeddings | `text-embedding-005` (Vertex AI) |
| Database | MongoDB Atlas (operational + vector search) |
| MCP Integration | `@mongodb-js/mongodb-mcp-server` (official) |
| Backend | FastAPI (Python 3.11) |
| Frontend | Vanilla HTML/CSS/JS (glassmorphism design) |
| Hosting | Google Cloud Run |
| Agent Builder | Google AI Studio Agent Platform |
| OCR | Gemini Vision API |
| Quality Scoring | Gemini 1.5 Flash (4-criterion rubric) |

---

## Key Features

### NoteBot AI Agent
- Multi-step agentic reasoning (up to 8 Gemini function-call rounds per query)
- Semantic search via MongoDB Atlas Vector Search (`$vectorSearch` aggregation)
- Generates exam summaries, flashcards (Q&A pairs), and practice MCQ quizzes
- **Academic Memory Graph**: Tracks strong/weak topics per student, updates after every quiz
- **Emergency Exam Mode**: Pulls weak topics from MongoDB memory → finds targeted notes → generates 24-hour study plan

### Platform Features
- Upload notes (manual or OCR from photo/PDF)
- AI quality scoring (4 criteria: completeness, clarity, examples, exam usefulness)
- Points system (+50 per upload, +100 per fulfilled request)
- College leaderboard (live from MongoDB)
- Note Request Board
- College Forum
- Personal Academic Memory Graph (SVG knowledge graph)

---

## Local Development

```bash
# 1. Clone
git clone https://github.com/MiniatureJourney/notely-agent.git
cd notely-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env with your MONGODB_URI and GOOGLE_CLOUD_PROJECT

# 4. Run
uvicorn main:app --host 0.0.0.0 --port 8080

# 5. Seed sample data
python seed_data.py
```

---

## Deploy to Google Cloud Run

```bash
gcloud run deploy notely \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,VERTEX_REGION=us-central1,MONGODB_URI=YOUR_MONGODB_URI"
```

---

## Google AI Studio Agent Builder Connection

1. Go to **Google AI Studio → Agent Platform → Your Agent → Tools**
2. Under **MCP**, click **MCP Server (+)**
3. Enter server URL: `https://notely-287509159734.us-central1.run.app/mcp`
4. The agent will discover all 28 MongoDB tools automatically

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | NoteBot agentic chat (Gemini + MCP) |
| `/api/exam-mode` | POST | Emergency exam study plan |
| `/api/notes` | GET | Browse all notes |
| `/api/notes/search` | GET | Semantic vector search |
| `/api/upload-note` | POST | Upload with AI quality scoring |
| `/api/leaderboard` | GET | Live contributor rankings |
| `/api/forum` | GET/POST | College discussion board |
| `/api/requests` | GET/POST | Note request board |
| `/mcp` | POST | MCP JSON-RPC transport (Agent Builder) |
| `/mcp/health` | GET | MCP connection health check |
| `/openapi.json` | GET | OpenAPI spec for Agent Builder |
| `/health` | GET | Service health |

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

*Built for the Google Cloud Rapid Agent Hackathon 2026 — MongoDB Partner Track*
