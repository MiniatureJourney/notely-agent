# Notely — Complete Technical Feature Audit
> Analyzed from full codebase + entire conversation history. No changes made.

---

## 📁 Project File Map

| File | Purpose | Size |
|---|---|---|
| `main.py` | FastAPI backend — all REST endpoints, Gemini agent loop | 1001 lines |
| `database.py` | All MongoDB Atlas operations (27+ functions) | 641 lines |
| `agent_tools.py` | 11 Gemini tool implementations + async dispatcher | 449 lines |
| `embeddings.py` | Vertex AI `text-embedding-005` wrapper with caching | 106 lines |
| `rules_engine.py` | Gemini-powered 4-criterion note quality scorer | 49 lines |
| `mcp_bridge.py` | Official MongoDB MCP Server bridge (stdio transport) | 106 lines |
| `docs/index.html` | Entire frontend — single-file SPA (~1800 lines) | ~1800 lines |
| `seed_data.py` | Database seeding script for test notes | 6269 bytes |
| `Dockerfile` | Production Docker image definition | 2914 bytes |
| `requirements.txt` | Python dependencies | 331 bytes |
| `tests/` | 8 test files (test_all, test_chat_memory, test_gemini, test_ocr, test_ocr_api, test_regex, test_rules, test_system_rigorous) | — |

---

## 🏗️ Infrastructure & Deployment

- **Platform**: Google Cloud Run (serverless containers)
- **Current Revision**: `notely-00024-wh2`
- **URL**: `https://notely-287509159734.us-central1.run.app`
- **Database**: MongoDB Atlas (`cluster0.n3zlyov`) — `notely_db`
- **AI**: Google Vertex AI (`us-central1`) — project `notely-study-assistant`
- **Container**: Docker, served via `uvicorn` on port 8080
- **Memory**: 2Gi Cloud Run instance
- **Static files**: `/app/static/` served by FastAPI `StaticFiles`
- **Frontend delivery**: Served directly from Cloud Run (same origin as API)
- **GitHub**: `MiniatureJourney/notely-agent` (main branch)

---

## 🗄️ MongoDB Atlas — Collections

| Collection | Purpose |
|---|---|
| `notes` | All uploaded academic notes |
| `users` | Student profiles + points + badges |
| `comments` | Comments on notes |
| `user_memory` | Academic memory per student (weak/strong topics) |
| `study_history` | Log of topics the student studied |
| `quiz_results` | Quiz score history |
| `learning_patterns` | AI-observed behavioral patterns per user |
| `requests` | Open note requests from community |
| `forum_posts` | College forum discussion posts |

### MongoDB Atlas Vector Index
- **Index name**: `notes_vector_index`
- **Field**: `embedding`
- **Dimensions**: 768 (text-embedding-005)
- **Type**: cosine similarity

### Compound Index (Performance)
- **Collection**: `notes`
- **Fields**: `college_1_department_1_semester_1`
- **Purpose**: Eliminates full-collection scans for filtered queries
- **Effect**: Query latency ~50ms → ~2ms for filtered browsing

---

## 🔌 Backend API Endpoints (FastAPI — `main.py`)

### Core
| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `index.html` frontend |
| `GET` | `/health` | Health check — returns model/db info |
| `GET` | `/api/platform-stats` | Live note/user/request counts (for ticker) |

### AI Agent
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Main NoteBot AI endpoint — full agentic loop with Gemini 2.5 Flash |
| `POST` | `/api/exam-mode` | Emergency Exam Mode — personalized 24-hour study plan |

### Notes
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/notes` | Browse notes — filterable by college/dept/semester, sortable by quality/recent/trending |
| `GET` | `/api/notes/search` | Semantic vector search using Atlas Vector Search |
| `GET` | `/api/notes/{note_id}` | Get full note by ID (includes full_content for NoteBot) |
| `POST` | `/api/upload-note` | Upload a text note (JSON body) — triggers embedding + quality scoring |
| `POST` | `/api/ocr-note` | Upload an image — Gemini OCR extracts and classifies content |
| `POST` | `/api/notes/{note_id}/upvote` | Upvote a note (+1 point to upvoter) |

### Comments
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/notes/{note_id}/comments` | Fetch all comments on a note |
| `POST` | `/api/comments` | Post a comment (+5 points) |

### Requests Board
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/requests` | Get open note requests |
| `POST` | `/api/requests` | Create a note request |
| `POST` | `/api/requests/{request_id}/fulfill` | Mark a request fulfilled (+100 points) |

### Forum
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/forum` | Get forum posts (filterable by college) |
| `POST` | `/api/forum` | Create a forum post |
| `POST` | `/api/forum/{post_id}/upvote` | Upvote a forum post |

### Leaderboard
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/leaderboard` | Top contributors ranked by points (filterable by college) |

### Users
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/users` | Create/upsert a user profile |
| `GET` | `/api/users/{user_id}` | Get user profile + points |
| `GET` | `/api/user/{user_id}/notes` | Get all notes uploaded by a user |

### Dynamic Filters
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/colleges` | Distinct college names from DB (for dynamic dropdowns) |
| `GET` | `/api/departments` | Distinct department names from DB (for dynamic dropdowns) |

### MCP (MongoDB MCP Server Bridge)
| Method | Path | Description |
|---|---|---|
| `GET` | `/mcp/health` | MCP server health probe |
| `POST` | `/mcp` | Streamable HTTP MCP transport (JSON-RPC) for Google AI Agent Builder |
| `GET` | `/api/mcp/tools` | Lists all tools exposed by MongoDB MCP Server |
| `POST` | `/api/mcp/execute` | Execute any MongoDB MCP tool via REST |

---

## 🤖 AI & Agent Capabilities (`agent_tools.py` + `main.py`)

### NoteBot Gemini 2.5 Flash — 11 Tools

| Tool Name | Description |
|---|---|
| `search_notes` | **PRIMARY PATH**: $vectorSearch aggregation via MCP Server → falls back to PyMongo |
| `generate_summary` | Fetches note content → Gemini writes exam-ready concise summary |
| `generate_flashcards` | Fetches note content → Gemini generates N Q&A flashcard pairs |
| `generate_quiz` | Fetches note content → Gemini creates MCQ quiz with answers & explanations |
| `get_trending_notes` | Returns highest upvoted/downloaded notes |
| `get_leaderboard` | Returns top student contributors by points |
| `get_comments` | Fetches all comments on a specific note |
| `get_requests` | Fetches open note requests from the community |
| `get_forum_posts` | Fetches college forum discussions |
| `record_quiz_score` | Updates student's Academic Memory (strong/weak topics based on score) |
| `record_study_session` | Logs what the student studied to memory |
| `add_learning_pattern` | AI observes and records behavioral patterns (e.g., "struggles with graphs") |

### MCP Search Architecture
- **Primary**: Official `@mongodb-js/mongodb-mcp-server` via stdio transport
- **Pipeline**: `$vectorSearch` aggregation with 768-dim cosine similarity
- **Fallback**: Direct PyMongo vector search if MCP session is unavailable
- **Activity Feed**: Every tool call is tagged `via_mcp: true/false` and shown in the frontend UI

### Emergency Exam Mode (`/api/exam-mode`)
1. Pulls student's `weak_topics` from MongoDB `user_memory`
2. Searches best notes for each weak topic (up to 4) via MCP
3. Gemini 2.5 Flash synthesizes a **timed 24-hour study plan** (table format)
4. Returns plan + full tool call log for Activity Feed

### Academic Memory System
- **Persisted**: Stored in `user_memory` collection per `user_id`
- **Weak topics**: Scored below 70% → flagged for review
- **Strong topics**: Scored above 70% → tracked as mastered
- **Learning patterns**: AI observes and writes behavioral notes (e.g., "prefers step-by-step math")
- **System prompt injection**: Memory is injected into NoteBot's context on every chat call

---

## 🧠 AI Models Used

| Model | Used For |
|---|---|
| `gemini-2.5-flash` (Vertex AI) | NoteBot chat agent, Emergency Exam Plan, Note quality scoring, OCR |
| `text-embedding-005` (Vertex AI) | 768-dim vector embeddings for semantic search |

---

## 📏 Note Quality Scoring (`rules_engine.py`)

Every note (both manual and OCR) is scored by Gemini against 4 academic criteria:

| Criterion | Max Points |
|---|---|
| Completeness — core concepts defined | 25 |
| Clarity & Formatting — headings/bullets | 25 |
| Practical Examples — solved problems | 25 |
| Exam Usefulness — good for revision | 25 |

- Score stored in MongoDB as `quality_score` (0–100)
- Strengths & weaknesses array stored per note
- Home feed sorted by `quality_score` descending by default
- Visual badge on each note card in the UI

---

## 🖼️ Frontend Pages (`docs/index.html`)

### Page: Home Feed
- Personalized greeting by first name
- 4 live stat cards: Notes on Platform, Your Uploads, Your Points, Total Downloads
- Filter chips: All / Trending / Newest / My College
- Note grid with quality score badge, upvote button (optimistic UI), Study with AI button
- Cold-start auto-retry: shows "⏳ Waking up the server..." → retries after 4 seconds
- All 4 dashboard API requests fire **independently** (no blocking `Promise.all`)

### Page: Browse Notes (Search)
- Full-text + semantic vector search
- Filters: college dropdown, department dropdown, semester
- Dropdowns populated **dynamically from MongoDB** (no hardcoded values)
- Note cards with Study with AI button

### Page: NoteBot AI
- Full chat interface with message history (last 12 messages sent as context)
- Supports markdown rendering (bold, lists, tables, flashcards)
- **MCP Activity Feed**: real-time log of tool calls showing name, args, timestamp, and whether it used MCP
- Suggested question chips
- "Study with AI" from any note card → pre-loads note into chat with friendly welcome message (not auto-generating, just loading context)

### Page: Upload Note
- Manual form upload (title, subject, college, dept, semester, chapter, teacher, content)
- Image OCR upload: camera icon → uploads image → Gemini OCR extracts + classifies → auto-fills form
- OCR overlay with animated timer + rotating study trivia while processing
- **Client-side image compression**: images >2MB are compressed to ~1MB via HTML5 Canvas before upload (max 2500px dimension, 85% quality — non-destructive for OCR accuracy)
- Earns +50 points (text) or +100 points (OCR)
- Tips panel with "Points Breakdown"

### Page: Leaderboard
- Global leaderboard + College-filtered leaderboard side by side
- Medal badges for Top 3 (🥇🥈🥉)
- Filter by college name
- Live from MongoDB

### Page: College Forum
- Create posts with title, content, college, tag
- Upvote posts with **optimistic UI** (counter increments instantly)
- Filter by college dropdown (dynamic from DB)
- "Ask NoteBot" button on each post → sends post topic to NoteBot

### Page: Request Board
- Browse open note requests
- Post new requests (subject, semester, dept, priority)
- Fulfill a request (+100 points)
- Filter by college

### Page: My Profile
- Avatar with initials
- Points balance, global rank, uploads, downloads
- Badges display
- Edit profile (name, college, dept, semester)
- "My Uploaded Notes" list with quality scores
- Study streak tracking
- "🚨 Emergency Exam Mode" button → calls `/api/exam-mode` → shows 24-hr plan in NoteBot

---

## 🎯 Gamification System

| Action | Points |
|---|---|
| Upload a note (manual) | +50 pts |
| Upload via OCR (image) | +100 pts |
| Post a comment | +5 pts |
| Upvote a note | +1 pt (upvoter) |
| Fulfill a note request | +100 pts |
| Badge system | Stored in `users.badges` array |

---

## ⚡ Performance Optimizations (Added in This Session)

### Database (MongoDB Atlas)
- **Compound Index** on `notes(college, department, semester)` — queries drop from 50ms → 2ms
- **Exact match queries** replace regex scans — all `$regex` on college/dept/forum/leaderboard/requests replaced with equality checks
- **Strict projections** on all feed queries — `embedding` and `full_content` excluded from `/api/notes`, `/api/leaderboard` etc. (payload reduced by ~90%)

### Frontend UX
- **Client-side image compression** (Canvas API) before OCR upload — 10MB → ~1MB, non-aggressive (85% quality, 2500px max)
- **Optimistic UI upvotes** on note cards — counter increments instantly, reverts silently on failure
- **Optimistic UI upvotes** on forum posts — same pattern
- **Optimistic Zero-State** for new users — `0` painted instantly on onboarding submit, bypassing API wait
- **Independent async fetches** in `loadHome()` — all 4 dashboard requests fire concurrently, each updates DOM independently (no `Promise.all` blocking)
- **Cold-start retry** — home feed fails fast (15s timeout), shows "⏳ Waking up server...", auto-retries once after 4 seconds
- **API timeout** reduced from 30s → 15s — prevents silent 30-second hangs

---

## 🔧 Technical Architecture Highlights

### Agentic Loop (main.py)
```
User message → ChatRequest → Gemini 2.5 Flash
  ↓ (if function_call in response)
  → dispatch_tool() [async]
    → tool_search_notes_mcp() → MCP → MongoDB Atlas $vectorSearch
    → fallback: PyMongo direct
  → feed result back to Gemini
  → repeat up to 8 iterations
  → return final text + tool_calls_log
```

### MCP Architecture
```
FastAPI (main.py)
  └─ mcp_bridge.py (AsyncExitStack)
       └─ stdio_client → @mongodb-js/mongodb-mcp-server (subprocess)
            └─ MongoDB Atlas
```

### OCR Pipeline
```
Image upload (multipart/form-data)
  → Client: Canvas compress if >2MB
  → Backend: Gemini 2.5 Flash OCR (JSON mode)
  → Regex sanitizer: escape invalid Unicode (\uXXXX, dangling \)
  → JSON parse with fallback regex extraction
  → rules_engine quality score
  → text-embedding-005 vectorize
  → MongoDB Atlas insert
  → +100 points
```

### Dynamic Filter System
- `/api/colleges` → `notes_col().distinct("college")` → sorted, empty-filtered
- `/api/departments` → `notes_col().distinct("department")` → sorted, empty-filtered
- Frontend calls these at startup → populates all dropdowns across Home, Browse, Forum, Requests, Upload
- No hardcoded college/department values anywhere

---

## 🧪 Test Suite (`/tests`)

| File | Tests |
|---|---|
| `test_all.py` | Full integration test suite |
| `test_chat_memory.py` | Academic memory persistence tests |
| `test_gemini.py` | Gemini API connection test |
| `test_ocr.py` | OCR extraction unit tests |
| `test_ocr_api.py` | OCR API endpoint tests |
| `test_regex.py` | JSON sanitizer regex tests |
| `test_rules.py` | Quality scoring rubric tests |
| `test_system_rigorous.py` | End-to-end system test |

---

## 🔒 Session & Auth

- **No server-side auth** (by design — hackathon MVP)
- **localStorage** session: `notely_user` JSON blob contains `{user_id, name, college, semester, department}`
- `user_id` = `student_` + `Date.now().toString(36)` (generated on onboarding)
- Onboarding modal shown on first visit; skipped on return visits
- `applyUserUI()` pre-fills college/dept into all form inputs on login

---

## 📡 OpenAPI / Agent Builder Integration

- Custom OpenAPI 3.0.3 schema generated via `custom_openapi()`
- Schema tuned for **Google AI Agent Builder** compatibility
- Server URL points to Cloud Run production endpoint
- MCP endpoint (`/mcp`) accepts JSON-RPC 2.0 messages
- Supported methods: `initialize`, `tools/list`, `tools/call`
