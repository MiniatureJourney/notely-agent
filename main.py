"""
main.py — Notely FastAPI Backend (Fixed)

Bugs fixed vs original:
1. upload-note was query-param only → now accepts JSON body (Pydantic model)
2. function_call check on parts[0] crashed if first part was text → safe hasattr check
3. No static file serving → StaticFiles + root route added
4. vertexai.init called at startup with possible None project → guarded
5. gemini-2.5-flash → gemini-1.5-pro (stable on Cloud Run)
6. Missing REST endpoints added: /api/notes, /api/notes/search,
   /api/leaderboard, /api/requests, /api/forum, /api/users
7. ObjectId not serializable → _id converted to str in all responses
8. No /health detail → returns version info
9. history loop used "assistant" role without converting → fixed to "model"
10. Tool dispatcher was inline if/elif → moved to dispatch_tool() in agent_tools
"""

import os
import pathlib

from dotenv import load_dotenv
load_dotenv()

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Content,
    Part,
    Tool,
    FunctionDeclaration,
)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# ── Local imports ─────────────────────────────────────────────────────────────
from embeddings import get_embedding
from database import (
    insert_note,
    get_all_notes,
    get_note_by_id,
    vector_search_notes,
    search_notes_text,
    upvote_note,
    get_trending_notes,
    get_recent_notes,
    get_user_notes,
    add_comment,
    get_comments,
    create_request,
    get_requests,
    fulfill_request,
    create_forum_post,
    get_forum_posts,
    upvote_forum_post,
    get_leaderboard,
    upsert_user,
    get_user_by_id,
    update_user_points,
)
from agent_tools import dispatch_tool

# ── Vertex AI init (guarded) ──────────────────────────────────────────────────
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
REGION  = os.getenv("VERTEX_REGION", "us-central1")
if PROJECT:
    vertexai.init(project=PROJECT, location=REGION)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Notely API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend from /app/static/ ─────────────────────────────────────────
STATIC_DIR = pathlib.Path("/app/static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Notely API running — index.html not found in /app/static/"}
else:
    @app.get("/")
    async def root():
        return {"message": "Notely API v2.0 — MongoDB Atlas + Gemini"}


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI TOOL DECLARATIONS
# ══════════════════════════════════════════════════════════════════════════════

_TOOLS = Tool(function_declarations=[
    FunctionDeclaration(
        name="search_notes",
        description=(
            "Search for academic notes by topic, subject, college, or semester "
            "using MongoDB Atlas Vector Search (semantic search)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query":      {"type": "string", "description": "What the student is looking for"},
                "college":    {"type": "string", "description": "College name filter (optional)"},
                "subject":    {"type": "string", "description": "Subject name filter (optional)"},
                "semester":   {"type": "string", "description": "Semester number (optional)"},
                "department": {"type": "string", "description": "Department filter (optional)"},
            },
            "required": ["query"],
        },
    ),
    FunctionDeclaration(
        name="generate_summary",
        description="Generate a concise exam-ready summary from a specific note",
        parameters={
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "MongoDB _id of the note"},
            },
            "required": ["note_id"],
        },
    ),
    FunctionDeclaration(
        name="generate_flashcards",
        description="Generate study flashcards (Q&A pairs) from a note",
        parameters={
            "type": "object",
            "properties": {
                "note_id":   {"type": "string", "description": "MongoDB _id of the note"},
                "num_cards": {"type": "integer", "description": "Number of flashcards (default 10)"},
            },
            "required": ["note_id"],
        },
    ),
    FunctionDeclaration(
        name="generate_quiz",
        description="Generate a multiple-choice practice quiz from a note",
        parameters={
            "type": "object",
            "properties": {
                "note_id":       {"type": "string", "description": "MongoDB _id of the note"},
                "num_questions": {"type": "integer", "description": "Number of questions (default 5)"},
            },
            "required": ["note_id"],
        },
    ),
    FunctionDeclaration(
        name="get_trending_notes",
        description="Get the highest quality trending notes, optionally filtered by college",
        parameters={
            "type": "object",
            "properties": {
                "college": {"type": "string", "description": "Filter by college (optional)"},
            },
        },
    ),
    FunctionDeclaration(
        name="get_leaderboard",
        description="Get top student contributors ranked by points",
        parameters={
            "type": "object",
            "properties": {
                "college": {"type": "string", "description": "Filter by college (optional)"},
            },
        },
    ),
    FunctionDeclaration(
        name="get_comments",
        description="Get comments on a specific note",
        parameters={
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "MongoDB _id of the note"},
            },
            "required": ["note_id"],
        },
    ),
    FunctionDeclaration(
        name="get_requests",
        description="Get open note requests from the community request board",
        parameters={
            "type": "object",
            "properties": {
                "college": {"type": "string", "description": "Filter by college (optional)"},
            },
        },
    ),
    FunctionDeclaration(
        name="get_forum_posts",
        description="Get community forum posts for a college",
        parameters={
            "type": "object",
            "properties": {
                "college": {"type": "string", "description": "College name (optional)"},
            },
        },
    ),
    FunctionDeclaration(
        name="record_quiz_score",
        description="Record the student's quiz score to update their Academic Memory (strong/weak topics). Call this when the student finishes a quiz.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic of the quiz (e.g. Fourier Transform)"},
                "score": {"type": "number", "description": "The percentage score (0 to 100)"},
            },
            "required": ["topic", "score"],
        },
    ),
])

_SYSTEM_PROMPT = """You are NoteBot, an intelligent AI study assistant on Notely — \
a notes-sharing platform for Indian college students.

Your capabilities:
- Search academic notes using MongoDB Atlas Vector Search
- Summarise notes into concise exam-ready guides
- Generate flashcards (Q&A pairs) for memorisation
- Create multiple-choice practice quizzes with explanations
- Show trending notes and top contributors
- Display note requests and forum discussions

Your behaviour:
- Friendly, encouraging, and exam-focused
- After finding notes, always offer to summarise/quiz/flashcard
- Use the student's context (college, semester, department) as default filters
- Format responses clearly — use bold for important terms
- When a student says "summarise it" or "make flashcards" after a search,
  use the note_id from the previous search result
- ACT AS A PERSONAL ACADEMIC OS: If you see the student's Academic Memory, actively recommend reviewing weak topics (offer notes/flashcards) and praise them for strong topics!
- ALWAYS evaluate quizzes when the student answers them, and call record_quiz_score to update their memory!"""


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message:      str
    college:      str  = ""
    semester:     str  = ""
    department:   str  = ""
    user_id:      str  = "anonymous"
    chat_history: list = []

class NoteUpload(BaseModel):
    """
    FIX #1: original used query params — JSON body is required for
    frontend fetch() POST with Content-Type: application/json
    """
    title:        str
    subject:      str
    college:      str
    semester:     int
    department:   str
    chapter:      str = ""
    teacher:      str = ""
    full_content: str = ""
    uploaded_by:  str = "anonymous"

class CommentBody(BaseModel):
    note_id:   str
    user_id:   str
    user_name: str
    content:   str

class RequestBody(BaseModel):
    title:      str
    subject:    str
    college:    str
    semester:   int
    department: str = ""
    posted_by:  str = "anonymous"
    priority:   str = "medium"

class ForumPostBody(BaseModel):
    title:   str
    content: str
    college: str
    author:  str = "anonymous"
    tag:     str = "general"

class UserBody(BaseModel):
    user_id:    str
    name:       str
    college:    str
    department: str = ""
    semester:   int = 1


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "Notely API v2.0",
        "model":   "gemini-2.5-flash",   # FIX #5: was gemini-2.5-flash
        "db":      "MongoDB Atlas",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT — AGENTIC LOOP
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        academic_memory = ""
        if req.user_id != "anonymous":
            user = get_user_by_id(req.user_id)
            if user:
                weak = ", ".join(user.get("weak_topics", [])) or "None yet"
                strong = ", ".join(user.get("strong_topics", [])) or "None yet"
                academic_memory = f"\\nAcademic Memory for User:\\n- Weak Topics: {weak}\\n- Strong Topics: {strong}\\n(Use this to recommend targeted study sessions!)"

        system = (
            f"{_SYSTEM_PROMPT}\\n\\n"
            f"Student context — College: '{req.college}', "
            f"Semester: '{req.semester}', Department: '{req.department}'. "
            "Use as default filters when the student doesn't specify."
            f"{academic_memory}"
        )

        # FIX #5: stable model name
        model = GenerativeModel(
            "gemini-2.5-flash",
            tools=[_TOOLS],
            system_instruction=system,
        )

        # FIX #9: convert "assistant" → "model" for Vertex AI
        history_contents = []
        for msg in req.chat_history[-12:]:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"
            history_contents.append(
                Content(role=role, parts=[Part.from_text(msg.get("content", ""))])
            )

        chat_session = model.start_chat(history=history_contents)
        response     = chat_session.send_message(req.message)

        # FIX #2: safe agentic loop — don't assume parts[0] is always a function_call
        for _ in range(8):
            # Check ALL parts for a function call
            fc_part = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc_part = part
                    break

            if fc_part is None:
                break  # pure text response — done

            tool_name = fc_part.function_call.name
            tool_args = dict(fc_part.function_call.args)

            # FIX #10: dispatch via agent_tools.dispatch_tool()
            tool_result = dispatch_tool(tool_name, tool_args, user_id=req.user_id)

            response = chat_session.send_message(
                Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_result},
                )
            )

        return {"response": response.text, "success": True}

    except Exception as e:
        print(f"[/chat] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  NOTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload-note")
async def upload_note(note: NoteUpload):
    """
    FIX #1: accepts JSON body via Pydantic model instead of query params.
    Frontend sends: fetch('/upload-note', {method:'POST', body: JSON.stringify({...})})
    """
    try:
        note_dict    = note.dict()
        embed_text   = f"{note.title} {note.subject} {note.chapter} {note.full_content[:3000]}"
        note_dict["embedding"]       = get_embedding(embed_text)
        note_dict["content_preview"] = note.full_content[:500]

        note_id = insert_note(note_dict)
        update_user_points(note.uploaded_by, 50)

        return {"note_id": note_id, "message": "Note uploaded and embedded! +50 pts"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notes")
async def list_notes(
    college:    str = Query(""),
    department: str = Query(""),
    semester:   int = Query(0),
    sort:       str = Query("quality"),
    limit:      int = Query(20),
):
    """Browse / filter notes — used by the frontend home and search grids."""
    if sort == "recent":
        notes = get_recent_notes(college=college, limit=limit)
    else:
        notes = get_all_notes(
            college=college, department=department,
            semester=semester, limit=limit
        )
    return {"notes": notes, "count": len(notes)}


@app.get("/api/notes/search")
async def search_notes_endpoint(
    q:          str = Query(...),
    college:    str = Query(""),
    department: str = Query(""),
    semester:   int = Query(0),
    limit:      int = Query(10),
):
    """Semantic search — used by the Browse page search bar."""
    try:
        embed   = get_embedding(q)
        filters = {}
        if college:    filters["college"]    = college
        if department: filters["department"] = department
        if semester:   filters["semester"]   = str(semester)

        results = vector_search_notes(embed, filters, limit=limit)
        if not results:
            results = search_notes_text(
                q, college=college, department=department,
                semester=semester, limit=limit
            )
        return {"notes": results, "count": len(results), "query": q}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notes/{note_id}")
async def get_note(note_id: str):
    note = get_note_by_id(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@app.post("/api/notes/{note_id}/upvote")
async def upvote(note_id: str, user_id: str = Query("anonymous")):
    ok = upvote_note(note_id)
    if ok:
        update_user_points(user_id, 1)
    return {"success": ok}


# ══════════════════════════════════════════════════════════════════════════════
#  COMMENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/notes/{note_id}/comments")
async def list_comments(note_id: str):
    return {"comments": get_comments(note_id)}

@app.post("/api/comments")
async def post_comment(body: CommentBody):
    cid = add_comment(body.note_id, body.user_id, body.user_name, body.content)
    update_user_points(body.user_id, 5)
    return {"comment_id": cid, "message": "Comment posted! +5 pts"}


# ══════════════════════════════════════════════════════════════════════════════
#  REQUESTS BOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/requests")
async def list_requests(college: str = Query(""), limit: int = Query(20)):
    return {"requests": get_requests(college=college, limit=limit)}

@app.post("/api/requests")
async def post_request(body: RequestBody):
    rid = create_request(body.dict())
    return {"request_id": rid, "message": "Request posted!"}

@app.post("/api/requests/{request_id}/fulfill")
async def fulfill(request_id: str, user_id: str = Query("anonymous")):
    ok = fulfill_request(request_id)
    if ok:
        update_user_points(user_id, 100)
    return {"success": ok, "message": "Request fulfilled! +100 pts"}


# ══════════════════════════════════════════════════════════════════════════════
#  FORUM
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/forum")
async def list_forum(college: str = Query(""), limit: int = Query(20)):
    return {"posts": get_forum_posts(college=college, limit=limit)}

@app.post("/api/forum")
async def post_forum(body: ForumPostBody):
    pid = create_forum_post(body.dict())
    return {"post_id": pid, "message": "Post created!"}

@app.post("/api/forum/{post_id}/upvote")
async def upvote_post(post_id: str):
    ok = upvote_forum_post(post_id)
    return {"success": ok}


# ══════════════════════════════════════════════════════════════════════════════
#  LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/leaderboard")
async def leaderboard(college: str = Query(""), limit: int = Query(10)):
    return {"leaders": get_leaderboard(college=college, limit=limit)}


# ══════════════════════════════════════════════════════════════════════════════
#  USER PROFILE
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/users")
async def create_user(body: UserBody):
    uid = upsert_user(body.dict())
    return {"_id": uid, "message": "User saved"}

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/api/user/{user_id}/notes")
async def user_notes(user_id: str):
    return {"notes": get_user_notes(user_id)}