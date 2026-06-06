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
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request
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
    get_learning_patterns,
    get_platform_stats,
    get_user_memory,
)
from agent_tools import dispatch_tool
from rules_engine import evaluate_note_quality
from mcp_bridge import mcp_router, start_mcp, stop_mcp
from contextlib import asynccontextmanager

# ── Vertex AI init (guarded) ──────────────────────────────────────────────────
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
REGION  = os.getenv("VERTEX_REGION", "us-central1")
if PROJECT:
    vertexai.init(project=PROJECT, location=REGION)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_mcp()
    yield
    await stop_mcp()

from fastapi.openapi.utils import get_openapi

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Notely Agentic OS",
    description="Backend API for Notely, an AI-powered academic memory platform. Connects to Google AI Agent Builder for real-time tool execution, vector search, and student interactions.",
    version="2.0.0",
    lifespan=lifespan,
    servers=[{"url": "https://notely-api.run.app", "description": "Google Cloud Run Production"}]
)

def custom_openapi():
    """Generates an OpenAPI 3.0 schema strictly optimized for Google AI Agent Builder."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version="3.0.3",
        description=app.description,
        routes=app.routes,
        servers=app.servers,
    )
    # Agent builder needs a clear info block and strictly typed responses
    openapi_schema["info"]["x-logo"] = {"url": "https://cdn-icons-png.flaticon.com/512/3143/3143641.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.include_router(mcp_router)

# ══════════════════════════════════════════════════════════════════════════════
#  MCP STREAMABLE HTTP TRANSPORT — Required by Google AI Studio Agent Builder
#  Agent Builder expects a proper SSE/streamable endpoint, not just REST.
#  This endpoint translates MCP JSON-RPC messages to our MongoDB MCP bridge.
# ══════════════════════════════════════════════════════════════════════════════

from fastapi.responses import StreamingResponse
import json as _json_module

@app.get("/mcp/health")
async def mcp_health():
    """Google Agent Builder probes this before connecting."""
    from mcp_bridge import mcp_ctx
    return {
        "status": "ok",
        "mcp_connected": mcp_ctx.session is not None,
        "protocol": "MCP/1.0",
        "transport": "streamable-http",
        "service": "Notely MongoDB MCP Bridge",
    }

@app.post("/mcp")
async def mcp_streamable_endpoint(request: Request):
    """
    Streamable HTTP MCP transport for Google AI Studio Agent Builder.
    Accepts MCP JSON-RPC messages, routes them to the MongoDB MCP server,
    and returns the response in MCP protocol format.
    """
    from mcp_bridge import mcp_ctx
    try:
        body = await request.json()
    except Exception:
        return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}

    method  = body.get("method", "")
    params  = body.get("params", {})
    req_id  = body.get("id", 1)

    # ── tools/list ────────────────────────────────────────────────────────────
    if method == "tools/list":
        if mcp_ctx.session:
            try:
                resp = await mcp_ctx.session.list_tools()
                tools = [
                    {"name": t.name, "description": t.description,
                     "inputSchema": t.inputSchema}
                    for t in resp.tools
                ]
                return {"jsonrpc": "2.0", "id": req_id,
                        "result": {"tools": tools}}
            except Exception as e:
                return {"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32000, "message": str(e)}}
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": []}}

    # ── tools/call ────────────────────────────────────────────────────────────
    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments  = params.get("arguments", {})
        if mcp_ctx.session:
            try:
                result = await mcp_ctx.session.call_tool(tool_name, arguments=arguments)
                content = [{"type": "text", "text": c.text}
                           for c in result.content if c.type == "text"]
                return {"jsonrpc": "2.0", "id": req_id,
                        "result": {"content": content, "isError": result.isError}}
            except Exception as e:
                return {"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32000, "message": str(e)}}
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32001, "message": "MCP session not initialized"}}

    # ── initialize (handshake) ────────────────────────────────────────────────
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "Notely MongoDB MCP Bridge",
                    "version": "2.0.0"
                }
            }
        }

    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}}

@app.get("/api/platform-stats")
async def api_platform_stats():
    """Fetch live counts of notes, users, and open requests."""
    try:
        stats = get_platform_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    FunctionDeclaration(
        name="record_study_session",
        description="Record what the student just studied to update their study history",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic studied (e.g. Binary Trees)"},
            },
            "required": ["topic"],
        },
    ),
    FunctionDeclaration(
        name="add_learning_pattern",
        description="Log a learning pattern or observation about how the student learns",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The pattern (e.g. 'Struggles with visualising graphs', 'Prefers step-by-step math')"},
            },
            "required": ["pattern"],
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
- When a student asks for a summary, flashcards, or quiz for a SPECIFIC note, use the appropriate tool (e.g., generate_quiz) with that note_id.
- CRITICAL: If the student asks you to summarise, create flashcards, or generate a test for "ALL notes in this chat" or "from this chat", DO NOT call any tools! Simply use the information already present in the chat history to generate the summary, flashcards, or quiz directly! Combine the topics seamlessly.
- ACT AS A PERSONAL ACADEMIC OS: If you see the student's Academic Memory, actively recommend reviewing weak topics (offer notes/flashcards) and praise them for strong topics!
- ALWAYS evaluate quizzes when the student answers them, and call record_quiz_score to update their memory!
- Notice patterns in the student's behavior and record them via add_learning_pattern. Record study sessions using record_study_session."""


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

class ExamModeRequest(BaseModel):
    user_id:    str = "anonymous"
    college:    str = ""
    semester:   str = "5"
    department: str = ""
    subject:    str = ""   # optional fallback if no weak topics stored


# ══════════════════════════════════════════════════════════════════════════════
#  EMERGENCY EXAM MODE
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/exam-mode")
async def exam_mode(req: ExamModeRequest):
    """
    One-click Emergency Exam Mode.
    1. Fetches the student's weak_topics from MongoDB user_memory.
    2. For each weak topic (up to 4), searches for the best notes via MCP.
    3. Passes all results to Gemini to produce a timed, prioritised study plan.
    4. Returns the plan + a tool_calls log for the Activity Feed.
    """
    from agent_tools import tool_search_notes_mcp
    from datetime import datetime as _dt

    tool_calls_log = []

    # Step 1 ── Pull weak topics from MongoDB
    memory = get_user_memory(req.user_id)
    weak_topics = memory.get("weak_topics", [])
    strong_topics = memory.get("strong_topics", [])

    # If no weak topics recorded yet, fall back to the subject hint or generic
    if not weak_topics:
        weak_topics = [req.subject] if req.subject else ["core exam topics"]
        no_memory = True
    else:
        no_memory = False
        weak_topics = weak_topics[:4]   # cap at 4 to avoid long wait

    tool_calls_log.append({
        "name":      "get_user_memory",
        "args":      f"user_id={req.user_id}",
        "timestamp": _dt.utcnow().strftime("%H:%M:%S"),
        "via_mcp":   False,
    })

    # Step 2 ── Search best notes for each weak topic
    note_blobs = []
    for topic in weak_topics:
        ts = _dt.utcnow().strftime("%H:%M:%S")
        try:
            result_str, via_mcp = await tool_search_notes_mcp(
                query=topic,
                college=req.college,
                semester=req.semester,
                department=req.department,
            )
        except Exception:
            result_str, via_mcp = f"No notes found for {topic}.", False

        tool_calls_log.append({
            "name":      "search_notes",
            "args":      f"query={topic[:35]}, college={req.college[:20]}",
            "timestamp": ts,
            "via_mcp":   via_mcp,
        })
        note_blobs.append(f"### Topic: {topic}\n{result_str}")

    # Step 3 ── Ask Gemini to synthesise a timed study plan
    try:
        from vertexai.generative_models import GenerativeModel as _GM, Part as _Part
        planner = _GM(
            "gemini-2.5-flash",
            system_instruction=(
                "You are an emergency exam coach for an Indian college student. "
                "Given their weak topics and the notes available in their platform, "
                "produce a STRICT, timed 24-hour study plan formatted EXACTLY as:\n"
                "## 🚨 24-Hour Emergency Exam Plan\n"
                "**Strong Topics (skip or skim):** <comma list>\n"
                "**Weak Topics (focus):** <comma list>\n\n"
                "| Time Slot | Topic | Action | Notes Available |\n"
                "|-----------|-------|--------|-----------------|\n"
                "| 06:00–07:30 | ... | ... | ... |\n"
                "... (cover 24 hrs with realistic slots including breaks)\n\n"
                "**Final 2-hr Blitz:** bullet list of the single most important concept per weak topic.\n"
                "Keep it punchy, motivating, and achievable. Max 600 words."
            ),
        )
        prompt = (
            f"Student: {req.college}, Semester {req.semester}, {req.department}\n"
            f"Strong topics (already knows): {', '.join(strong_topics) or 'none recorded'}\n"
            f"Weak topics (needs focus): {', '.join(weak_topics)}\n"
            f"No prior memory: {no_memory}\n\n"
            "Available notes found for each weak topic:\n"
            + "\n\n".join(note_blobs)
        )
        response = planner.generate_content(prompt)
        plan_text = response.text

        tool_calls_log.append({
            "name":      "generate_exam_plan",
            "args":      f"topics={len(weak_topics)}, notes_blobs={len(note_blobs)}",
            "timestamp": _dt.utcnow().strftime("%H:%M:%S"),
            "via_mcp":   False,
        })

    except Exception as e:
        plan_text = (
            f"## 🚨 Emergency Exam Plan\n"
            f"**Focus on these weak topics:** {', '.join(weak_topics)}\n\n"
            "Could not generate a full AI plan right now. "
            "Study each weak topic for 2 hours using the notes found above, "
            "then do 30-minute revision. Good luck! 💪"
        )

    return {
        "success":     True,
        "plan":        plan_text,
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "no_memory":   no_memory,
        "tool_calls":  tool_calls_log,
    }


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

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        academic_memory = ""
        if req.user_id != "anonymous":
            user = get_user_by_id(req.user_id)
            if user:
                weak = ", ".join(user.get("weak_topics", [])) or "None yet"
                strong = ", ".join(user.get("strong_topics", [])) or "None yet"
                last = user.get("last_studied", "Nothing yet")
                patterns = get_learning_patterns(req.user_id)
                pattern_str = "\\n- Patterns: " + ", ".join(patterns) if patterns else ""
                
                academic_memory = (
                    f"\nAcademic Memory for User:\n"
                    f"- Weak Topics: {weak}\n"
                    f"- Strong Topics: {strong}\n"
                    f"- Last Studied: {last}{pattern_str}\n"
                    f"(Use this to recommend targeted study sessions and adapt to their patterns!)"
                )

        system = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Student context — College: '{req.college}', "
            f"Semester: '{req.semester}', Department: '{req.department}'. "
            "Use as default filters when the student doesn't specify."
            f"{academic_memory}"
        )

        # Stable model
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

        # Collect tool calls for the MCP Activity Feed visible in the frontend
        from datetime import datetime as _dt
        tool_calls_log = []

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

            # Log this tool call for the Activity Feed
            arg_summary = ", ".join(
                f"{k}={str(v)[:40]}" for k, v in tool_args.items() if v
            ) or "no params"

            # dispatch_tool is now async and returns (result_str, via_mcp: bool)
            tool_result, via_mcp = await dispatch_tool(
                tool_name, tool_args, user_id=req.user_id
            )

            tool_calls_log.append({
                "name":      tool_name,
                "args":      arg_summary,
                "timestamp": _dt.utcnow().strftime("%H:%M:%S"),
                "via_mcp":   via_mcp,
            })

            response = chat_session.send_message(
                Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_result},
                )
            )

        return {"response": response.text, "success": True, "tool_calls": tool_calls_log}

    except Exception as e:
        print(f"[/chat] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  NOTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload-note")
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
        
        # Apply Rule-Based Scoring
        eval_data = evaluate_note_quality(note.full_content)
        note_dict["quality_score"] = eval_data["score"]
        note_dict["strengths"] = eval_data["strengths"]
        note_dict["weaknesses"] = eval_data["weaknesses"]

        note_id = insert_note(note_dict)
        update_user_points(note.uploaded_by, 50)

        return {"note_id": note_id, "message": "Note uploaded and embedded! +50 pts"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import json

@app.post("/api/ocr-note")
async def ocr_note(
    file: UploadFile = File(...),
    uploaded_by: str = Form("anonymous"),
    college: str = Form(""),
    department: str = Form("")
):
    """
    Accepts an image, sends to Gemini 2.5 Flash for OCR, auto-classifies subject/topic/semester,
    and automatically uploads the extracted note.
    """
    try:
        file_bytes = await file.read()
        
        model = GenerativeModel("gemini-2.5-flash")
        prompt = '''Analyze this handwritten or typed note.
Extract the raw text comprehensively.
Also, auto-classify it into a Subject, Topic (chapter), Semester (1-8), and an appropriate Title.
Respond ONLY in valid JSON format exactly like this:
{
  "content": "extracted text...",
  "subject": "e.g. Engineering Mathematics",
  "chapter": "e.g. Fourier Series",
  "semester": 5,
  "title": "e.g. Fourier Transforms Overview"
}'''
        
        response = model.generate_content([
            Part.from_data(data=file_bytes, mime_type=file.content_type),
            prompt
        ])
        
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:-3]
        elif text.startswith("```"): text = text[3:-3]
        
        parsed = json.loads(text.strip())
        
        note_dict = {
            "title": parsed.get("title", "Untitled OCR Note"),
            "subject": parsed.get("subject", "Unknown Subject"),
            "chapter": parsed.get("chapter", "Unknown Chapter"),
            "semester": int(parsed.get("semester", 1)),
            "college": college,
            "department": department,
            "uploaded_by": uploaded_by,
            "full_content": parsed.get("content", ""),
            "upvotes": 0,
            "downloads": 0,
        }
        
        # Apply Rule-Based Scoring
        eval_data = evaluate_note_quality(note_dict["full_content"])
        note_dict["quality_score"] = eval_data["score"]
        note_dict["strengths"] = eval_data["strengths"]
        note_dict["weaknesses"] = eval_data["weaknesses"]
        
        embed_text = f"{note_dict['title']} {note_dict['subject']} {note_dict['chapter']} {note_dict['full_content'][:3000]}"
        note_dict["embedding"] = get_embedding(embed_text)
        note_dict["content_preview"] = note_dict["full_content"][:500]

        note_id = insert_note(note_dict)
        update_user_points(uploaded_by, 100) # Give 100 points for OCR
        
        return {
            "note_id": note_id, 
            "message": f"OCR successful! Classified as {note_dict['subject']} - {note_dict['chapter']}. Note embedded +100 pts", 
            "extracted": parsed
        }

    except Exception as e:
        print(f"[/api/ocr-note] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from database import notes_col

@app.get("/api/colleges")
async def list_colleges():
    """Fetch distinct colleges from the database for dynamic dropdowns."""
    try:
        colleges = notes_col().distinct("college")
        # Filter out empty strings and sort alphabetically
        colleges = sorted([c for c in colleges if str(c).strip()])
        return {"success": True, "colleges": colleges}
    except Exception as e:
        return {"success": False, "colleges": ["VNIT Nagpur", "IIT Bombay", "COEP Pune", "NIT Warangal", "IIT Delhi"]}


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
    elif sort == "trending":
        notes = get_trending_notes(college=college, limit=limit)
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