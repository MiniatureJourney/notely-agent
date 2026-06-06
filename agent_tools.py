"""
agent_tools.py — All tools the Gemini agent can call

Key architecture changes in this version:
- search_notes now routes through the official MongoDB MCP Server
  using an async $vectorSearch aggregation pipeline call.
  Falls back to direct PyMongo if the MCP server is unavailable.
- dispatch_tool() is now async so it can await the MCP search.
- dispatch_tool() returns (result_str, via_mcp: bool) so main.py
  can tag Activity Feed entries with whether they used MCP.
- All other tools remain synchronous and are called normally.
"""

import json as _json

from database import (
    vector_search_notes,
    get_note_full_content,
    get_trending_notes,
    get_leaderboard,
    get_comments,
    get_requests,
    get_forum_posts,
)
from database import update_academic_memory, record_study_session, add_learning_pattern
from embeddings import get_embedding, ZERO_VECTOR
from datetime import datetime


# ── Helper ────────────────────────────────────────────────────────────────────

def _fmt_date(dt) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%d %b %Y")
    return str(dt) if dt else ""


def _format_note_results(results: list, query: str) -> str:
    """Shared formatter for both MCP and PyMongo search results."""
    if not results:
        return (
            f"No notes found for '{query}'. "
            "Try broader keywords or remove filters. "
            "You can also post a request on the Note Request Board!"
        )
    lines = [f"Found **{len(results)} notes** for '{query}':\n"]
    for i, note in enumerate(results, 1):
        note_id = str(note.get("_id", ""))
        preview = note.get("content_preview", "")[:300]
        score   = f" · Relevance: {note.get('score', 0):.2f}" if note.get("score") else ""
        lines.append(
            f"**{i}. {note.get('title', 'Untitled')}**\n"
            f"   Subject: {note.get('subject','?')} | "
            f"Sem {note.get('semester','?')} | "
            f"{note.get('college','?')}\n"
            f"   Teacher: {note.get('teacher','Unknown')} | "
            f"⬆ {note.get('upvotes', 0)} upvotes | "
            f"Quality: {note.get('quality_score', 0)}{score}\n"
            f"   Note ID: `{note_id}`\n"
            f"   Preview: {preview}...\n"
        )
    lines.append(
        "Would you like me to **summarise** one, "
        "make **flashcards**, or run a **practice quiz**? "
        "Just say the number!"
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — Search Notes (PyMongo fallback — used when MCP is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def tool_search_notes(query: str, college: str = "", subject: str = "",
                      semester: str = "", department: str = "") -> str:
    """Direct PyMongo path — used as fallback when MCP session is unavailable."""
    query_embedding = get_embedding(query)

    filters = {}
    if college:    filters["college"]    = college
    if subject:    filters["subject"]    = subject
    if semester:   filters["semester"]   = semester
    if department: filters["department"] = department

    results = vector_search_notes(query_embedding, filters, limit=5)
    return _format_note_results(results, query)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1-MCP — Search Notes via Official MongoDB MCP Server (PRIMARY PATH)
# ══════════════════════════════════════════════════════════════════════════════

async def tool_search_notes_mcp(query: str, college: str = "", subject: str = "",
                                 semester: str = "", department: str = "") -> tuple:
    """
    PRIMARY search path — executes a $vectorSearch aggregation pipeline
    through the official @mongodb-js/mongodb-mcp-server via MCP protocol.

    Returns (result_string, via_mcp: bool):
      via_mcp=True  — query ran through MCP (authentic partner integration)
      via_mcp=False — fell back to PyMongo (MCP session unavailable)
    """
    from mcp_bridge import mcp_ctx

    # Get Vertex AI embedding
    query_embedding = get_embedding(query)

    # If Vertex AI embedding failed, skip vector search
    if not query_embedding or all(x == 0.0 for x in query_embedding):
        return tool_search_notes(query, college, subject, semester, department), False

    # Build $vectorSearch pre-filter (only non-empty values)
    pre_filter: dict = {}
    if college:    pre_filter["college"]    = college
    if subject:    pre_filter["subject"]    = subject
    if department: pre_filter["department"] = department
    if semester:
        try:
            pre_filter["semester"] = {"$eq": int(semester)}
        except (ValueError, TypeError):
            pass

    # Build $vectorSearch stage
    vector_stage: dict = {
        "index":         "notes_vector_index",
        "path":          "embedding",
        "queryVector":   query_embedding,
        "numCandidates": 100,
        "limit":         5,
    }
    if pre_filter:
        vector_stage["filter"] = pre_filter

    pipeline = [
        {"$vectorSearch": vector_stage},
        {"$project": {
            "_id":             1,
            "title":           1,
            "subject":         1,
            "college":         1,
            "semester":        1,
            "department":      1,
            "teacher":         1,
            "upvotes":         1,
            "quality_score":   1,
            "content_preview": 1,
            "score":           {"$meta": "vectorSearchScore"},
        }},
    ]

    # ── Attempt MCP path ──────────────────────────────────────────────────────
    if mcp_ctx.session:
        try:
            mcp_result = await mcp_ctx.session.call_tool(
                "aggregate",
                {
                    "collection": "notes",
                    "database":   "notely_db",
                    "pipeline":   pipeline,
                },
            )
            # MCP returns a list of content objects; collect all text parts
            raw_text = "\n".join(
                c.text for c in mcp_result.content if c.type == "text"
            )
            # Parse the JSON array the MCP server returns
            try:
                results = _json.loads(raw_text)
                if not isinstance(results, list):
                    results = []
                # MCP returns extended JSON: _id may be {"$oid": "..."} — normalise it
                for r in results:
                    if isinstance(r.get("_id"), dict):
                        r["_id"] = r["_id"].get("$oid", str(r["_id"]))
            except (_json.JSONDecodeError, Exception) as parse_err:
                print(f"[MCP search] JSON parse error: {parse_err} — raw: {raw_text[:200]}")
                results = []

            return _format_note_results(results, query), True

        except Exception as mcp_err:
            print(f"[MCP search] call failed ({mcp_err}), falling back to PyMongo")

    # ── PyMongo fallback ──────────────────────────────────────────────────────
    return tool_search_notes(query, college, subject, semester, department), False


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — Generate Summary
# ══════════════════════════════════════════════════════════════════════════════

def tool_generate_summary(note_id: str) -> str:
    """Fetch note content so Gemini can write a clean exam-ready summary."""
    note = get_note_full_content(note_id)
    if not note:
        return f"Note `{note_id}` not found. Please search for it first."

    content = note.get("full_content") or note.get("content_preview") or ""
    if not content:
        return "This note has no text content to summarise."

    return (
        f"SUMMARISE_THIS_NOTE\n"
        f"Title: {note.get('title', 'Untitled')}\n"
        f"Subject: {note.get('subject', '')}\n"
        f"College: {note.get('college', '')} | "
        f"Sem {note.get('semester', '')} | "
        f"Teacher: {note.get('teacher', 'Unknown')}\n\n"
        f"CONTENT:\n{content[:6000]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 3 — Generate Flashcards
# ══════════════════════════════════════════════════════════════════════════════

def tool_generate_flashcards(note_id: str, num_cards: int = 10) -> str:
    """Fetch note content so Gemini can produce Q&A flashcard pairs."""
    note = get_note_full_content(note_id)
    if not note:
        return f"Note `{note_id}` not found. Search for it first."

    content = note.get("full_content") or note.get("content_preview") or ""
    if not content:
        return "No content available to generate flashcards from."

    return (
        f"GENERATE_FLASHCARDS\n"
        f"Number of cards: {num_cards}\n"
        f"Format EXACTLY as:\n"
        f"CARD N\nQ: [question]\nA: [concise answer]\n\n"
        f"Title: {note.get('title', '')}\n"
        f"Subject: {note.get('subject', '')}\n\n"
        f"CONTENT:\n{content[:5500]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 4 — Generate Quiz
# ══════════════════════════════════════════════════════════════════════════════

def tool_generate_quiz(note_id: str, num_questions: int = 5) -> str:
    """Fetch note content so Gemini can write a multiple-choice practice quiz."""
    note = get_note_full_content(note_id)
    if not note:
        return f"Note `{note_id}` not found. Search for it first."

    content = note.get("full_content") or note.get("content_preview") or ""
    if not content:
        return "No content available to generate a quiz from."

    return (
        f"GENERATE_QUIZ\n"
        f"Questions: {num_questions}\n"
        f"Format EXACTLY:\n"
        f"Q[N]: [question]\n"
        f"A) [option]  B) [option]  C) [option]  D) [option]\n"
        f"✅ Answer: [letter] — [one-line explanation]\n\n"
        f"Title: {note.get('title', '')}\n"
        f"Subject: {note.get('subject', '')}\n\n"
        f"CONTENT:\n{content[:5500]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 5 — Trending Notes
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_trending(college: str = "") -> str:
    notes = get_trending_notes(college=college, limit=8)
    if not notes:
        return "No notes yet — be the first to upload!"

    header = f"📈 **Trending Notes{' at ' + college if college else ''}:**\n"
    rows = []
    for i, note in enumerate(notes, 1):
        rows.append(
            f"{i}. **{note.get('title', 'Untitled')}**\n"
            f"   {note.get('subject','?')} | "
            f"Sem {note.get('semester','?')} | "
            f"⬆ {note.get('upvotes', 0)} | "
            f"Quality: {note.get('quality_score', 0)}\n"
            f"   ID: `{note.get('_id', '')}`"
        )
    return header + "\n\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 6 — Leaderboard
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_leaderboard(college: str = "") -> str:
    leaders = get_leaderboard(college=college, limit=10)
    if not leaders:
        return "No contributors yet — upload a note to be the first!"

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    header = f"🏆 **Top Contributors{' at ' + college if college else ''}:**\n"
    rows   = []
    for i, u in enumerate(leaders, 1):
        medal  = medals.get(i, f"#{i}")
        badges = ", ".join(u.get("badges", []))
        rows.append(
            f"{medal} **{u.get('name', 'Unknown')}** — "
            f"{u.get('college', '?')}\n"
            f"   Points: {u.get('points', 0):,}"
            + (f" | Badges: {badges}" if badges else "")
        )
    return header + "\n\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 7 — Get Comments
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_comments(note_id: str) -> str:
    comments = get_comments(note_id)
    if not comments:
        return "No comments yet on this note. Be the first to comment!"

    lines = [f"💬 **Comments ({len(comments)}):**\n"]
    for c in comments[:10]:
        lines.append(
            f"**{c.get('user_name', 'Anonymous')}** "
            f"— {_fmt_date(c.get('created_at'))}\n"
            f"  {c.get('content', '')}\n"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 8 — Note Request Board
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_requests(college: str = "") -> str:
    reqs = get_requests(college=college, limit=10)
    if not reqs:
        return "No open requests right now. Great time to upload your notes!"

    header = f"📋 **Open Note Requests{' at ' + college if college else ''}:**\n"
    rows   = []
    for r in reqs:
        rows.append(
            f"**{r.get('title', 'Untitled')}**\n"
            f"   {r.get('subject','?')} | "
            f"{r.get('college','?')} | "
            f"Sem {r.get('semester','?')}\n"
            f"   Posted by: {r.get('posted_by','Anonymous')} | "
            f"Priority: {r.get('priority','medium').upper()}\n"
            f"   Request ID: `{r.get('_id', '')}`"
        )
    return header + "\n\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 9 — Forum Posts
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_forum_posts(college: str = "") -> str:
    posts = get_forum_posts(college=college, limit=8)
    if not posts:
        return "No forum posts yet. Start a discussion!"

    header = f"💬 **Forum Posts{' at ' + college if college else ''}:**\n"
    rows   = []
    for p in posts:
        rows.append(
            f"**{p.get('title', 'Untitled')}**\n"
            f"   By: {p.get('author','Anonymous')} | "
            f"{_fmt_date(p.get('created_at'))} | "
            f"⬆ {p.get('upvotes', 0)} | "
            f"💬 {p.get('replies', 0)} replies"
        )
    return header + "\n\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 10 — Academic Memory
# ══════════════════════════════════════════════════════════════════════════════

def tool_record_quiz_score(topic: str, score: float, user_id: str = "") -> str:
    """Record a quiz score and update the user's strong/weak topics."""
    return update_academic_memory(user_id, topic, score)

def tool_record_study_session(topic: str, user_id: str = "") -> str:
    """Record that the student has studied a topic."""
    return record_study_session(user_id, topic)

def tool_add_learning_pattern(pattern: str, user_id: str = "") -> str:
    """Record a learning pattern (e.g. 'Student struggles with visualising graphs')."""
    return add_learning_pattern(user_id, pattern)


# ══════════════════════════════════════════════════════════════════════════════
#  ASYNC DISPATCHER — main.py imports and awaits this
# ══════════════════════════════════════════════════════════════════════════════

async def dispatch_tool(tool_name: str, tool_args: dict, user_id: str = "") -> tuple:
    """
    Async central dispatcher called by main.py's agentic loop.

    Returns (result_string, via_mcp: bool):
      via_mcp=True  — tool used the MongoDB MCP Server (shown in Activity Feed)
      via_mcp=False — tool used direct PyMongo / Gemini

    Never raises — always returns a usable string result.
    """
    # Inject user_id for memory tools (copy to avoid mutating caller's dict)
    if tool_name in ("record_quiz_score", "record_study_session", "add_learning_pattern"):
        tool_args = {**tool_args, "user_id": user_id}

    # ── MCP-routed: search_notes ───────────────────────────────────────────────
    if tool_name == "search_notes":
        try:
            return await tool_search_notes_mcp(**tool_args)
        except TypeError as e:
            return f"search_notes called with wrong args: {e}", False
        except Exception as e:
            return f"search_notes failed: {e}", False

    # ── Sync tools (PyMongo / Gemini) ─────────────────────────────────────────
    sync_mapping = {
        "generate_summary":    tool_generate_summary,
        "generate_flashcards": tool_generate_flashcards,
        "generate_quiz":       tool_generate_quiz,
        "get_trending_notes":  tool_get_trending,
        "get_leaderboard":     tool_get_leaderboard,
        "get_comments":        tool_get_comments,
        "get_requests":        tool_get_requests,
        "get_forum_posts":     tool_get_forum_posts,
        "record_quiz_score":   tool_record_quiz_score,
        "record_study_session":tool_record_study_session,
        "add_learning_pattern":tool_add_learning_pattern,
    }

    fn = sync_mapping.get(tool_name)
    if fn is None:
        return (
            f"Unknown tool: '{tool_name}'. "
            f"Available: search_notes, {', '.join(sync_mapping.keys())}",
            False,
        )

    try:
        return fn(**tool_args), False
    except TypeError as e:
        return f"Tool '{tool_name}' called with wrong arguments: {e}", False
    except Exception as e:
        return f"Tool '{tool_name}' failed: {e}", False