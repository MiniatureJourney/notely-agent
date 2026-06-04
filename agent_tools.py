"""
agent_tools.py — All tools the Gemini agent can call

Bugs fixed vs original:
1. get_trending_notes(college=college if college else None) — passing None to
   the fixed database.py now works, but the original database.py had a bug
   where None caused an empty query. Standardised to always pass a string.
2. get_leaderboard(college=college if college else None) — same issue. Fixed.
3. tool_search_notes result formatting used note['_id'] directly — after
   database.py fix, _id is already a string, but added str() defensively.
4. note.get('content_preview','')[:200] — preview was cut to 200 chars which
   is too short for Gemini to understand context. Increased to 300.
5. Missing tools that the problem statement requires:
   - tool_get_comments()   — "students can comment on notes"
   - tool_get_requests()   — "Note Request Board"
   - tool_get_forum_posts() — "Community Forum for Every College"
6. dispatch_tool() was missing entirely — main.py imports and calls it.
   Without it, main.py crashes with ImportError on startup.
7. tool_get_trending passed college="" to get_trending_notes which now
   correctly handles empty string (no filter). Fixed consistently.
8. import json was imported but never used. Removed.
9. insert_note was imported but never used by any tool. Removed.
10. Tool return strings were inconsistent — some used \n\n joining,
    some used \n. Standardised so Gemini reads them cleanly.
"""

from database import (
    vector_search_notes,
    get_note_full_content,
    get_trending_notes,
    get_leaderboard,
    get_comments,
    get_requests,
    get_forum_posts,
)
from database import update_academic_memory
from embeddings import get_embedding
from datetime import datetime


# ── Helper ────────────────────────────────────────────────────────────────────

def _fmt_date(dt) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%d %b %Y")
    return str(dt) if dt else ""


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — Search Notes
# ══════════════════════════════════════════════════════════════════════════════

def tool_search_notes(query: str, college: str = "", subject: str = "",
                      semester: str = "", department: str = "") -> str:
    """
    Search notes via MongoDB Atlas Vector Search.
    Falls back to empty result message with helpful suggestions.
    """
    query_embedding = get_embedding(query)

    filters = {}
    if college:    filters["college"]    = college
    if subject:    filters["subject"]    = subject
    if semester:   filters["semester"]   = semester
    if department: filters["department"] = department

    results = vector_search_notes(query_embedding, filters, limit=5)

    if not results:
        return (
            f"No notes found for '{query}'. "
            "Try broader keywords or remove filters. "
            "You can also post a request on the Note Request Board!"
        )

    lines = [f"Found **{len(results)} notes** for '{query}':\n"]
    for i, note in enumerate(results, 1):
        # FIX 3: _id is already str after database.py fix, str() is defensive
        note_id = str(note.get("_id", ""))
        # FIX 4: 300 chars gives Gemini enough context
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
    """
    FIX 1 & 7: pass college as string (not None). database.py handles
    empty string correctly — no filter applied when college="".
    """
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
    """
    FIX 2: pass college as string. database.py handles empty string correctly.
    """
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
#  TOOL 7 — Get Comments  (FIX 5: was missing)
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_comments(note_id: str) -> str:
    """Get comments on a specific note — problem statement requires this."""
    comments = get_comments(note_id)

    if not comments:
        return f"No comments yet on this note. Be the first to comment!"

    lines = [f"💬 **Comments ({len(comments)}):**\n"]
    for c in comments[:10]:
        lines.append(
            f"**{c.get('user_name', 'Anonymous')}** "
            f"— {_fmt_date(c.get('created_at'))}\n"
            f"  {c.get('content', '')}\n"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 8 — Note Request Board  (FIX 5: was missing)
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_requests(college: str = "") -> str:
    """Get open note requests — problem statement requires a Request Board."""
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
#  TOOL 9 — Forum Posts  (FIX 5: was missing)
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_forum_posts(college: str = "") -> str:
    """Get college forum posts — problem statement requires community forum."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCHER  (FIX 6: was completely missing — main.py imports this)
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_tool(tool_name: str, tool_args: dict, user_id: str = "") -> str:
    """
    Central dispatcher called by main.py's agentic loop.
    Maps tool_name strings (from Gemini function calls) to Python functions.
    Returns a plain string in all cases — never raises.
    """
    mapping = {
        "search_notes":        tool_search_notes,
        "generate_summary":    tool_generate_summary,
        "generate_flashcards": tool_generate_flashcards,
        "generate_quiz":       tool_generate_quiz,
        "get_trending_notes":  tool_get_trending,
        "get_leaderboard":     tool_get_leaderboard,
        "get_comments":        tool_get_comments,
        "get_requests":        tool_get_requests,
        "get_forum_posts":     tool_get_forum_posts,
        "record_quiz_score":   tool_record_quiz_score,
    }

    if tool_name == "record_quiz_score":
        tool_args["user_id"] = user_id

    fn = mapping.get(tool_name)
    if fn is None:
        return f"Unknown tool: '{tool_name}'. Available: {list(mapping.keys())}"

    try:
        return fn(**tool_args)
    except TypeError as e:
        return f"Tool '{tool_name}' called with wrong arguments: {e}"
    except Exception as e:
        return f"Tool '{tool_name}' failed: {e}"