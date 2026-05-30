from database import (
    vector_search_notes, get_note_full_content, 
    get_trending_notes, get_leaderboard, insert_note
)
from embeddings import get_embedding
import json


def tool_search_notes(query: str, college: str = "", subject: str = "", 
                       semester: str = "", department: str = "") -> str:
    """
    Tool: Search for notes using semantic search.
    The agent calls this when a student asks to find notes.
    """
    # Convert the student's query into a vector
    query_embedding = get_embedding(query)
    
    # Build filters from what the student specified
    filters = {}
    if college: filters["college"] = college
    if subject: filters["subject"] = subject
    if semester: filters["semester"] = semester
    if department: filters["department"] = department
    
    # Search MongoDB Atlas Vector Search
    results = vector_search_notes(query_embedding, filters, limit=5)
    
    if not results:
        return "No notes found matching your search. Try broadening your filters."
    
    # Format results nicely for the agent
    formatted = []
    for i, note in enumerate(results, 1):
        formatted.append(
            f"{i}. **{note['title']}**\n"
            f"   Subject: {note['subject']} | Semester: {note['semester']}\n"
            f"   College: {note['college']} | Teacher: {note.get('teacher', 'Unknown')}\n"
            f"   Quality Score: {note['quality_score']} | Upvotes: {note['upvotes']}\n"
            f"   Note ID: {str(note['_id'])}\n"
            f"   Preview: {note.get('content_preview', 'No preview available')[:200]}..."
        )
    
    return "\n\n".join(formatted)


def tool_generate_summary(note_id: str) -> str:
    """
    Tool: Get the full content of a note and return it for summarization.
    The agent will then summarize it using Gemini.
    """
    note = get_note_full_content(note_id)
    if not note:
        return "Note not found."
    
    content = note.get("full_content", "No content available")
    title = note.get("title", "Untitled")
    
    return f"NOTE TITLE: {title}\n\nCONTENT:\n{content[:6000]}"


def tool_generate_flashcards(note_id: str, num_cards: int = 10) -> str:
    """
    Tool: Fetch note content to generate flashcards from.
    Returns the content — Gemini will create the actual flashcards.
    """
    note = get_note_full_content(note_id)
    if not note:
        return "Note not found."
    
    return (
        f"Generate exactly {num_cards} flashcards from this note.\n"
        f"Format each as: Q: [question] | A: [answer]\n\n"
        f"NOTE CONTENT:\n{note.get('full_content', '')[:5000]}"
    )


def tool_generate_quiz(note_id: str, num_questions: int = 5) -> str:
    """
    Tool: Fetch note content to generate a practice quiz.
    """
    note = get_note_full_content(note_id)
    if not note:
        return "Note not found."
    
    return (
        f"Generate a {num_questions}-question multiple choice quiz from this note.\n"
        f"Format: Q: [question]\nA) option B) option C) option D) option\nAnswer: [letter]\nExplanation: [why]\n\n"
        f"NOTE CONTENT:\n{note.get('full_content', '')[:5000]}"
    )


def tool_get_trending(college: str = "") -> str:
    """
    Tool: Get trending notes, optionally filtered by college.
    """
    notes = get_trending_notes(college=college if college else None, limit=8)
    
    if not notes:
        return "No notes available yet."
    
    formatted = ["📈 **Trending Notes:**\n"]
    for i, note in enumerate(notes, 1):
        formatted.append(
            f"{i}. {note['title']} — {note['subject']}, Sem {note['semester']}\n"
            f"   ⬆ {note['upvotes']} upvotes | Score: {note['quality_score']}"
        )
    
    return "\n".join(formatted)


def tool_get_leaderboard(college: str = "") -> str:
    """
    Tool: Get top student contributors.
    """
    leaders = get_leaderboard(college=college if college else None, limit=10)
    
    if not leaders:
        return "No contributors yet."
    
    formatted = ["🏆 **Top Contributors:**\n"]
    for i, user in enumerate(leaders, 1):
        formatted.append(
            f"{i}. {user['name']} — {user.get('college', 'Unknown College')}\n"
            f"   Points: {user['points']} | Badges: {', '.join(user.get('badges', []))}"
        )
    
    return "\n".join(formatted)