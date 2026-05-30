from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
import os
from dotenv import load_dotenv
from agent_tools import (
    tool_search_notes, tool_generate_summary,
    tool_generate_flashcards, tool_generate_quiz,
    tool_get_trending, tool_get_leaderboard
)
from embeddings import get_embedding
from database import insert_note
import json

load_dotenv()

app = FastAPI(title="Notely AI Agent API")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vertex AI
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-central1"
)


# ---- Define tools the Gemini agent can use ----

search_notes_tool = FunctionDeclaration(
    name="search_notes",
    description="Search for academic notes by topic, subject, college, or semester using semantic search",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What the student is looking for"},
            "college": {"type": "string", "description": "College name filter (optional)"},
            "subject": {"type": "string", "description": "Subject name filter (optional)"},
            "semester": {"type": "string", "description": "Semester number filter (optional)"},
            "department": {"type": "string", "description": "Department filter (optional)"}
        },
        "required": ["query"]
    }
)

summarize_note_tool = FunctionDeclaration(
    name="generate_summary",
    description="Generate a clean study summary from a specific note",
    parameters={
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "description": "The MongoDB ID of the note"}
        },
        "required": ["note_id"]
    }
)

flashcard_tool = FunctionDeclaration(
    name="generate_flashcards",
    description="Generate study flashcards from a note",
    parameters={
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "description": "The MongoDB ID of the note"},
            "num_cards": {"type": "integer", "description": "Number of flashcards to generate (default 10)"}
        },
        "required": ["note_id"]
    }
)

quiz_tool = FunctionDeclaration(
    name="generate_quiz",
    description="Generate a practice quiz from a note",
    parameters={
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "description": "The MongoDB ID of the note"},
            "num_questions": {"type": "integer", "description": "Number of questions (default 5)"}
        },
        "required": ["note_id"]
    }
)

trending_tool = FunctionDeclaration(
    name="get_trending_notes",
    description="Get trending and popular notes, optionally filtered by college",
    parameters={
        "type": "object",
        "properties": {
            "college": {"type": "string", "description": "Filter by college name (optional)"}
        }
    }
)

leaderboard_tool = FunctionDeclaration(
    name="get_leaderboard",
    description="Get the top contributing students on the platform",
    parameters={
        "type": "object",
        "properties": {
            "college": {"type": "string", "description": "Filter by college (optional)"}
        }
    }
)

# Bundle all tools
notebot_tools = Tool(function_declarations=[
    search_notes_tool, summarize_note_tool,
    flashcard_tool, quiz_tool,
    trending_tool, leaderboard_tool
])


# ---- Tool dispatcher ----

def handle_tool_call(tool_name: str, tool_args: dict) -> str:
    if tool_name == "search_notes":
        return tool_search_notes(**tool_args)
    elif tool_name == "generate_summary":
        return tool_generate_summary(**tool_args)
    elif tool_name == "generate_flashcards":
        return tool_generate_flashcards(**tool_args)
    elif tool_name == "generate_quiz":
        return tool_generate_quiz(**tool_args)
    elif tool_name == "get_trending_notes":
        return tool_get_trending(**tool_args)
    elif tool_name == "get_leaderboard":
        return tool_get_leaderboard(**tool_args)
    else:
        return f"Unknown tool: {tool_name}"


# ---- Main agent chat endpoint ----

class ChatRequest(BaseModel):
    message: str
    college: str = ""
    semester: str = ""
    department: str = ""
    chat_history: list = []

@app.post("/chat")
async def chat_with_notebot(request: ChatRequest):
    """Main endpoint — student sends a message, agent responds"""
    
    model = GenerativeModel(
        "gemini-1.5-pro",
        tools=[notebot_tools],
        system_instruction=(
            "You are NoteBot, an intelligent AI study assistant on Notely — "
            "a note-sharing platform for Indian college students. "
            "You help students find academic notes, understand concepts, create summaries, "
            "generate flashcards, and take practice quizzes. "
            "Always be friendly, encouraging, and focused on helping students learn. "
            f"The student's context: College='{request.college}', "
            f"Semester='{request.semester}', Department='{request.department}'. "
            "Use this context when searching for notes if the student doesn't specify."
        )
    )
    
    # Build conversation history
    history = []
    for msg in request.chat_history[-10:]:  # Last 10 messages for context
        history.append({"role": msg["role"], "parts": [msg["content"]]})
    
    chat = model.start_chat(history=history)
    
    # Agentic loop — agent can call multiple tools in sequence
    response = chat.send_message(request.message)
    
    max_iterations = 5  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check if the agent wants to call a tool
        if response.candidates[0].content.parts[0].function_call:
            func_call = response.candidates[0].content.parts[0].function_call
            tool_name = func_call.name
            tool_args = dict(func_call.args)
            
            # Execute the tool (this queries MongoDB)
            tool_result = handle_tool_call(tool_name, tool_args)
            
            # Send tool result back to agent
            response = chat.send_message(
                vertexai.generative_models.Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_result}
                )
            )
        else:
            # Agent gave a final text response — we're done
            break
    
    return {
        "response": response.text,
        "success": True
    }


@app.post("/upload-note")
async def upload_note(
    title: str,
    subject: str,
    college: str,
    semester: int,
    department: str,
    chapter: str = "",
    teacher: str = "",
    content: str = "",
    uploaded_by: str = "anonymous"
):
    """Upload a new note — generates embedding and stores in MongoDB"""
    
    # Generate embedding for semantic search
    text_to_embed = f"{title} {subject} {chapter} {content[:2000]}"
    embedding = get_embedding(text_to_embed)
    
    note_data = {
        "title": title,
        "subject": subject,
        "college": college,
        "semester": semester,
        "department": department,
        "chapter": chapter,
        "teacher": teacher,
        "full_content": content,
        "content_preview": content[:500],
        "embedding": embedding,
        "uploaded_by": uploaded_by
    }
    
    note_id = insert_note(note_data)
    
    return {"note_id": note_id, "message": "Note uploaded successfully!"}


@app.get("/health")
async def health_check():
    return {"status": "Notely Agent is running!"}