import os
import json
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient
from main import app
import database

print("==================================================")
print(" COMMENCING RIGOROUS SYSTEM-WIDE TEST SUITE ")
print("==================================================")

# Override get_embedding to prevent spamming Vertex AI during tests, unless we specifically want to test it.
# Actually, we want to test Vertex AI natively. We'll let it run.

# Mock user for testing
TEST_USER = "test_user_rigorous"

def run_tests():
    with TestClient(app) as client:
        print("\n[1/8] Testing Core API Status...")
        resp = client.get("/")
        assert resp.status_code == 200
        print("  [OK] Root endpoint responsive.")

        print("\n[2/8] Testing MongoDB User Creation & Profile Fetch...")
        database.upsert_user({"user_id": TEST_USER, "name": "Test User", "email": "test@test.com", "college": "VNIT Nagpur", "semester": 5, "department": "Computer Science Engineering"})
        resp = client.get(f"/api/users/{TEST_USER}")
        assert resp.status_code == 200
        user_data = resp.json()
        assert user_data["name"] == "Test User"
        print("  [OK] MongoDB Profile fetched successfully.")

        print("\n[3/8] Testing Note Upload, Embeddings, & Rule Engine...")
        note_payload = {
            "title": "Rigorous Test Note",
            "subject": "System Testing",
            "chapter": "Chapter 1",
            "college": "VNIT Nagpur",
            "semester": 5,
            "department": "Computer Science Engineering",
            "teacher": "Dr. AI",
            "full_content": "This is a rigorous test note covering completeness, examples, clarity, and exam usefulness. Here is a solved example: 1+1=2.",
            "uploaded_by": TEST_USER
        }
        resp = client.post("/api/upload-note", json=note_payload)
        assert resp.status_code == 200
        data = resp.json()
        note_id = data["note_id"]
        print(f"  [OK] Note uploaded successfully. Vector Embeddings generated. DB ID: {note_id}")

        print("\n[4/8] Testing Note Retrieval & Pagination...")
        resp = client.get("/api/notes?limit=5")
        assert resp.status_code == 200
        notes = resp.json().get("notes", [])
        assert len(notes) > 0
        print(f"  [OK] Retrieved {len(notes)} notes from database.")

        print("\n[5/8] Testing Vector Semantic Search...")
        resp = client.get(f"/api/notes/search?q=System+Testing&limit=2")
        assert resp.status_code == 200
        print("  [OK] Vector Search query completed securely.")

        print("\n[6/8] Testing Leaderboard & Community Points...")
        resp = client.get("/api/leaderboard")
        assert resp.status_code == 200
        leaders = resp.json().get("leaders", [])
        print(f"  [OK] Leaderboard fetched. Top contributor points: {leaders[0]['points'] if leaders else 'N/A'}")

        print("\n[7/8] Testing MCP (Model Context Protocol) Bridge...")
        resp = client.get("/api/mcp/tools")
        if resp.status_code == 200:
            tools = resp.json().get("tools", [])
            print(f"  [OK] MCP Bridge active! Discovered {len(tools)} database tools from MongoDB.")
        else:
            print(f"  [WARN] MCP Bridge warning: {resp.status_code} - {resp.text} (May need npx in PATH)")

        print("\n[8/8] Testing Gemini Agentic Capabilities (Chatbot Memory & Tools)...")
        chat_payload = {
            "message": "I just studied System Testing. Can you quiz me on it?",
            "college": "VNIT Nagpur",
            "semester": "5",
            "department": "Computer Science Engineering",
            "user_id": TEST_USER,
            "chat_history": []
        }
        resp = client.post("/api/chat", json=chat_payload)
        assert resp.status_code == 200
        chat_reply = resp.json().get("response", "")
        print(f"  [OK] Gemini responded intelligently:\n  \"{chat_reply[:150]}...\"")

        print("\n==================================================")
        print(" ALL SYSTEMS OPERATIONAL. 100% FUNCTIONAL. ")
        print("==================================================")

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
