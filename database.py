from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["notely_db"]

notes_collection = db["notes"]
users_collection = db["users"]
requests_collection = db["requests"]
forum_collection = db["forum_posts"]


def insert_note(note_data: dict):
    """Insert a new note into MongoDB"""
    note_data["created_at"] = datetime.utcnow()
    note_data["upvotes"] = 0
    note_data["downvotes"] = 0
    note_data["quality_score"] = 0
    result = notes_collection.insert_one(note_data)
    return str(result.inserted_id)


def vector_search_notes(query_embedding: list, filters: dict = {}, limit: int = 5):
    """Search notes using MongoDB Atlas Vector Search"""
    
    # Build the filter pipeline
    pre_filter = {}
    if filters.get("college"):
        pre_filter["college"] = filters["college"]
    if filters.get("subject"):
        pre_filter["subject"] = filters["subject"]
    if filters.get("semester"):
        pre_filter["semester"] = {"$eq": int(filters["semester"])}
    if filters.get("department"):
        pre_filter["department"] = filters["department"]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "notes_vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": limit,
                "filter": pre_filter if pre_filter else None
            }
        },
        {
            "$project": {
                "score": {"$meta": "vectorSearchScore"},
                "title": 1,
                "subject": 1,
                "college": 1,
                "semester": 1,
                "department": 1,
                "chapter": 1,
                "teacher": 1,
                "upvotes": 1,
                "quality_score": 1,
                "content_preview": 1,
                "uploaded_by": 1,
                "created_at": 1
            }
        }
    ]
    
    # Remove None filter
    pipeline[0]["$vectorSearch"] = {
        k: v for k, v in pipeline[0]["$vectorSearch"].items() if v is not None
    }
    
    results = list(notes_collection.aggregate(pipeline))
    return results


def get_note_full_content(note_id: str):
    """Get the full text content of a specific note"""
    from bson import ObjectId
    note = notes_collection.find_one({"_id": ObjectId(note_id)})
    return note


def get_trending_notes(college: str = None, limit: int = 10):
    """Get trending notes by quality score"""
    query = {}
    if college:
        query["college"] = college
    
    notes = list(
        notes_collection.find(query)
        .sort("quality_score", -1)
        .limit(limit)
        .project({"embedding": 0})
    )
    return notes


def upvote_note(note_id: str, user_id: str):
    """Upvote a note and update quality score"""
    from bson import ObjectId
    notes_collection.update_one(
        {"_id": ObjectId(note_id)},
        {"$inc": {"upvotes": 1, "quality_score": 2}}
    )


def get_leaderboard(college: str = None, limit: int = 10):
    """Get top contributors"""
    query = {}
    if college:
        query["college"] = college
    
    return list(
        users_collection.find(query)
        .sort("points", -1)
        .limit(limit)
        .project({"_id": 0, "name": 1, "college": 1, "points": 1, "badges": 1})
    )
