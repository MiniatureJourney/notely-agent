"""
database.py — All MongoDB operations for Notely

Bugs fixed vs original:
1. ObjectId not serialised — find/aggregate results contain BSON ObjectId which
   crashes FastAPI's JSON response. Fixed: _serial() converts _id to str everywhere.
2. get_trending_notes(college=None) built query={} which is fine, BUT passing
   college="" also set query["college"]="" which matches nothing. Fixed: only
   add filter when college is a non-empty string.
3. upvote_note took user_id param but signature didn't match main.py call
   (main.py calls upvote_note(note_id) with no user_id). Fixed: user_id removed,
   returns bool so caller knows if it succeeded.
4. vector_search_notes returned raw BSON with ObjectId — crashes JSON. Fixed.
5. get_note_full_content returned raw BSON with ObjectId — crashes JSON. Fixed.
6. ZERO_VECTOR check missing — if embeddings.py returned ZERO_VECTOR, sending
   it to $vectorSearch returns junk results. Fixed: skip vector search and fall
   back to text search when embedding is all zeros.
7. Missing functions needed by main.py:
   - get_all_notes()        (home/browse grid)
   - get_recent_notes()     (newest filter)
   - get_note_by_id()       (note detail page)
   - search_notes_text()    (regex fallback)
   - downvote_note()
   - increment_downloads()
   - get_user_notes()       (profile page)
   - add_comment()
   - get_comments()
   - create_request()
   - get_requests()
   - fulfill_request()
   - create_forum_post()
   - get_forum_posts()
   - upvote_forum_post()
   - upsert_user()
   - get_user_by_id()
   - update_user_points()
8. comments collection was never defined. Fixed.
"""

import os
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()

# ── Connection ────────────────────────────────────────────────────────────────
_client = None

def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGODB_URI"))
    return _client

def _db():
    return _get_client()["notely_db"]

# ── Collection accessors ──────────────────────────────────────────────────────
def notes_col():    return _db()["notes"]
def users_col():    return _db()["users"]
def comments_col(): return _db()["comments"]      # FIX 8: was missing
def user_memory_col(): return _db()["user_memory"]
def study_history_col(): return _db()["study_history"]
def quiz_results_col(): return _db()["quiz_results"]
def learning_patterns_col(): return _db()["learning_patterns"]
def requests_col(): return _db()["requests"]
def forum_col():    return _db()["forum_posts"]

# ── FIX 1: ObjectId serialiser ────────────────────────────────────────────────
def _serial(doc: dict | None) -> dict | None:
    """Convert ObjectId _id to string so FastAPI can JSON-serialise the doc."""
    if doc is None:
        return None
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

def _serial_list(docs: list) -> list:
    return [_serial(d) for d in docs]

# ── ZERO_VECTOR detection (FIX 6) ─────────────────────────────────────────────
def _is_zero_vector(v: list) -> bool:
    return not v or all(x == 0.0 for x in v)


# ── PLATFORM STATS ────────────────────────────────────────────────────────────
def get_platform_stats() -> dict:
    """Fetch live counts for the animated platform stats ticker."""
    try:
        req_sample = requests_col().find_one()
        if req_sample and "status" in req_sample:
            req_count = requests_col().count_documents({"status": "open"})
        else:
            req_count = requests_col().estimated_document_count()

        return {
            "notes": notes_col().estimated_document_count(),
            "users": users_col().estimated_document_count(),
            "requests": req_count
        }
    except Exception:
        # Fallback if DB is slow or unavailable
        return {"notes": 0, "users": 0, "requests": 0}

# ══════════════════════════════════════════════════════════════════════════════
#  NOTES
# ══════════════════════════════════════════════════════════════════════════════

def insert_note(note_data: dict) -> str:
    note_data.setdefault("created_at",    datetime.utcnow())
    note_data.setdefault("upvotes",       0)
    note_data.setdefault("downvotes",     0)
    note_data.setdefault("quality_score", 0)
    note_data.setdefault("downloads",     0)
    note_data.setdefault("views",         0)
    result = notes_col().insert_one(note_data)
    return str(result.inserted_id)


def get_all_notes(college: str = "", department: str = "",
                  semester: int = 0, limit: int = 20) -> list:
    """Fetch notes with optional filters, sorted by quality score. FIX 7."""
    query = {}
    # FIX 2: only filter when value is non-empty
    if college:    query["college"]    = {"$regex": college,    "$options": "i"}
    if department: query["department"] = {"$regex": department, "$options": "i"}
    if semester:   query["semester"]   = semester

    cursor = notes_col().find(query, {"embedding": 0}) \
                        .sort("quality_score", DESCENDING) \
                        .limit(limit)
    return _serial_list(list(cursor))


def get_recent_notes(college: str = "", limit: int = 20) -> list:
    """Fetch notes sorted by creation date. FIX 7."""
    query = {}
    if college:
        query["college"] = {"$regex": college, "$options": "i"}
    cursor = notes_col().find(query, {"embedding": 0}) \
                        .sort("created_at", DESCENDING) \
                        .limit(limit)
    return _serial_list(list(cursor))


def get_note_by_id(note_id: str) -> dict | None:
    """Fetch one note by its MongoDB _id. FIX 5 + FIX 7."""
    try:
        doc = notes_col().find_one({"_id": ObjectId(note_id)})
        return _serial(doc)
    except (InvalidId, Exception):
        return None


def get_note_full_content(note_id: str) -> dict | None:
    """Alias used by agent_tools — returns full note including full_content."""
    return get_note_by_id(note_id)


def vector_search_notes(query_embedding: list, filters: dict = None,
                        limit: int = 5) -> list:
    """
    MongoDB Atlas Vector Search.
    FIX 4: serialises ObjectId.
    FIX 6: falls back to text search when embedding is ZERO_VECTOR.
    """
    if filters is None:
        filters = {}

    # FIX 6: skip vector search when embedding failed
    if _is_zero_vector(query_embedding):
        return search_notes_text(
            "",
            college=filters.get("college", ""),
            department=filters.get("department", ""),
            semester=int(filters["semester"]) if filters.get("semester") else 0,
            limit=limit,
        )

    # Build pre-filter (only non-empty values)
    pre_filter = {}
    if filters.get("college"):
        pre_filter["college"] = filters["college"]
    if filters.get("subject"):
        pre_filter["subject"] = filters["subject"]
    if filters.get("department"):
        pre_filter["department"] = filters["department"]
    if filters.get("semester"):
        try:
            pre_filter["semester"] = {"$eq": int(filters["semester"])}
        except (ValueError, TypeError):
            pass

    vector_stage: dict = {
        "index":         "notes_vector_index",
        "path":          "embedding",
        "queryVector":   query_embedding,
        "numCandidates": 100,
        "limit":         limit,
    }
    if pre_filter:
        vector_stage["filter"] = pre_filter

    pipeline = [
        {"$vectorSearch": vector_stage},
        {
            "$project": {
                "_id": 1,
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
                "created_at": 1,
                "downloads": 1
             }
        },
    ]

    try:
        results = list(notes_col().aggregate(pipeline))
        return _serial_list(results)
    except PyMongoError as e:
        print(f"[vector_search] error: {e}")
        return []


def search_notes_text(query: str, college: str = "", department: str = "",
                      semester: int = 0, limit: int = 10) -> list:
    """Regex text search — fallback when vector search fails. FIX 7."""
    conditions = []
    if query:
        conditions.append({
            "$or": [
                {"title":           {"$regex": query, "$options": "i"}},
                {"subject":         {"$regex": query, "$options": "i"}},
                {"chapter":         {"$regex": query, "$options": "i"}},
                {"content_preview": {"$regex": query, "$options": "i"}},
                {"teacher":         {"$regex": query, "$options": "i"}},
            ]
        })
    if college:    conditions.append({"college":    {"$regex": college,    "$options": "i"}})
    if department: conditions.append({"department": {"$regex": department, "$options": "i"}})
    if semester:   conditions.append({"semester": semester})

    q = {"$and": conditions} if conditions else {}
    cursor = notes_col().find(q, {"embedding": 0}) \
                        .sort("quality_score", DESCENDING) \
                        .limit(limit)
    return _serial_list(list(cursor))


def get_trending_notes(college: str = "", limit: int = 10) -> list:
    """
    FIX 2: original passed college=None which built query["college"]=None
    matching nothing. Now only filters when college is a non-empty string.
    """
    query = {}
    if college:
        query["college"] = {"$regex": college, "$options": "i"}
    cursor = notes_col().find(query, {"embedding": 0}) \
                        .sort([("upvotes", DESCENDING), ("downloads", DESCENDING)]) \
                        .limit(limit)
    return _serial_list(list(cursor))


def upvote_note(note_id: str) -> bool:
    """
    FIX 3: removed unused user_id param (main.py calls upvote_note(note_id)).
    Returns bool so caller knows if it succeeded.
    """
    try:
        notes_col().update_one(
            {"_id": ObjectId(note_id)},
            {"$inc": {"upvotes": 1, "quality_score": 2}},
        )
        return True
    except Exception:
        return False


def downvote_note(note_id: str) -> bool:
    """FIX 7: was missing."""
    try:
        notes_col().update_one(
            {"_id": ObjectId(note_id)},
            {"$inc": {"downvotes": 1, "quality_score": -1}},
        )
        return True
    except Exception:
        return False


def increment_downloads(note_id: str) -> None:
    """FIX 7: was missing — called when a note is fetched via /api/notes/{id}."""
    try:
        notes_col().update_one(
            {"_id": ObjectId(note_id)},
            {"$inc": {"downloads": 1}},
        )
    except Exception:
        pass


def get_user_notes(uploaded_by: str, limit: int = 20) -> list:
    """FIX 7: was missing — used by profile page and leaderboard rank calc."""
    cursor = notes_col().find(
        {"uploaded_by": uploaded_by},
        {"embedding": 0},
    ).sort("created_at", DESCENDING).limit(limit)
    return _serial_list(list(cursor))


# ══════════════════════════════════════════════════════════════════════════════
#  USERS
# ══════════════════════════════════════════════════════════════════════════════

def upsert_user(user_data: dict) -> str:
    """FIX 7: was missing — create or update a user by user_id."""
    user_id = user_data.get("user_id")
    if not user_id:
        raise ValueError("user_data must contain user_id")
    users_col().update_one(
        {"user_id": user_id},
        {
            "$set":         {k: v for k, v in user_data.items() if k != "points"},
            "$setOnInsert": {
                "points":     0,
                "badges":     [],
                "created_at": datetime.utcnow(),
                "study_streak": 0,
            },
        },
        upsert=True,
    )
    doc = users_col().find_one({"user_id": user_id}, {"_id": 1})
    return str(doc["_id"]) if doc else ""


def get_user_by_id(user_id: str) -> dict | None:
    """FIX 7: was missing — used by profile page. Includes merged memory."""
    doc = users_col().find_one({"user_id": user_id})
    if not doc:
        return None
    mem = user_memory_col().find_one({"user_id": user_id}) or {}
    doc["weak_topics"] = mem.get("weak_topics", [])
    doc["strong_topics"] = mem.get("strong_topics", [])
    doc["last_studied"] = mem.get("last_studied", "Nothing yet")
    return _serial(doc)


def update_user_points(user_id: str, delta: int, badge: str = "") -> bool:
    """FIX 7: was missing — awards points and optional badge."""
    if not user_id or user_id == "anonymous":
        return False
    update: dict = {"$inc": {"points": delta}}
    if badge:
        update["$addToSet"] = {"badges": badge}
    try:
        users_col().update_one({"user_id": user_id}, update, upsert=True)
        return True
    except Exception:
        return False


def get_leaderboard(college: str = "", limit: int = 10) -> list:
    """
    FIX 2: original used college=None which added None to query.
    FIX 7: include user_id so frontend can highlight current user.
    """
    query = {}
    if college:
        query["college"] = {"$regex": college, "$options": "i"}
    cursor = users_col().find(
        query,
        {"_id": 0, "user_id": 1, "name": 1, "college": 1,
         "points": 1, "badges": 1, "department": 1},
    ).sort("points", DESCENDING).limit(limit)
    return list(cursor)


def get_user_memory(user_id: str) -> dict:
    mem = user_memory_col().find_one({"user_id": user_id})
    if mem: return _serial(mem)
    return {"weak_topics": [], "strong_topics": [], "last_studied": "Nothing yet"}

def get_learning_patterns(user_id: str) -> list:
    cursor = learning_patterns_col().find({"user_id": user_id}).sort("detected_at", DESCENDING).limit(5)
    return [c["pattern"] for c in cursor]

def update_academic_memory(user_id: str, topic: str, score: float) -> str:
    """Updates quiz history, strong/weak topics based on score (0 to 100)."""
    if not user_id or user_id == "anonymous":
        return "Anonymous user, cannot save memory."
    
    quiz_results_col().insert_one({
        "user_id": user_id,
        "topic": topic,
        "score": score,
        "date": datetime.utcnow()
    })
    
    update_op = {}
    if score >= 75:
        update_op["$addToSet"] = {"strong_topics": topic}
        update_op["$pull"] = {"weak_topics": topic}
    elif score <= 50:
        update_op["$addToSet"] = {"weak_topics": topic}
        update_op["$pull"] = {"strong_topics": topic}
    else:
        update_op["$pull"] = {"weak_topics": topic}

    if update_op:
        user_memory_col().update_one(
            {"user_id": user_id},
            {**update_op, "$setOnInsert": {"user_id": user_id}},
            upsert=True
        )
    return f"Recorded {score}% in {topic} to quiz_results collection. Memory updated!"

def record_study_session(user_id: str, topic: str) -> str:
    if not user_id or user_id == "anonymous": return "Anonymous"
    study_history_col().insert_one({
        "user_id": user_id,
        "topic": topic,
        "date": datetime.utcnow()
    })
    user_memory_col().update_one(
        {"user_id": user_id},
        {"$set": {"last_studied": topic}, "$setOnInsert": {"weak_topics": [], "strong_topics": []}},
        upsert=True
    )
    return f"Recorded study session for {topic} in study_history collection."

def add_learning_pattern(user_id: str, pattern: str) -> str:
    if not user_id or user_id == "anonymous": return "Anonymous"
    learning_patterns_col().insert_one({
        "user_id": user_id,
        "pattern": pattern,
        "detected_at": datetime.utcnow()
    })
    return f"Learning pattern logged to learning_patterns collection."


# ══════════════════════════════════════════════════════════════════════════════
#  COMMENTS
# ══════════════════════════════════════════════════════════════════════════════

def add_comment(note_id: str, user_id: str,
                user_name: str, content: str) -> str:
    """FIX 7 + FIX 8: was missing, collection was never defined."""
    doc = {
        "note_id":    note_id,
        "user_id":    user_id,
        "user_name":  user_name,
        "content":    content,
        "created_at": datetime.utcnow(),
        "upvotes":    0,
    }
    result = comments_col().insert_one(doc)
    # Bump quality score for the note
    try:
        notes_col().update_one(
            {"_id": ObjectId(note_id)},
            {"$inc": {"quality_score": 1}},
        )
    except Exception:
        pass
    return str(result.inserted_id)


def get_comments(note_id: str) -> list:
    """FIX 7: was missing."""
    cursor = comments_col().find({"note_id": note_id}) \
                           .sort("created_at", DESCENDING) \
                           .limit(50)
    return _serial_list(list(cursor))


# ══════════════════════════════════════════════════════════════════════════════
#  NOTE REQUESTS
# ══════════════════════════════════════════════════════════════════════════════

def create_request(data: dict) -> str:
    """FIX 7: was missing."""
    data["created_at"]   = datetime.utcnow()
    data["status"]       = "open"
    data["fulfillments"] = 0
    result = requests_col().insert_one(data)
    return str(result.inserted_id)


def get_requests(college: str = "", limit: int = 20) -> list:
    """FIX 7: was missing. FIX 2: empty-string guard."""
    query = {}
    if college:
        query["college"] = {"$regex": college, "$options": "i"}
    cursor = requests_col().find(query) \
                           .sort("created_at", DESCENDING) \
                           .limit(limit)
    return _serial_list(list(cursor))


def fulfill_request(request_id: str) -> bool:
    """FIX 7: was missing."""
    try:
        requests_col().update_one(
            {"_id": ObjectId(request_id)},
            {"$inc": {"fulfillments": 1}},
        )
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  FORUM
# ══════════════════════════════════════════════════════════════════════════════

def create_forum_post(data: dict) -> str:
    """FIX 7: was missing."""
    data["created_at"] = datetime.utcnow()
    data["replies"]    = 0
    data["upvotes"]    = 0
    result = forum_col().insert_one(data)
    return str(result.inserted_id)


def get_forum_posts(college: str = "", limit: int = 20) -> list:
    """FIX 7: was missing. FIX 2: empty-string guard."""
    query = {}
    if college:
        query["college"] = {"$regex": college, "$options": "i"}
    cursor = forum_col().find(query) \
                        .sort("created_at", DESCENDING) \
                        .limit(limit)
    return _serial_list(list(cursor))


def upvote_forum_post(post_id: str) -> bool:
    """FIX 7: was missing."""
    try:
        forum_col().update_one(
            {"_id": ObjectId(post_id)},
            {"$inc": {"upvotes": 1}},
        )
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  ACADEMIC MEMORY — called by agent_tools.py dispatch
# ══════════════════════════════════════════════════════════════════════════════

def update_academic_memory(user_id: str, topic: str, score: float) -> str:
    """
    Update the user's strong/weak topics based on quiz score.
    Score >= 70 → strong topic. Score < 70 → weak topic.
    Also writes to quiz_results_col for audit history.
    """
    if not user_id or user_id == "anonymous":
        return "Memory not saved — anonymous session."
    try:
        is_strong = score >= 70
        field = "strong_topics" if is_strong else "weak_topics"
        opposite = "weak_topics" if is_strong else "strong_topics"

        # Add to correct list, remove from opposite list
        user_memory_col().update_one(
            {"user_id": user_id},
            {
                "$addToSet": {field: topic},
                "$pull":     {opposite: topic},
                "$set":      {"last_studied": topic, "updated_at": datetime.utcnow()},
            },
            upsert=True,
        )
        # Persist quiz result for history
        quiz_results_col().insert_one({
            "user_id":    user_id,
            "topic":      topic,
            "score":      score,
            "created_at": datetime.utcnow(),
        })
        label = "strong topic" if is_strong else "weak topic"
        return f"Academic Memory updated! '{topic}' marked as a {label} ({score:.0f}%)."
    except Exception as e:
        return f"Memory update failed: {e}"


def record_study_session(user_id: str, topic: str) -> str:
    """Record that the student has studied a topic (updates last_studied)."""
    if not user_id or user_id == "anonymous":
        return "Session not saved — anonymous."
    try:
        user_memory_col().update_one(
            {"user_id": user_id},
            {
                "$set":      {"last_studied": topic, "updated_at": datetime.utcnow()},
                "$addToSet": {"study_history": topic},
            },
            upsert=True,
        )
        study_history_col().insert_one({
            "user_id":    user_id,
            "topic":      topic,
            "created_at": datetime.utcnow(),
        })
        return f"Study session recorded: '{topic}'."
    except Exception as e:
        return f"Session record failed: {e}"


def add_learning_pattern(user_id: str, pattern: str) -> str:
    """Log an observed learning pattern for this user."""
    if not user_id or user_id == "anonymous":
        return "Pattern not saved — anonymous."
    try:
        learning_patterns_col().insert_one({
            "user_id":    user_id,
            "pattern":    pattern,
            "created_at": datetime.utcnow(),
        })
        return f"Learning pattern logged: '{pattern}'."
    except Exception as e:
        return f"Pattern log failed: {e}"