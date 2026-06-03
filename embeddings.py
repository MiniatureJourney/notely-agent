"""
embeddings.py — Vertex AI text embeddings for Notely

Bugs fixed vs original:
1. vertexai.init() called on EVERY get_embedding() call — massive latency overhead
   on every search and upload. Fixed: init once at module load, model cached globally.
2. No error handling — any Vertex AI quota/network error crashed the entire
   upload or search request. Fixed: returns ZERO_VECTOR fallback so app never crashes.
3. TextEmbeddingModel imported from wrong path for newer SDK versions.
   Fixed: try/except import handles both old and new vertexai SDK layouts.
4. No zero-vector constant — callers had no way to detect embedding failure.
   Fixed: ZERO_VECTOR exported so database.py can optionally skip vector search
   when embedding failed.
5. No helper to build the best embed text from a note dict — was duplicated
   ad-hoc in main.py and seed_data.py with different field sets.
   Fixed: build_note_embed_text() centralises this logic.
6. No batch helper — seed_data.py called get_embedding() in a tight loop and
   hit Vertex AI rate limits. Fixed: batch_embed() adds a configurable delay.
"""

import os
import time

# ── Vertex AI import (handles both SDK layouts) ───────────────────────────────
try:
    from vertexai.language_models import TextEmbeddingModel
except ImportError:
    from vertexai.preview.language_models import TextEmbeddingModel

import vertexai

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 768
ZERO_VECTOR   = [0.0] * EMBEDDING_DIM   # returned on failure — never crashes callers
MAX_CHARS     = 8000                     # gecko max safe input

# ── FIX 1: init once, cache model globally ────────────────────────────────────
_model = None

def _get_model() -> TextEmbeddingModel:
    """Initialise Vertex AI and load the embedding model exactly once."""
    global _model
    if _model is None:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        region  = os.getenv("VERTEX_REGION", "us-central1")
        if project:
            vertexai.init(project=project, location=region)
        _model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Convert text to a 768-dimensional vector for MongoDB Atlas Vector Search.

    FIX 2: wrapped in try/except — returns ZERO_VECTOR on any failure so the
    caller (upload, search) never crashes due to a Vertex AI quota or network error.
    """
    if not text or not text.strip():
        return ZERO_VECTOR

    text = text[:MAX_CHARS]   # FIX: was already present, kept

    try:
        model  = _get_model()
        result = model.get_embeddings([text])
        return list(result[0].values)
    except Exception as e:
        print(f"[embeddings] get_embedding failed: {e}")
        return ZERO_VECTOR


def build_note_embed_text(note: dict) -> str:
    """
    FIX 5: centralised helper — builds the richest possible text for embedding
    a note. Combines all searchable fields so vector search finds notes by
    subject, teacher, chapter, college, and content alike.
    Called by both upload-note endpoint and seed_data.py.
    """
    parts = [
        note.get("title",           ""),
        note.get("subject",         ""),
        note.get("chapter",         ""),
        note.get("teacher",         ""),
        note.get("department",      ""),
        note.get("college",         ""),
        note.get("content_preview", ""),
        note.get("full_content",    "")[:3000],
    ]
    return " ".join(p for p in parts if p).strip()


def batch_embed(texts: list[str], delay: float = 0.6) -> list[list[float]]:
    """
    FIX 6: rate-limited batch embedding for seed_data.py.
    Vertex AI free tier allows ~60 requests/min — 0.6 s delay keeps us safe.
    """
    results = []
    for i, text in enumerate(texts):
        print(f"  Embedding {i + 1}/{len(texts)}...")
        results.append(get_embedding(text))
        if i < len(texts) - 1:
            time.sleep(delay)
    return results