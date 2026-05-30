import vertexai
from vertexai.language_models import TextEmbeddingModel
import os

def get_embedding(text: str) -> list:
    """
    Convert text to a vector embedding using Google's model.
    This is what enables semantic search in MongoDB.
    """
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    
    # Truncate text if too long
    text = text[:8000] if len(text) > 8000 else text
    
    embeddings = model.get_embeddings([text])
    return embeddings[0].values  # Returns a list of 768 floats