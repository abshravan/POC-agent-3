"""Database module for speaker identification."""

from src.database.vector_store import VectorStore, SearchResult
from src.database.qdrant_client import QdrantVectorStore
from src.database.speaker_repository import SpeakerRepository, Speaker

__all__ = [
    "VectorStore",
    "SearchResult",
    "QdrantVectorStore",
    "SpeakerRepository",
    "Speaker",
]
