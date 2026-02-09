"""
FastAPI dependency injection.

Provides shared instances of core components.
"""

from functools import lru_cache
from typing import AsyncGenerator

from config.settings import get_settings, Settings
from config.logging_config import get_logger
from src.embeddings.extractor import EmbeddingExtractor
from src.database.qdrant_client import QdrantVectorStore
from src.database.speaker_repository import SpeakerRepository
from src.identification.engine import IdentificationEngine
from src.identification.threshold import ThresholdManager

logger = get_logger(__name__)

# Global instances
_vector_store: QdrantVectorStore | None = None
_speaker_repository: SpeakerRepository | None = None
_embedding_extractor: EmbeddingExtractor | None = None
_identification_engine: IdentificationEngine | None = None
_threshold_manager: ThresholdManager | None = None


@lru_cache()
def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    return get_settings()


def get_vector_store() -> QdrantVectorStore:
    """Get vector store instance."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = QdrantVectorStore(
            collection_name=settings.qdrant_collection_name,
            path=settings.qdrant_path,
            embedding_dim=settings.embedding_dimension,
        )
    return _vector_store


def get_speaker_repository() -> SpeakerRepository:
    """Get speaker repository instance."""
    global _speaker_repository
    if _speaker_repository is None:
        _speaker_repository = SpeakerRepository()
    return _speaker_repository


def get_embedding_extractor() -> EmbeddingExtractor:
    """Get embedding extractor instance."""
    global _embedding_extractor
    if _embedding_extractor is None:
        settings = get_settings()
        _embedding_extractor = EmbeddingExtractor(
            model_name=settings.embedding_model_name,
            device=settings.device,
        )
    return _embedding_extractor


def get_threshold_manager() -> ThresholdManager:
    """Get threshold manager instance."""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = ThresholdManager()
    return _threshold_manager


def get_identification_engine() -> IdentificationEngine:
    """Get identification engine instance."""
    global _identification_engine
    if _identification_engine is None:
        _identification_engine = IdentificationEngine(
            vector_store=get_vector_store(),
            speaker_repository=get_speaker_repository(),
            embedding_extractor=get_embedding_extractor(),
            threshold_manager=get_threshold_manager(),
        )
    return _identification_engine


async def initialize_services() -> None:
    """Initialize all services on startup."""
    logger.info("initializing_services")

    # Initialize vector store
    vector_store = get_vector_store()
    await vector_store.initialize()

    # Preload embedding model (optional, can be lazy)
    # This triggers model download if not cached
    # extractor = get_embedding_extractor()
    # _ = extractor.model

    logger.info("services_initialized")


async def shutdown_services() -> None:
    """Cleanup services on shutdown."""
    global _vector_store

    logger.info("shutting_down_services")

    if _vector_store is not None:
        await _vector_store.close()

    logger.info("services_shutdown_complete")
