"""
Admin API endpoints.

Handles system configuration and monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from config.settings import get_settings
from config.logging_config import get_logger
from src.api.dependencies import (
    get_embedding_extractor,
    get_speaker_repository,
    get_threshold_manager,
    get_vector_store,
)
from src.api.schemas import (
    ThresholdResponse,
    UpdateThresholdRequest,
    SpeakerThresholdRequest,
    SystemStatsResponse,
)
from src.database.qdrant_client import QdrantVectorStore
from src.database.speaker_repository import SpeakerRepository
from src.embeddings.extractor import EmbeddingExtractor
from src.identification.threshold import ThresholdManager

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/thresholds",
    response_model=ThresholdResponse,
    summary="Get threshold configuration",
    description="Get current threshold configuration for identification.",
)
async def get_thresholds(
    manager: ThresholdManager = Depends(get_threshold_manager),
) -> ThresholdResponse:
    """
    Get current threshold configuration.
    """
    config = manager.get_config()

    return ThresholdResponse(
        identification_threshold=config["identification_threshold"],
        enrollment_verification_threshold=config["enrollment_verification_threshold"],
        high_confidence_threshold=config["high_confidence_threshold"],
        unknown_rejection_threshold=config["unknown_rejection_threshold"],
        speaker_overrides=config["speaker_overrides"],
    )


@router.put(
    "/thresholds",
    response_model=ThresholdResponse,
    summary="Update identification threshold",
    description="Update the global identification threshold.",
)
async def update_threshold(
    request: UpdateThresholdRequest,
    manager: ThresholdManager = Depends(get_threshold_manager),
) -> ThresholdResponse:
    """
    Update the global identification threshold.
    """
    logger.info("threshold_update", new_threshold=request.threshold)

    manager.update_global_threshold(request.threshold)

    config = manager.get_config()
    return ThresholdResponse(
        identification_threshold=config["identification_threshold"],
        enrollment_verification_threshold=config["enrollment_verification_threshold"],
        high_confidence_threshold=config["high_confidence_threshold"],
        unknown_rejection_threshold=config["unknown_rejection_threshold"],
        speaker_overrides=config["speaker_overrides"],
    )


@router.put(
    "/thresholds/speaker",
    response_model=ThresholdResponse,
    summary="Set per-speaker threshold",
    description="Set a custom threshold for a specific speaker.",
)
async def set_speaker_threshold(
    request: SpeakerThresholdRequest,
    manager: ThresholdManager = Depends(get_threshold_manager),
    repository: SpeakerRepository = Depends(get_speaker_repository),
) -> ThresholdResponse:
    """
    Set a custom threshold for a specific speaker.

    Useful for speakers who are frequently confused with others.
    """
    # Verify speaker exists
    speaker = await repository.get(request.speaker_id)
    if not speaker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker not found: {request.speaker_id}",
        )

    logger.info(
        "speaker_threshold_set",
        speaker_id=request.speaker_id,
        threshold=request.threshold,
    )

    manager.set_speaker_threshold(request.speaker_id, request.threshold)

    config = manager.get_config()
    return ThresholdResponse(
        identification_threshold=config["identification_threshold"],
        enrollment_verification_threshold=config["enrollment_verification_threshold"],
        high_confidence_threshold=config["high_confidence_threshold"],
        unknown_rejection_threshold=config["unknown_rejection_threshold"],
        speaker_overrides=config["speaker_overrides"],
    )


@router.delete(
    "/thresholds/speaker/{speaker_id}",
    response_model=ThresholdResponse,
    summary="Remove per-speaker threshold",
    description="Remove custom threshold for a speaker.",
)
async def remove_speaker_threshold(
    speaker_id: str,
    manager: ThresholdManager = Depends(get_threshold_manager),
) -> ThresholdResponse:
    """
    Remove custom threshold for a speaker.
    """
    manager.remove_speaker_threshold(speaker_id)

    config = manager.get_config()
    return ThresholdResponse(
        identification_threshold=config["identification_threshold"],
        enrollment_verification_threshold=config["enrollment_verification_threshold"],
        high_confidence_threshold=config["high_confidence_threshold"],
        unknown_rejection_threshold=config["unknown_rejection_threshold"],
        speaker_overrides=config["speaker_overrides"],
    )


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="Get system statistics",
    description="Get system statistics including speaker and embedding counts.",
)
async def get_stats(
    repository: SpeakerRepository = Depends(get_speaker_repository),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    extractor: EmbeddingExtractor = Depends(get_embedding_extractor),
) -> SystemStatsResponse:
    """
    Get system statistics.
    """
    settings = get_settings()

    total_speakers = await repository.count(include_inactive=True)
    active_speakers = await repository.count(include_inactive=False)
    total_embeddings = await vector_store.count()

    return SystemStatsResponse(
        total_speakers=total_speakers,
        active_speakers=active_speakers,
        total_embeddings=total_embeddings,
        embedding_dimension=extractor.embedding_dimension,
        model_name=extractor.model_name,
        device=settings.device,
    )


@router.post(
    "/cache/clear",
    summary="Clear embedding cache",
    description="Clear the embedding cache.",
)
async def clear_cache(
    extractor: EmbeddingExtractor = Depends(get_embedding_extractor),
) -> dict:
    """
    Clear the embedding cache.
    """
    cleared = extractor.clear_cache()
    logger.info("cache_cleared", entries=cleared)

    return {
        "success": True,
        "entries_cleared": cleared,
    }
