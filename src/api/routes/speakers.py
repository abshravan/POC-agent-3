"""
Speaker management API endpoints.

Handles speaker CRUD operations.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from config.logging_config import get_logger
from src.api.dependencies import (
    get_identification_engine,
    get_speaker_repository,
)
from src.api.schemas import (
    SpeakerResponse,
    SpeakerListResponse,
    RemovalResponse,
)
from src.database.speaker_repository import SpeakerRepository
from src.identification.engine import IdentificationEngine

logger = get_logger(__name__)

router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.get(
    "",
    response_model=SpeakerListResponse,
    summary="List enrolled speakers",
    description="Get a paginated list of enrolled speakers.",
)
async def list_speakers(
    include_inactive: bool = Query(False, description="Include deactivated speakers"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    repository: SpeakerRepository = Depends(get_speaker_repository),
) -> SpeakerListResponse:
    """
    List all enrolled speakers with pagination.
    """
    speakers = await repository.list(
        include_inactive=include_inactive,
        limit=limit,
        offset=offset,
    )

    total = await repository.count(include_inactive=include_inactive)

    return SpeakerListResponse(
        speakers=[
            SpeakerResponse(
                id=s.id,
                employee_id=s.employee_id,
                name=s.name,
                department=s.department,
                enrolled_at=s.enrolled_at,
                updated_at=s.updated_at,
                is_active=s.is_active,
                embedding_count=s.embedding_count,
                consistency_score=s.consistency_score,
            )
            for s in speakers
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{speaker_id}",
    response_model=SpeakerResponse,
    summary="Get speaker details",
    description="Get details of a specific speaker.",
)
async def get_speaker(
    speaker_id: str,
    repository: SpeakerRepository = Depends(get_speaker_repository),
) -> SpeakerResponse:
    """
    Get details of a specific speaker by ID.
    """
    speaker = await repository.get(speaker_id)

    if not speaker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker not found: {speaker_id}",
        )

    return SpeakerResponse(
        id=speaker.id,
        employee_id=speaker.employee_id,
        name=speaker.name,
        department=speaker.department,
        enrolled_at=speaker.enrolled_at,
        updated_at=speaker.updated_at,
        is_active=speaker.is_active,
        embedding_count=speaker.embedding_count,
        consistency_score=speaker.consistency_score,
    )


@router.get(
    "/by-employee/{employee_id}",
    response_model=SpeakerResponse,
    summary="Get speaker by employee ID",
    description="Get speaker details by employee ID.",
)
async def get_speaker_by_employee_id(
    employee_id: str,
    repository: SpeakerRepository = Depends(get_speaker_repository),
) -> SpeakerResponse:
    """
    Get speaker details by employee ID.
    """
    speaker = await repository.get_by_employee_id(employee_id)

    if not speaker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker not found for employee: {employee_id}",
        )

    return SpeakerResponse(
        id=speaker.id,
        employee_id=speaker.employee_id,
        name=speaker.name,
        department=speaker.department,
        enrolled_at=speaker.enrolled_at,
        updated_at=speaker.updated_at,
        is_active=speaker.is_active,
        embedding_count=speaker.embedding_count,
        consistency_score=speaker.consistency_score,
    )


@router.delete(
    "/{speaker_id}",
    response_model=RemovalResponse,
    summary="Remove a speaker",
    description="Remove a speaker from the system (deactivate or permanently delete).",
)
async def remove_speaker(
    speaker_id: str,
    hard_delete: bool = Query(
        False,
        description="Permanently delete all data (GDPR erasure)",
    ),
    engine: IdentificationEngine = Depends(get_identification_engine),
) -> RemovalResponse:
    """
    Remove a speaker from the system.

    - Default: Soft delete (deactivate, keep data for audit)
    - hard_delete=true: Permanent deletion (GDPR right to erasure)
    """
    logger.info(
        "remove_speaker_request",
        speaker_id=speaker_id,
        hard_delete=hard_delete,
    )

    result = await engine.remove_speaker(
        speaker_id=speaker_id,
        hard_delete=hard_delete,
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"],
        )

    return RemovalResponse(
        success=True,
        speaker_id=speaker_id,
        speaker_name=result.get("speaker_name"),
        embeddings_deleted=result.get("embeddings_deleted", 0),
        action=result.get("action", "deactivated"),
        message=result["message"],
    )


@router.get(
    "/search/{query}",
    response_model=list[SpeakerResponse],
    summary="Search speakers",
    description="Search speakers by name or employee ID.",
)
async def search_speakers(
    query: str,
    include_inactive: bool = Query(False),
    repository: SpeakerRepository = Depends(get_speaker_repository),
) -> list[SpeakerResponse]:
    """
    Search speakers by name or employee ID.
    """
    speakers = await repository.search(
        query=query,
        include_inactive=include_inactive,
    )

    return [
        SpeakerResponse(
            id=s.id,
            employee_id=s.employee_id,
            name=s.name,
            department=s.department,
            enrolled_at=s.enrolled_at,
            updated_at=s.updated_at,
            is_active=s.is_active,
            embedding_count=s.embedding_count,
            consistency_score=s.consistency_score,
        )
        for s in speakers
    ]
