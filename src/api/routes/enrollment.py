"""
Enrollment API endpoints.

Handles speaker enrollment with audio samples.
"""

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from config.logging_config import get_logger
from src.api.dependencies import get_identification_engine
from src.api.schemas import (
    EnrollmentResponse,
    AddSamplesResponse,
    SimilarSpeakerWarning,
)
from src.identification.engine import IdentificationEngine

logger = get_logger(__name__)

router = APIRouter(prefix="/enroll", tags=["enrollment"])


@router.post(
    "",
    response_model=EnrollmentResponse,
    summary="Enroll a new speaker",
    description="Enroll a new speaker with audio samples. Requires at least 3 audio samples.",
)
async def enroll_speaker(
    employee_id: str = Form(..., description="Unique employee identifier"),
    name: str = Form(..., description="Speaker's display name"),
    department: str = Form(None, description="Optional department"),
    audio_files: List[UploadFile] = File(..., description="Audio sample files (min 3)"),
    engine: IdentificationEngine = Depends(get_identification_engine),
) -> EnrollmentResponse:
    """
    Enroll a new speaker with audio samples.

    - Requires at least 3 audio samples for reliable enrollment
    - Audio should be clear speech, 2-10 seconds each
    - Returns warnings if similar speakers already exist
    """
    logger.info(
        "enrollment_request",
        employee_id=employee_id,
        name=name,
        num_samples=len(audio_files),
    )

    # Read audio files
    audio_samples = []
    for file in audio_files:
        try:
            content = await file.read()
            audio_samples.append(content)
        except Exception as e:
            logger.error("audio_read_error", filename=file.filename, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read audio file {file.filename}: {e}",
            )

    # Perform enrollment
    try:
        result = await engine.enroll(
            employee_id=employee_id,
            name=name,
            audio_samples=audio_samples,
            department=department,
        )
    except Exception as e:
        logger.error("enrollment_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {e}",
        )

    # Convert similar speakers to response format
    similar_warnings = [
        SimilarSpeakerWarning(
            speaker_id=s["speaker_id"],
            name=s["name"],
            similarity=s["similarity"],
        )
        for s in result.similar_speakers
    ]

    return EnrollmentResponse(
        success=result.success,
        message=result.message,
        speaker_id=result.speaker_id if result.success else None,
        speaker_name=result.speaker_name,
        employee_id=result.employee_id,
        embeddings_stored=result.embeddings_stored,
        consistency_score=result.consistency_score,
        similar_speakers=similar_warnings,
    )


@router.post(
    "/{speaker_id}/add-samples",
    response_model=AddSamplesResponse,
    summary="Add samples to existing speaker",
    description="Add additional audio samples to an existing enrolled speaker.",
)
async def add_samples(
    speaker_id: str,
    audio_files: List[UploadFile] = File(..., description="Additional audio samples"),
    engine: IdentificationEngine = Depends(get_identification_engine),
) -> AddSamplesResponse:
    """
    Add additional audio samples to an existing speaker.

    Useful for improving recognition accuracy over time.
    """
    logger.info(
        "add_samples_request",
        speaker_id=speaker_id,
        num_samples=len(audio_files),
    )

    # Read audio files
    audio_samples = []
    for file in audio_files:
        try:
            content = await file.read()
            audio_samples.append(content)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read audio file: {e}",
            )

    # Add samples
    try:
        result = await engine.add_samples(
            speaker_id=speaker_id,
            audio_samples=audio_samples,
        )
    except Exception as e:
        logger.error("add_samples_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add samples: {e}",
        )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"],
        )

    return AddSamplesResponse(
        success=True,
        speaker_id=speaker_id,
        samples_added=result["samples_added"],
        total_samples=result["total_samples"],
    )
