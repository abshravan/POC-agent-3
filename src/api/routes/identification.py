"""
Identification API endpoints.

Handles speaker identification from audio.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from config.logging_config import get_logger
from src.api.dependencies import get_identification_engine
from src.api.schemas import (
    IdentificationResponse,
    SpeakerMatch,
    AudioQualityInfo,
)
from src.identification.engine import IdentificationEngine

logger = get_logger(__name__)

router = APIRouter(prefix="/identify", tags=["identification"])


@router.post(
    "",
    response_model=IdentificationResponse,
    summary="Identify speaker from audio",
    description="Identify the speaker from an audio sample.",
)
async def identify_speaker(
    audio_file: UploadFile = File(..., description="Audio file to identify"),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Custom confidence threshold (uses default if not specified)",
    ),
    engine: IdentificationEngine = Depends(get_identification_engine),
) -> IdentificationResponse:
    """
    Identify the speaker from an audio sample.

    - Audio should be 1-30 seconds of clear speech
    - Returns confidence score and all potential matches
    - Uses configured threshold if not specified
    """
    logger.info(
        "identification_request",
        filename=audio_file.filename,
        threshold=threshold,
    )

    # Read audio file
    try:
        audio_content = await audio_file.read()
    except Exception as e:
        logger.error("audio_read_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {e}",
        )

    # Perform identification
    try:
        result = await engine.identify(
            audio=audio_content,
            threshold=threshold,
        )
    except Exception as e:
        logger.error("identification_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identification failed: {e}",
        )

    # Convert matches to response format
    matches = [
        SpeakerMatch(
            speaker_id=m.speaker_id,
            score=m.score,
            match_count=m.match_count,
            name=m.payload.get("name"),
            employee_id=m.payload.get("employee_id"),
        )
        for m in result.all_matches[:5]
    ]

    # Audio quality info
    quality_info = AudioQualityInfo(
        is_acceptable=result.audio_quality.is_acceptable,
        rms_energy=result.audio_quality.rms_energy,
        snr_db=result.audio_quality.snr_db,
        duration=result.audio_quality.duration,
        quality_score=result.audio_quality.quality_score,
        rejection_reasons=result.audio_quality.rejection_reasons,
    )

    return IdentificationResponse(
        speaker_id=result.speaker_id,
        speaker_name=result.speaker_name,
        employee_id=result.employee_id,
        confidence=result.confidence,
        is_identified=result.is_identified,
        all_matches=matches,
        processing_time_ms=result.processing_time_ms,
        audio_duration=result.audio_duration,
        audio_quality=quality_info,
        threshold_used=result.threshold_used,
    )


@router.post(
    "/batch",
    response_model=list[IdentificationResponse],
    summary="Identify speakers from multiple audio files",
    description="Identify speakers from multiple audio samples.",
)
async def identify_speakers_batch(
    audio_files: list[UploadFile] = File(..., description="Audio files to identify"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    engine: IdentificationEngine = Depends(get_identification_engine),
) -> list[IdentificationResponse]:
    """
    Identify speakers from multiple audio samples.

    Returns results for each audio file in order.
    """
    results = []

    for audio_file in audio_files:
        try:
            audio_content = await audio_file.read()
            result = await engine.identify(
                audio=audio_content,
                threshold=threshold,
            )

            matches = [
                SpeakerMatch(
                    speaker_id=m.speaker_id,
                    score=m.score,
                    match_count=m.match_count,
                    name=m.payload.get("name"),
                    employee_id=m.payload.get("employee_id"),
                )
                for m in result.all_matches[:5]
            ]

            quality_info = AudioQualityInfo(
                is_acceptable=result.audio_quality.is_acceptable,
                rms_energy=result.audio_quality.rms_energy,
                snr_db=result.audio_quality.snr_db,
                duration=result.audio_quality.duration,
                quality_score=result.audio_quality.quality_score,
                rejection_reasons=result.audio_quality.rejection_reasons,
            )

            results.append(
                IdentificationResponse(
                    speaker_id=result.speaker_id,
                    speaker_name=result.speaker_name,
                    employee_id=result.employee_id,
                    confidence=result.confidence,
                    is_identified=result.is_identified,
                    all_matches=matches,
                    processing_time_ms=result.processing_time_ms,
                    audio_duration=result.audio_duration,
                    audio_quality=quality_info,
                    threshold_used=result.threshold_used,
                )
            )

        except Exception as e:
            logger.error(
                "batch_identification_error",
                filename=audio_file.filename,
                error=str(e),
            )
            # Add error result
            results.append(
                IdentificationResponse(
                    speaker_id=None,
                    speaker_name=None,
                    employee_id=None,
                    confidence=0.0,
                    is_identified=False,
                    all_matches=[],
                    processing_time_ms=0.0,
                    audio_duration=0.0,
                    audio_quality=AudioQualityInfo(
                        is_acceptable=False,
                        rms_energy=0.0,
                        snr_db=0.0,
                        duration=0.0,
                        quality_score=0.0,
                        rejection_reasons=[f"Error: {str(e)}"],
                    ),
                    threshold_used=threshold or 0.55,
                )
            )

    return results
