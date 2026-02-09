"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============ Common Schemas ============

class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ============ Speaker Schemas ============

class SpeakerBase(BaseModel):
    """Base speaker schema."""

    employee_id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=255)
    department: Optional[str] = Field(None, max_length=100)


class SpeakerCreate(SpeakerBase):
    """Schema for creating a speaker (enrollment)."""

    pass


class SpeakerResponse(SpeakerBase):
    """Speaker response schema."""

    id: str
    enrolled_at: datetime
    updated_at: datetime
    is_active: bool
    embedding_count: int
    consistency_score: float

    class Config:
        from_attributes = True


class SpeakerListResponse(BaseModel):
    """List of speakers response."""

    speakers: List[SpeakerResponse]
    total: int
    limit: int
    offset: int


# ============ Enrollment Schemas ============

class EnrollmentRequest(SpeakerCreate):
    """Enrollment request (audio files sent separately as multipart)."""

    pass


class SimilarSpeakerWarning(BaseModel):
    """Warning about similar existing speaker."""

    speaker_id: str
    name: str
    similarity: float


class EnrollmentResponse(BaseModel):
    """Enrollment response."""

    success: bool
    message: str
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    employee_id: Optional[str] = None
    embeddings_stored: int = 0
    consistency_score: float = 0.0
    similar_speakers: List[SimilarSpeakerWarning] = []


class AddSamplesResponse(BaseModel):
    """Response for adding samples to existing speaker."""

    success: bool
    speaker_id: str
    samples_added: int
    total_samples: int


# ============ Identification Schemas ============

class SpeakerMatch(BaseModel):
    """Individual speaker match in identification."""

    speaker_id: str
    score: float
    match_count: int
    name: Optional[str] = None
    employee_id: Optional[str] = None


class AudioQualityInfo(BaseModel):
    """Audio quality information."""

    is_acceptable: bool
    rms_energy: float
    snr_db: float
    duration: float
    quality_score: float
    rejection_reasons: List[str] = []


class IdentificationResponse(BaseModel):
    """Identification response."""

    speaker_id: Optional[str]
    speaker_name: Optional[str]
    employee_id: Optional[str]
    confidence: float
    is_identified: bool
    all_matches: List[SpeakerMatch] = []
    processing_time_ms: float
    audio_duration: float
    audio_quality: AudioQualityInfo
    threshold_used: float


# ============ Admin Schemas ============

class ThresholdConfig(BaseModel):
    """Threshold configuration."""

    identification_threshold: float = Field(..., ge=0.0, le=1.0)
    enrollment_verification_threshold: float = Field(..., ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(..., ge=0.0, le=1.0)


class ThresholdResponse(BaseModel):
    """Threshold configuration response."""

    identification_threshold: float
    enrollment_verification_threshold: float
    high_confidence_threshold: float
    unknown_rejection_threshold: float
    speaker_overrides: Dict[str, float] = {}


class UpdateThresholdRequest(BaseModel):
    """Request to update identification threshold."""

    threshold: float = Field(..., ge=0.0, le=1.0)


class SpeakerThresholdRequest(BaseModel):
    """Request to set per-speaker threshold."""

    speaker_id: str
    threshold: float = Field(..., ge=0.0, le=1.0)


# ============ Removal Schemas ============

class RemovalResponse(BaseModel):
    """Speaker removal response."""

    success: bool
    speaker_id: str
    speaker_name: Optional[str] = None
    embeddings_deleted: int = 0
    action: str  # 'deactivated' or 'permanently deleted'
    message: str


# ============ Stats Schemas ============

class SystemStatsResponse(BaseModel):
    """System statistics response."""

    total_speakers: int
    active_speakers: int
    total_embeddings: int
    embedding_dimension: int
    model_name: str
    device: str
