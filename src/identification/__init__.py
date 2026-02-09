"""Speaker identification module."""

from src.identification.engine import (
    IdentificationEngine,
    IdentificationResult,
    EnrollmentResult,
)
from src.identification.scorer import ConfidenceScorer, ScoredMatch
from src.identification.threshold import ThresholdManager, ThresholdConfig

__all__ = [
    "IdentificationEngine",
    "IdentificationResult",
    "EnrollmentResult",
    "ConfidenceScorer",
    "ScoredMatch",
    "ThresholdManager",
    "ThresholdConfig",
]
