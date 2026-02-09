"""Speaker identification module."""

from src.identification.engine import IdentificationEngine, IdentificationResult
from src.identification.scorer import ConfidenceScorer, ScoredMatch
from src.identification.threshold import ThresholdManager

__all__ = [
    "IdentificationEngine",
    "IdentificationResult",
    "ConfidenceScorer",
    "ScoredMatch",
    "ThresholdManager",
]
