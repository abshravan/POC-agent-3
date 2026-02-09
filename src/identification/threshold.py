"""
Threshold management for speaker identification.

Provides adaptive and configurable thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdConfig:
    """Threshold configuration."""

    identification_threshold: float = 0.55
    enrollment_verification_threshold: float = 0.70
    high_confidence_threshold: float = 0.75
    unknown_rejection_threshold: float = 0.40


class ThresholdManager:
    """
    Manage identification thresholds.

    Provides:
    - Global default thresholds
    - Per-speaker thresholds (optional)
    - Threshold recommendations based on data
    """

    def __init__(
        self,
        config: Optional[ThresholdConfig] = None,
    ):
        """
        Initialize threshold manager.

        Args:
            config: Threshold configuration (uses settings if None)
        """
        settings = get_settings()

        if config:
            self._config = config
        else:
            self._config = ThresholdConfig(
                identification_threshold=settings.identification_threshold,
                enrollment_verification_threshold=settings.enrollment_verification_threshold,
                high_confidence_threshold=settings.high_confidence_threshold,
            )

        # Per-speaker threshold overrides
        self._speaker_thresholds: Dict[str, float] = {}

        logger.info(
            "threshold_manager_initialized",
            identification=self._config.identification_threshold,
            enrollment=self._config.enrollment_verification_threshold,
            high_confidence=self._config.high_confidence_threshold,
        )

    @property
    def identification_threshold(self) -> float:
        """Get default identification threshold."""
        return self._config.identification_threshold

    @property
    def enrollment_verification_threshold(self) -> float:
        """Get enrollment verification threshold."""
        return self._config.enrollment_verification_threshold

    @property
    def high_confidence_threshold(self) -> float:
        """Get high confidence threshold."""
        return self._config.high_confidence_threshold

    @property
    def unknown_rejection_threshold(self) -> float:
        """Get threshold below which all speakers are rejected."""
        return self._config.unknown_rejection_threshold

    def get_threshold_for_speaker(
        self,
        speaker_id: str,
    ) -> float:
        """
        Get threshold for a specific speaker.

        Args:
            speaker_id: Speaker UUID

        Returns:
            Speaker-specific or default threshold
        """
        return self._speaker_thresholds.get(
            speaker_id,
            self._config.identification_threshold,
        )

    def set_speaker_threshold(
        self,
        speaker_id: str,
        threshold: float,
    ) -> None:
        """
        Set custom threshold for a speaker.

        Useful for speakers who are frequently confused with others.

        Args:
            speaker_id: Speaker UUID
            threshold: Custom threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self._speaker_thresholds[speaker_id] = threshold
        logger.info(
            "speaker_threshold_set",
            speaker_id=speaker_id,
            threshold=threshold,
        )

    def remove_speaker_threshold(
        self,
        speaker_id: str,
    ) -> None:
        """
        Remove custom threshold for a speaker.

        Args:
            speaker_id: Speaker UUID
        """
        if speaker_id in self._speaker_thresholds:
            del self._speaker_thresholds[speaker_id]
            logger.info("speaker_threshold_removed", speaker_id=speaker_id)

    def update_global_threshold(
        self,
        threshold: float,
    ) -> None:
        """
        Update the global identification threshold.

        Args:
            threshold: New threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self._config.identification_threshold = threshold
        logger.info("global_threshold_updated", threshold=threshold)

    def get_config(self) -> dict:
        """Get current threshold configuration."""
        return {
            "identification_threshold": self._config.identification_threshold,
            "enrollment_verification_threshold": self._config.enrollment_verification_threshold,
            "high_confidence_threshold": self._config.high_confidence_threshold,
            "unknown_rejection_threshold": self._config.unknown_rejection_threshold,
            "speaker_overrides": dict(self._speaker_thresholds),
        }

    def classify_confidence(
        self,
        confidence: float,
    ) -> str:
        """
        Classify confidence level.

        Args:
            confidence: Confidence score

        Returns:
            Classification string
        """
        if confidence >= self._config.high_confidence_threshold:
            return "high"
        elif confidence >= self._config.identification_threshold:
            return "medium"
        elif confidence >= self._config.unknown_rejection_threshold:
            return "low"
        else:
            return "rejected"

    def should_auto_accept(
        self,
        confidence: float,
    ) -> bool:
        """
        Check if confidence is high enough for auto-acceptance.

        Args:
            confidence: Confidence score

        Returns:
            True if should auto-accept without review
        """
        return confidence >= self._config.high_confidence_threshold

    def should_flag_for_review(
        self,
        confidence: float,
    ) -> bool:
        """
        Check if confidence suggests human review.

        Scores between threshold and high_confidence should be reviewed.

        Args:
            confidence: Confidence score

        Returns:
            True if should flag for human review
        """
        return (
            confidence >= self._config.identification_threshold
            and confidence < self._config.high_confidence_threshold
        )
