"""
Confidence scoring for speaker identification.

Provides calibrated confidence scores from raw similarity scores.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredMatch:
    """Scored speaker match result."""

    speaker_id: str
    score: float
    match_count: int
    payload: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "score": round(self.score, 4),
            "match_count": self.match_count,
            "name": self.payload.get("name"),
            "employee_id": self.payload.get("employee_id"),
        }


class ConfidenceScorer:
    """
    Compute calibrated confidence scores from similarity scores.

    Provides additional scoring features:
    - Score normalization
    - Multi-match aggregation
    - Distinctiveness scoring
    """

    def __init__(
        self,
        min_threshold: float = 0.3,
        max_threshold: float = 0.9,
    ):
        """
        Initialize the confidence scorer.

        Args:
            min_threshold: Minimum similarity for any confidence
            max_threshold: Similarity for 100% confidence
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def compute_confidence(
        self,
        similarity: float,
    ) -> float:
        """
        Convert raw similarity score to calibrated confidence.

        Maps similarity from [min_threshold, max_threshold] to [0, 1].

        Args:
            similarity: Raw cosine similarity score

        Returns:
            Calibrated confidence score (0 to 1)
        """
        if similarity < self.min_threshold:
            return 0.0

        if similarity >= self.max_threshold:
            return 1.0

        # Linear mapping
        confidence = (similarity - self.min_threshold) / (
            self.max_threshold - self.min_threshold
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def aggregate_scores(
        self,
        scores: List[float],
        method: str = "weighted_average",
    ) -> float:
        """
        Aggregate multiple similarity scores for a speaker.

        Args:
            scores: List of similarity scores
            method: Aggregation method ('average', 'max', 'weighted_average')

        Returns:
            Aggregated score
        """
        if not scores:
            return 0.0

        scores_array = np.array(scores)

        if method == "max":
            return float(np.max(scores_array))

        elif method == "average":
            return float(np.mean(scores_array))

        elif method == "weighted_average":
            # Weight by score^2 to emphasize higher scores
            weights = scores_array ** 2
            if np.sum(weights) > 0:
                return float(np.average(scores_array, weights=weights))
            return float(np.mean(scores_array))

        else:
            return float(np.mean(scores_array))

    def compute_distinctiveness(
        self,
        top_score: float,
        second_score: float,
        min_gap: float = 0.05,
    ) -> float:
        """
        Compute how distinctive the top match is from the second.

        A large gap indicates high confidence in the identification.
        A small gap suggests ambiguity between similar speakers.

        Args:
            top_score: Score of best match
            second_score: Score of second best match
            min_gap: Minimum gap for full distinctiveness

        Returns:
            Distinctiveness score (0 to 1)
        """
        gap = top_score - second_score

        if gap >= min_gap:
            return 1.0

        if gap <= 0:
            return 0.0

        return gap / min_gap

    def apply_distinctiveness_adjustment(
        self,
        confidence: float,
        distinctiveness: float,
        weight: float = 0.2,
    ) -> float:
        """
        Adjust confidence based on distinctiveness.

        Low distinctiveness reduces confidence even for high scores.

        Args:
            confidence: Base confidence score
            distinctiveness: Distinctiveness score
            weight: How much to weight distinctiveness

        Returns:
            Adjusted confidence score
        """
        adjustment = 1.0 - (weight * (1.0 - distinctiveness))
        return float(np.clip(confidence * adjustment, 0.0, 1.0))

    def score_with_context(
        self,
        matches: List[ScoredMatch],
    ) -> List[ScoredMatch]:
        """
        Enhance match scores with contextual information.

        Applies:
        - Confidence calibration
        - Distinctiveness adjustment
        - Match count bonus

        Args:
            matches: List of scored matches (sorted by score)

        Returns:
            List of enhanced scored matches
        """
        if not matches:
            return []

        # Get top two scores for distinctiveness
        top_score = matches[0].score
        second_score = matches[1].score if len(matches) > 1 else 0.0

        distinctiveness = self.compute_distinctiveness(top_score, second_score)

        enhanced = []
        for i, match in enumerate(matches):
            # Base confidence
            confidence = self.compute_confidence(match.score)

            # Apply distinctiveness only to top match
            if i == 0:
                confidence = self.apply_distinctiveness_adjustment(
                    confidence, distinctiveness
                )

            # Small bonus for multiple embedding matches
            if match.match_count > 1:
                match_bonus = min(0.05, (match.match_count - 1) * 0.01)
                confidence = min(1.0, confidence + match_bonus)

            enhanced.append(
                ScoredMatch(
                    speaker_id=match.speaker_id,
                    score=confidence,
                    match_count=match.match_count,
                    payload=match.payload,
                )
            )

        return enhanced
