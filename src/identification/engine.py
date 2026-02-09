"""
Core speaker identification engine.

Orchestrates the identification pipeline from audio to speaker identity.
"""

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from config.settings import get_settings
from config.logging_config import get_logger
from src.audio.preprocessing import AudioPreprocessor
from src.audio.quality import AudioQualityAssessor, QualityResult
from src.audio.vad import VoiceActivityDetector
from src.embeddings.extractor import EmbeddingExtractor
from src.database.vector_store import VectorStore, SearchResult
from src.database.speaker_repository import SpeakerRepository, Speaker
from src.identification.scorer import ConfidenceScorer, ScoredMatch
from src.identification.threshold import ThresholdManager

logger = get_logger(__name__)


@dataclass
class IdentificationResult:
    """Result of speaker identification."""

    # Identification outcome
    speaker_id: Optional[str]
    speaker_name: Optional[str]
    employee_id: Optional[str]
    confidence: float
    is_identified: bool

    # All matches for transparency
    all_matches: List[ScoredMatch]

    # Metadata
    processing_time_ms: float
    audio_duration: float
    audio_quality: QualityResult

    # Debug info
    embedding_time_ms: float
    search_time_ms: float
    threshold_used: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "employee_id": self.employee_id,
            "confidence": round(self.confidence, 4),
            "is_identified": self.is_identified,
            "all_matches": [m.to_dict() for m in self.all_matches[:5]],
            "processing_time_ms": round(self.processing_time_ms, 2),
            "audio_duration": round(self.audio_duration, 2),
            "audio_quality": self.audio_quality.to_dict(),
            "threshold_used": round(self.threshold_used, 3),
        }


@dataclass
class EnrollmentResult:
    """Result of speaker enrollment."""

    speaker_id: str
    speaker_name: str
    employee_id: str
    embeddings_stored: int
    consistency_score: float
    similar_speakers: List[Dict[str, Any]]
    success: bool
    message: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "employee_id": self.employee_id,
            "embeddings_stored": self.embeddings_stored,
            "consistency_score": round(self.consistency_score, 3),
            "similar_speakers": self.similar_speakers,
            "success": self.success,
            "message": self.message,
        }


class IdentificationEngine:
    """
    Core engine for speaker identification and enrollment.

    Orchestrates the complete pipeline from audio input to speaker identity.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        speaker_repository: SpeakerRepository,
        embedding_extractor: Optional[EmbeddingExtractor] = None,
        threshold_manager: Optional[ThresholdManager] = None,
    ):
        """
        Initialize the identification engine.

        Args:
            vector_store: Vector database for embeddings
            speaker_repository: Repository for speaker metadata
            embedding_extractor: Embedding extractor (created if not provided)
            threshold_manager: Threshold manager (created if not provided)
        """
        self.vector_store = vector_store
        self.speaker_repository = speaker_repository

        # Initialize components
        self.embedding_extractor = embedding_extractor or EmbeddingExtractor()
        self.threshold_manager = threshold_manager or ThresholdManager()
        self.scorer = ConfidenceScorer()
        self.preprocessor = AudioPreprocessor()
        self.quality_assessor = AudioQualityAssessor()
        self.vad = VoiceActivityDetector()

        settings = get_settings()
        self.min_enrollment_samples = settings.min_enrollment_samples
        self.similar_speaker_threshold = settings.similar_speaker_warning_threshold

        logger.info("identification_engine_initialized")

    async def identify(
        self,
        audio: Union[np.ndarray, bytes],
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
    ) -> IdentificationResult:
        """
        Identify speaker from audio.

        Args:
            audio: Audio data (numpy array or bytes)
            sample_rate: Audio sample rate
            threshold: Custom threshold (uses default if None)

        Returns:
            IdentificationResult with speaker identity and confidence
        """
        start_time = time.perf_counter()

        # Load and preprocess audio
        if isinstance(audio, bytes):
            audio_array, sr = self.preprocessor.load_audio(audio)
        else:
            audio_array = audio
            sr = sample_rate

        # Assess audio quality
        quality = self.quality_assessor.assess(audio_array, sr)

        if not quality.is_acceptable:
            return IdentificationResult(
                speaker_id=None,
                speaker_name=None,
                employee_id=None,
                confidence=0.0,
                is_identified=False,
                all_matches=[],
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                audio_duration=quality.duration,
                audio_quality=quality,
                embedding_time_ms=0.0,
                search_time_ms=0.0,
                threshold_used=threshold or self.threshold_manager.identification_threshold,
            )

        # Extract speech segments using VAD
        speech_audio = self.vad.extract_speech(audio_array, sr)
        if len(speech_audio) < sr:  # Less than 1 second of speech
            speech_audio = audio_array  # Use original if VAD fails

        # Preprocess
        processed = self.preprocessor.preprocess(speech_audio, sr)

        # Extract embedding
        embed_start = time.perf_counter()
        embed_result = self.embedding_extractor.extract(processed, sr)
        embedding_time = (time.perf_counter() - embed_start) * 1000

        # Search vector database
        search_start = time.perf_counter()
        search_results = await self.vector_store.search(
            vector=embed_result.embedding,
            limit=20,
        )
        search_time = (time.perf_counter() - search_start) * 1000

        # Score and aggregate results by speaker
        scored_matches = await self._score_and_aggregate(search_results)

        # Apply threshold
        used_threshold = threshold or self.threshold_manager.identification_threshold

        # Determine identification result
        if scored_matches and scored_matches[0].score >= used_threshold:
            best_match = scored_matches[0]
            speaker = await self.speaker_repository.get(best_match.speaker_id)

            return IdentificationResult(
                speaker_id=best_match.speaker_id,
                speaker_name=speaker.name if speaker else None,
                employee_id=speaker.employee_id if speaker else None,
                confidence=best_match.score,
                is_identified=True,
                all_matches=scored_matches,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                audio_duration=quality.duration,
                audio_quality=quality,
                embedding_time_ms=embedding_time,
                search_time_ms=search_time,
                threshold_used=used_threshold,
            )
        else:
            return IdentificationResult(
                speaker_id=None,
                speaker_name=None,
                employee_id=None,
                confidence=scored_matches[0].score if scored_matches else 0.0,
                is_identified=False,
                all_matches=scored_matches,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                audio_duration=quality.duration,
                audio_quality=quality,
                embedding_time_ms=embedding_time,
                search_time_ms=search_time,
                threshold_used=used_threshold,
            )

    async def enroll(
        self,
        employee_id: str,
        name: str,
        audio_samples: List[Union[np.ndarray, bytes]],
        sample_rate: int = 16000,
        department: Optional[str] = None,
    ) -> EnrollmentResult:
        """
        Enroll a new speaker with audio samples.

        Args:
            employee_id: Unique employee identifier
            name: Speaker's display name
            audio_samples: List of audio samples
            sample_rate: Audio sample rate
            department: Optional department name

        Returns:
            EnrollmentResult with enrollment status
        """
        # Validate minimum samples
        if len(audio_samples) < self.min_enrollment_samples:
            return EnrollmentResult(
                speaker_id="",
                speaker_name=name,
                employee_id=employee_id,
                embeddings_stored=0,
                consistency_score=0.0,
                similar_speakers=[],
                success=False,
                message=f"Need at least {self.min_enrollment_samples} audio samples",
            )

        # Check if employee ID already exists
        if await self.speaker_repository.exists_by_employee_id(employee_id):
            return EnrollmentResult(
                speaker_id="",
                speaker_name=name,
                employee_id=employee_id,
                embeddings_stored=0,
                consistency_score=0.0,
                similar_speakers=[],
                success=False,
                message=f"Employee ID already exists: {employee_id}",
            )

        # Extract embeddings from all samples
        embeddings = []
        for audio in audio_samples:
            if isinstance(audio, bytes):
                audio_array, _ = self.preprocessor.load_audio(audio)
            else:
                audio_array = audio

            # Check quality
            quality = self.quality_assessor.assess(audio_array, sample_rate)
            if not quality.is_acceptable:
                logger.warning(
                    "enrollment_sample_rejected",
                    reasons=quality.rejection_reasons,
                )
                continue

            # Extract embedding
            embed_result = self.embedding_extractor.extract(audio_array, sample_rate)
            embeddings.append(embed_result.embedding)

        if len(embeddings) < self.min_enrollment_samples:
            return EnrollmentResult(
                speaker_id="",
                speaker_name=name,
                employee_id=employee_id,
                embeddings_stored=0,
                consistency_score=0.0,
                similar_speakers=[],
                success=False,
                message=f"Only {len(embeddings)} valid samples after quality check",
            )

        # Check consistency
        embeddings_array = np.array(embeddings)
        centroid = np.mean(embeddings_array, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        similarities = np.dot(embeddings_array, centroid)
        consistency = float(np.mean(similarities))

        settings = get_settings()
        if consistency < settings.enrollment_consistency_threshold:
            return EnrollmentResult(
                speaker_id="",
                speaker_name=name,
                employee_id=employee_id,
                embeddings_stored=0,
                consistency_score=consistency,
                similar_speakers=[],
                success=False,
                message=f"Low sample consistency ({consistency:.2f}). Ensure all samples are from the same speaker.",
            )

        # Check for similar existing speakers
        similar_speakers = await self._find_similar_speakers(centroid)

        # Create speaker record
        speaker = await self.speaker_repository.create(
            employee_id=employee_id,
            name=name,
            department=department,
            embedding_count=len(embeddings),
            consistency_score=consistency,
        )

        # Store embeddings in vector database
        for i, embedding in enumerate(embeddings):
            await self.vector_store.insert(
                id=f"{speaker.id}_{i}",
                vector=embedding,
                payload={
                    "speaker_id": speaker.id,
                    "employee_id": employee_id,
                    "name": name,
                    "sample_index": i,
                },
            )

        logger.info(
            "speaker_enrolled",
            speaker_id=speaker.id,
            employee_id=employee_id,
            name=name,
            embeddings=len(embeddings),
            consistency=consistency,
        )

        return EnrollmentResult(
            speaker_id=speaker.id,
            speaker_name=name,
            employee_id=employee_id,
            embeddings_stored=len(embeddings),
            consistency_score=consistency,
            similar_speakers=similar_speakers,
            success=True,
            message="Enrollment successful",
        )

    async def remove_speaker(
        self,
        speaker_id: str,
        hard_delete: bool = False,
    ) -> Dict[str, Any]:
        """
        Remove a speaker from the system.

        Args:
            speaker_id: Speaker UUID
            hard_delete: If True, permanently remove all data

        Returns:
            Removal result with details
        """
        # Get speaker info
        speaker = await self.speaker_repository.get(speaker_id)
        if not speaker:
            return {
                "success": False,
                "message": f"Speaker not found: {speaker_id}",
            }

        # Delete embeddings from vector store
        deleted_count = await self.vector_store.delete(
            filter={"speaker_id": speaker_id}
        )

        # Delete or deactivate speaker record
        await self.speaker_repository.delete(speaker_id, hard_delete=hard_delete)

        action = "permanently deleted" if hard_delete else "deactivated"

        logger.info(
            "speaker_removed",
            speaker_id=speaker_id,
            employee_id=speaker.employee_id,
            embeddings_deleted=deleted_count,
            hard_delete=hard_delete,
        )

        return {
            "success": True,
            "speaker_id": speaker_id,
            "speaker_name": speaker.name,
            "embeddings_deleted": deleted_count,
            "action": action,
            "message": f"Speaker {action} successfully",
        }

    async def add_samples(
        self,
        speaker_id: str,
        audio_samples: List[Union[np.ndarray, bytes]],
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Add additional samples to an existing speaker.

        Args:
            speaker_id: Speaker UUID
            audio_samples: List of new audio samples
            sample_rate: Audio sample rate

        Returns:
            Result with number of samples added
        """
        speaker = await self.speaker_repository.get(speaker_id)
        if not speaker:
            return {
                "success": False,
                "message": f"Speaker not found: {speaker_id}",
            }

        added = 0
        for audio in audio_samples:
            if isinstance(audio, bytes):
                audio_array, _ = self.preprocessor.load_audio(audio)
            else:
                audio_array = audio

            quality = self.quality_assessor.assess(audio_array, sample_rate)
            if not quality.is_acceptable:
                continue

            embed_result = self.embedding_extractor.extract(audio_array, sample_rate)

            sample_id = f"{speaker_id}_{speaker.embedding_count + added}"
            await self.vector_store.insert(
                id=sample_id,
                vector=embed_result.embedding,
                payload={
                    "speaker_id": speaker_id,
                    "employee_id": speaker.employee_id,
                    "name": speaker.name,
                    "sample_index": speaker.embedding_count + added,
                },
            )
            added += 1

        # Update speaker embedding count
        await self.speaker_repository.update(
            speaker_id,
            embedding_count=speaker.embedding_count + added,
        )

        return {
            "success": True,
            "speaker_id": speaker_id,
            "samples_added": added,
            "total_samples": speaker.embedding_count + added,
        }

    async def _score_and_aggregate(
        self,
        search_results: List[SearchResult],
    ) -> List[ScoredMatch]:
        """
        Aggregate search results by speaker and compute final scores.

        Args:
            search_results: Raw search results from vector store

        Returns:
            List of ScoredMatch sorted by score descending
        """
        # Group by speaker
        speaker_scores: Dict[str, List[float]] = defaultdict(list)
        speaker_info: Dict[str, Dict[str, Any]] = {}

        for result in search_results:
            speaker_id = result.payload.get("speaker_id")
            if speaker_id:
                speaker_scores[speaker_id].append(result.score)
                if speaker_id not in speaker_info:
                    speaker_info[speaker_id] = result.payload

        # Compute aggregated scores
        scored_matches = []
        for speaker_id, scores in speaker_scores.items():
            # Weighted average: higher scores have more weight
            scores_array = np.array(scores)
            weights = scores_array ** 2
            aggregated_score = float(np.average(scores_array, weights=weights))

            scored_matches.append(
                ScoredMatch(
                    speaker_id=speaker_id,
                    score=aggregated_score,
                    match_count=len(scores),
                    payload=speaker_info.get(speaker_id, {}),
                )
            )

        # Sort by score descending
        scored_matches.sort(key=lambda m: m.score, reverse=True)

        return scored_matches

    async def _find_similar_speakers(
        self,
        embedding: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Find existing speakers similar to the given embedding.

        Used during enrollment to warn about potential confusion.

        Args:
            embedding: Query embedding

        Returns:
            List of similar speaker info dicts
        """
        search_results = await self.vector_store.search(
            vector=embedding,
            limit=10,
            score_threshold=self.similar_speaker_threshold,
        )

        # Aggregate by speaker
        seen_speakers = set()
        similar = []

        for result in search_results:
            speaker_id = result.payload.get("speaker_id")
            if speaker_id and speaker_id not in seen_speakers:
                seen_speakers.add(speaker_id)
                speaker = await self.speaker_repository.get(speaker_id)
                if speaker:
                    similar.append({
                        "speaker_id": speaker_id,
                        "name": speaker.name,
                        "similarity": round(result.score, 3),
                    })

        return similar
