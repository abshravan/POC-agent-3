"""
Prometheus metrics for speaker identification.

Provides observability for the speaker identification system.
"""

from functools import lru_cache
from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, Info

from config.settings import get_settings


class SpeakerIDMetrics:
    """Speaker identification metrics."""

    def __init__(self):
        """Initialize metrics."""
        # Request counters
        self.identification_requests = Counter(
            "speaker_id_requests_total",
            "Total identification requests",
            ["status"],  # success, unknown, error
        )

        self.enrollment_requests = Counter(
            "speaker_enrollments_total",
            "Total enrollment attempts",
            ["status"],  # success, failed_quality, failed_consistency, failed_duplicate
        )

        # Latency histograms
        self.identification_latency = Histogram(
            "speaker_id_latency_seconds",
            "Identification latency in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.embedding_extraction_latency = Histogram(
            "embedding_extraction_seconds",
            "Embedding extraction latency in seconds",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        )

        self.vector_search_latency = Histogram(
            "vector_search_seconds",
            "Vector search latency in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        )

        # Confidence score distribution
        self.confidence_scores = Histogram(
            "speaker_id_confidence_score",
            "Confidence score distribution",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Audio quality
        self.audio_quality_snr = Histogram(
            "audio_quality_snr_db",
            "Audio quality (SNR in dB)",
            buckets=[5, 10, 15, 20, 25, 30, 40],
        )

        self.audio_duration = Histogram(
            "audio_duration_seconds",
            "Audio duration in seconds",
            buckets=[1, 2, 5, 10, 15, 20, 30],
        )

        # System gauges
        self.enrolled_speakers = Gauge(
            "speaker_enrolled_total",
            "Total number of enrolled speakers",
        )

        self.active_speakers = Gauge(
            "speaker_active_total",
            "Number of active speakers",
        )

        self.embeddings_stored = Gauge(
            "speaker_embeddings_total",
            "Total number of stored embeddings",
        )

        # System info
        self.system_info = Info(
            "speaker_id_system",
            "Speaker identification system information",
        )

    def record_identification(
        self,
        status: str,
        confidence: float,
        latency_seconds: float,
        embedding_time_seconds: float,
        search_time_seconds: float,
        audio_snr_db: float,
        audio_duration_seconds: float,
    ) -> None:
        """
        Record identification request metrics.

        Args:
            status: Request status (success, unknown, error)
            confidence: Confidence score
            latency_seconds: Total latency
            embedding_time_seconds: Embedding extraction time
            search_time_seconds: Vector search time
            audio_snr_db: Audio SNR in dB
            audio_duration_seconds: Audio duration
        """
        self.identification_requests.labels(status=status).inc()
        self.identification_latency.observe(latency_seconds)
        self.confidence_scores.observe(confidence)
        self.embedding_extraction_latency.observe(embedding_time_seconds)
        self.vector_search_latency.observe(search_time_seconds)
        self.audio_quality_snr.observe(audio_snr_db)
        self.audio_duration.observe(audio_duration_seconds)

    def record_enrollment(self, status: str) -> None:
        """
        Record enrollment attempt.

        Args:
            status: Enrollment status
        """
        self.enrollment_requests.labels(status=status).inc()

    def update_speaker_counts(
        self,
        total: int,
        active: int,
        embeddings: int,
    ) -> None:
        """
        Update speaker count gauges.

        Args:
            total: Total enrolled speakers
            active: Active speakers
            embeddings: Total embeddings
        """
        self.enrolled_speakers.set(total)
        self.active_speakers.set(active)
        self.embeddings_stored.set(embeddings)

    def set_system_info(
        self,
        version: str,
        model_name: str,
        device: str,
        embedding_dim: int,
    ) -> None:
        """
        Set system information.

        Args:
            version: Application version
            model_name: Embedding model name
            device: Compute device
            embedding_dim: Embedding dimension
        """
        self.system_info.info({
            "version": version,
            "model_name": model_name,
            "device": device,
            "embedding_dimension": str(embedding_dim),
        })


# Global metrics instance
_metrics: Optional[SpeakerIDMetrics] = None


def setup_metrics() -> SpeakerIDMetrics:
    """Initialize and return metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = SpeakerIDMetrics()

        # Set system info
        settings = get_settings()
        _metrics.set_system_info(
            version=settings.app_version,
            model_name=settings.embedding_model_name,
            device=settings.device,
            embedding_dim=settings.embedding_dimension,
        )

    return _metrics


def get_metrics() -> SpeakerIDMetrics:
    """Get metrics instance."""
    if _metrics is None:
        return setup_metrics()
    return _metrics
