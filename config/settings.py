"""
Application settings management using Pydantic.

All settings can be overridden via environment variables.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Speaker Identification System"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    api_workers: int = 1

    # Audio Processing
    target_sample_rate: int = 16000  # NeMo expects 16kHz
    min_audio_duration: float = 1.0  # Minimum audio length in seconds
    max_audio_duration: float = 30.0  # Maximum audio length in seconds
    audio_chunk_duration: float = 10.0  # Chunk size for long audio
    audio_chunk_overlap: float = 0.5  # Overlap ratio between chunks

    # Voice Activity Detection
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)  # WebRTC VAD 0-3
    vad_frame_duration_ms: int = Field(default=30, ge=10, le=30)

    # Audio Quality Thresholds
    min_snr_db: float = 10.0  # Minimum Signal-to-Noise Ratio
    min_rms_energy: float = 0.005  # Minimum RMS energy

    # Embedding Model
    embedding_model_name: str = "nvidia/speakerverification_en_titanet_large"
    embedding_dimension: int = 192
    embedding_cache_enabled: bool = True
    embedding_cache_ttl: int = 3600  # seconds

    # Device Settings
    device: str = "cpu"  # cpu, cuda, cuda:0, etc.
    use_gpu: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string."""
        if v.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        return v

    # Vector Database (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "speaker_embeddings"
    qdrant_grpc_port: int = 6334
    qdrant_use_grpc: bool = False
    qdrant_path: Optional[str] = "./data/qdrant"  # Local path for embedded mode

    # Metadata Database
    database_url: str = "sqlite+aiosqlite:///./data/sqlite/speakers.db"

    # Identification Thresholds
    identification_threshold: float = 0.55
    enrollment_verification_threshold: float = 0.70
    high_confidence_threshold: float = 0.75
    similarity_search_top_k: int = 10

    # Enrollment Settings
    min_enrollment_samples: int = 3
    max_enrollment_samples: int = 10
    enrollment_consistency_threshold: float = 0.80
    similar_speaker_warning_threshold: float = 0.70

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # Security
    secret_key: str = Field(default="change-me-in-production-please")
    encrypt_embeddings: bool = False
    encryption_key: Optional[str] = None

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Redis (optional, for caching/rate limiting)
    redis_url: Optional[str] = None

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
