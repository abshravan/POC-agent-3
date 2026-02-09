"""
Speaker embedding extraction using NVIDIA NeMo TitaNet.

Provides high-quality speaker embeddings for identification and verification.
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from config.settings import get_settings
from config.logging_config import get_logger
from src.audio.preprocessing import AudioPreprocessor

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding extraction."""

    embedding: np.ndarray
    duration: float
    model_name: str
    extraction_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary (without embedding array)."""
        return {
            "embedding_dim": len(self.embedding),
            "duration": round(self.duration, 2),
            "model_name": self.model_name,
            "extraction_time_ms": round(self.extraction_time_ms, 2),
        }


class EmbeddingExtractor:
    """
    Extract speaker embeddings using NVIDIA NeMo TitaNet.

    Uses pre-trained TitaNet model for state-of-the-art speaker
    embedding extraction. The model is frozen and not fine-tuned.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        cache_enabled: bool = True,
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_name: NeMo model name (default from settings)
            device: Device to use (cpu, cuda, etc.)
            sample_rate: Expected sample rate (16kHz for NeMo)
            cache_enabled: Whether to cache embeddings by audio hash
        """
        settings = get_settings()

        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.device
        self.sample_rate = sample_rate
        self.cache_enabled = cache_enabled

        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor(target_sample_rate=sample_rate)

        # Load model lazily
        self._model = None
        self._embedding_dim = settings.embedding_dimension

        # Embedding cache
        self._cache: dict[str, np.ndarray] = {}
        self._max_cache_size = 1000

        logger.info(
            "embedding_extractor_initialized",
            model_name=self.model_name,
            device=self.device,
        )

    @property
    def model(self):
        """Lazy load the speaker embedding model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load the NeMo speaker embedding model."""
        import time
        start = time.perf_counter()

        try:
            # Try to import NeMo
            import nemo.collections.asr as nemo_asr

            logger.info("loading_nemo_model", model_name=self.model_name)

            # Load pre-trained model
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                self.model_name
            )

            # Move to device
            model = model.to(self.device)
            model.eval()

            # Freeze model parameters
            for param in model.parameters():
                param.requires_grad = False

            load_time = (time.perf_counter() - start) * 1000
            logger.info(
                "nemo_model_loaded",
                model_name=self.model_name,
                device=self.device,
                load_time_ms=round(load_time, 2),
            )

            return model

        except ImportError:
            logger.warning("nemo_not_available", fallback="speechbrain")
            return self._load_speechbrain_fallback()

        except Exception as e:
            logger.error("model_load_error", error=str(e))
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def _load_speechbrain_fallback(self):
        """Load SpeechBrain ECAPA-TDNN as fallback."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info("loading_speechbrain_model")

            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain-ecapa",
                run_opts={"device": self.device},
            )

            logger.info("speechbrain_model_loaded")
            return model

        except Exception as e:
            logger.error("speechbrain_load_error", error=str(e))
            raise ModelLoadError(f"Failed to load fallback model: {e}") from e

    def extract(
        self,
        audio: Union[np.ndarray, bytes, str, Path],
        sample_rate: Optional[int] = None,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio array, bytes, or file path
            sample_rate: Audio sample rate (uses default if None)
            use_cache: Whether to use embedding cache

        Returns:
            EmbeddingResult with embedding and metadata
        """
        import time
        start = time.perf_counter()

        # Load and preprocess audio
        if isinstance(audio, (str, Path, bytes)):
            audio_array, sr = self.preprocessor.load_and_preprocess(audio)
        else:
            sr = sample_rate or self.sample_rate
            audio_array = self.preprocessor.preprocess(audio, sr)

        duration = len(audio_array) / sr

        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = self._compute_cache_key(audio_array)
            if cache_key in self._cache:
                logger.debug("cache_hit", cache_key=cache_key[:8])
                extraction_time = (time.perf_counter() - start) * 1000
                return EmbeddingResult(
                    embedding=self._cache[cache_key],
                    duration=duration,
                    model_name=self.model_name,
                    extraction_time_ms=extraction_time,
                )

        # Extract embedding
        embedding = self._extract_embedding(audio_array)

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Update cache
        if use_cache and self.cache_enabled:
            self._update_cache(cache_key, embedding)

        extraction_time = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            embedding=embedding,
            duration=duration,
            model_name=self.model_name,
            extraction_time_ms=extraction_time,
        )

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract embedding using the loaded model.

        Args:
            audio: Preprocessed audio array

        Returns:
            Speaker embedding array
        """
        model = self.model

        with torch.no_grad():
            # Check if it's NeMo or SpeechBrain model
            if hasattr(model, 'get_embedding'):
                # NeMo model
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
                audio_length = torch.LongTensor([len(audio)]).to(self.device)

                # Get embedding
                _, embedding = model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_length,
                )
                embedding = embedding.squeeze().cpu().numpy()

            elif hasattr(model, 'encode_batch'):
                # SpeechBrain model
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
                embedding = model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()

            else:
                raise ModelError("Unknown model type - cannot extract embedding")

        return embedding.astype(np.float32)

    def extract_batch(
        self,
        audio_list: List[Union[np.ndarray, bytes, str, Path]],
        sample_rate: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """
        Extract embeddings from multiple audio samples.

        Args:
            audio_list: List of audio arrays, bytes, or file paths
            sample_rate: Audio sample rate

        Returns:
            List of EmbeddingResult
        """
        results = []
        for audio in audio_list:
            result = self.extract(audio, sample_rate)
            results.append(result)
        return results

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Ensure normalized
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)

        return float(np.dot(e1, e2))

    def _compute_cache_key(self, audio: np.ndarray) -> str:
        """Compute cache key from audio content."""
        # Use hash of audio bytes for cache key
        audio_bytes = audio.tobytes()
        return hashlib.md5(audio_bytes).hexdigest()

    def _update_cache(self, key: str, embedding: np.ndarray) -> None:
        """Update embedding cache with size limit."""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = embedding.copy()

    def clear_cache(self) -> int:
        """Clear embedding cache. Returns number of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        return count

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ModelError(Exception):
    """Raised when model inference fails."""
    pass


@lru_cache(maxsize=1)
def get_embedding_extractor() -> EmbeddingExtractor:
    """Get cached embedding extractor instance."""
    return EmbeddingExtractor()
