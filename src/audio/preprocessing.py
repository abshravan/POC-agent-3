"""
Audio preprocessing pipeline for speaker identification.

Handles format conversion, resampling, normalization, and enhancement.
"""

import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for speaker identification.

    Converts various audio formats to normalized 16kHz mono PCM suitable
    for NVIDIA NeMo speaker embedding extraction.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        pre_emphasis_coef: float = 0.97,
        normalize_audio: bool = True,
        remove_dc_offset: bool = True,
    ):
        """
        Initialize the audio preprocessor.

        Args:
            target_sample_rate: Target sample rate (16kHz for NeMo)
            pre_emphasis_coef: Pre-emphasis filter coefficient
            normalize_audio: Whether to normalize audio amplitude
            remove_dc_offset: Whether to remove DC offset
        """
        self.target_sample_rate = target_sample_rate
        self.pre_emphasis_coef = pre_emphasis_coef
        self.normalize_audio = normalize_audio
        self.remove_dc_offset = remove_dc_offset

    def load_audio(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO],
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from various sources and convert to numpy array.

        Args:
            audio_input: Audio file path, bytes, or file-like object
            target_sr: Target sample rate (uses default if None)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        target_sr = target_sr or self.target_sample_rate

        try:
            # Handle different input types
            if isinstance(audio_input, (str, Path)):
                audio, sr = sf.read(str(audio_input), dtype="float32")
            elif isinstance(audio_input, bytes):
                audio, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
            else:
                audio, sr = sf.read(audio_input, dtype="float32")

            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if necessary
            if sr != target_sr:
                audio = self._resample(audio, sr, target_sr)
                sr = target_sr

            return audio.astype(np.float32), sr

        except Exception as e:
            logger.error("audio_load_error", error=str(e))
            raise AudioLoadError(f"Failed to load audio: {e}") from e

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        Resample audio using high-quality sinc interpolation.

        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio

        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        n_samples = int(len(audio) * ratio)

        # Use scipy's resample for high-quality interpolation
        resampled = signal.resample(audio, n_samples)

        # Apply anti-aliasing filter for downsampling
        if ratio < 1:
            nyquist = target_sr / 2
            cutoff = nyquist * 0.9
            b, a = signal.butter(8, cutoff / (orig_sr / 2), btype="low")
            resampled = signal.filtfilt(b, a, resampled)

        return resampled.astype(np.float32)

    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        apply_pre_emphasis: bool = True,
    ) -> np.ndarray:
        """
        Apply preprocessing pipeline to audio.

        Pipeline:
        1. Remove DC offset
        2. Apply pre-emphasis filter
        3. Normalize amplitude

        Args:
            audio: Input audio array
            sample_rate: Audio sample rate
            apply_pre_emphasis: Whether to apply pre-emphasis

        Returns:
            Preprocessed audio array
        """
        processed = audio.copy()

        # Step 1: Remove DC offset
        if self.remove_dc_offset:
            processed = processed - np.mean(processed)

        # Step 2: Apply pre-emphasis filter
        if apply_pre_emphasis and self.pre_emphasis_coef > 0:
            processed = np.append(
                processed[0],
                processed[1:] - self.pre_emphasis_coef * processed[:-1]
            )

        # Step 3: Normalize amplitude
        if self.normalize_audio:
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                # Normalize to -3dB
                target_level = 0.708  # -3dB
                processed = processed * (target_level / max_val)

        return processed.astype(np.float32)

    def load_and_preprocess(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO],
    ) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio in one step.

        Args:
            audio_input: Audio file path, bytes, or file-like object

        Returns:
            Tuple of (preprocessed_audio, sample_rate)
        """
        audio, sr = self.load_audio(audio_input)
        processed = self.preprocess(audio, sr)
        return processed, sr

    def chunk_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_duration: float = 10.0,
        overlap: float = 0.5,
        min_chunk_duration: float = 1.0,
    ) -> list[np.ndarray]:
        """
        Split long audio into overlapping chunks.

        Args:
            audio: Input audio array
            sample_rate: Audio sample rate
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap ratio between chunks (0.0 to 1.0)
            min_chunk_duration: Minimum chunk duration in seconds

        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sample_rate)
        hop_samples = int(chunk_samples * (1 - overlap))
        min_samples = int(min_chunk_duration * sample_rate)

        # If audio is shorter than minimum, return as single chunk
        if len(audio) < min_samples:
            return []

        # If audio is shorter than chunk size, return as single chunk
        if len(audio) <= chunk_samples:
            return [audio]

        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Only include chunk if it meets minimum duration
            if len(chunk) >= min_samples:
                chunks.append(chunk)

            start += hop_samples

        return chunks

    def apply_noise_reduction(
        self,
        audio: np.ndarray,
        sample_rate: int,
        noise_reduction_factor: float = 0.75,
    ) -> np.ndarray:
        """
        Apply simple spectral gating noise reduction.

        Args:
            audio: Input audio array
            sample_rate: Audio sample rate
            noise_reduction_factor: Amount of noise reduction (0.0 to 1.0)

        Returns:
            Noise-reduced audio
        """
        # Use STFT for spectral analysis
        nperseg = 2048
        noverlap = nperseg // 2

        f, t, Zxx = signal.stft(
            audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Estimate noise from quietest 10% of frames
        magnitudes = np.abs(Zxx)
        frame_energies = np.mean(magnitudes, axis=0)
        noise_frames = np.argsort(frame_energies)[:int(len(frame_energies) * 0.1)]
        noise_estimate = np.mean(magnitudes[:, noise_frames], axis=1, keepdims=True)

        # Apply spectral gating
        mask = magnitudes > (noise_estimate * (1 + noise_reduction_factor))
        Zxx_filtered = Zxx * mask

        # Inverse STFT
        _, audio_filtered = signal.istft(
            Zxx_filtered, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
        )

        return audio_filtered.astype(np.float32)

    def get_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        """Get audio duration in seconds."""
        return len(audio) / sample_rate


class AudioLoadError(Exception):
    """Raised when audio loading fails."""
    pass
