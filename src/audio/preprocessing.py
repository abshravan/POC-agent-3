"""
Audio preprocessing pipeline for speaker identification.

Handles format conversion, resampling, normalization, and enhancement.
"""

import io
import struct
import subprocess
import tempfile
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

        Supports WAV, FLAC, OGG, MP3, WebM, and other formats via ffmpeg fallback.

        Args:
            audio_input: Audio file path, bytes, or file-like object
            target_sr: Target sample rate (uses default if None)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        target_sr = target_sr or self.target_sample_rate
        soundfile_error = None
        pydub_error = None

        # Try soundfile first (fastest for supported formats)
        try:
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
            soundfile_error = str(e)
            logger.debug("soundfile_failed", error=soundfile_error)

        # Try pydub as fallback (handles more formats via ffmpeg)
        try:
            audio, sr = self._load_with_pydub(audio_input, target_sr)
            return audio, sr
        except Exception as e:
            pydub_error = str(e)
            logger.debug("pydub_failed", error=pydub_error)

        # Try direct ffmpeg as last resort
        try:
            audio, sr = self._load_with_ffmpeg(audio_input, target_sr)
            return audio, sr
        except Exception as e:
            ffmpeg_error = str(e)
            logger.debug("ffmpeg_failed", error=ffmpeg_error)

        # Last resort: try to interpret as raw PCM
        try:
            audio, sr = self._load_as_raw_pcm(audio_input, target_sr)
            return audio, sr
        except Exception as e:
            logger.error("all_audio_loaders_failed",
                        soundfile_error=soundfile_error,
                        pydub_error=pydub_error,
                        ffmpeg_error=ffmpeg_error,
                        raw_pcm_error=str(e))
            raise AudioLoadError(
                f"Unsupported audio format. "
                f"soundfile error: {soundfile_error}, "
                f"pydub error: {pydub_error}"
            ) from e

    def _load_as_raw_pcm(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO],
        target_sr: int,
    ) -> Tuple[np.ndarray, int]:
        """Try to load as raw PCM data (last resort for browser audio)."""
        if isinstance(audio_input, bytes):
            data = audio_input
        elif isinstance(audio_input, (str, Path)):
            with open(str(audio_input), "rb") as f:
                data = f.read()
        else:
            data = audio_input.read()

        # Try float32 first (Web Audio API format)
        if len(data) % 4 == 0:
            try:
                audio = np.frombuffer(data, dtype=np.float32)
                if np.all(np.abs(audio) <= 1.5):  # Valid float audio range
                    logger.info("loaded_as_raw_float32", samples=len(audio))
                    return audio, target_sr
            except Exception:
                pass

        # Try int16 (common PCM format)
        if len(data) % 2 == 0:
            try:
                audio = np.frombuffer(data, dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                logger.info("loaded_as_raw_int16", samples=len(audio))
                return audio, target_sr
            except Exception:
                pass

        raise AudioLoadError("Could not interpret as raw PCM")

    def _load_with_pydub(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO],
        target_sr: int,
    ) -> Tuple[np.ndarray, int]:
        """Load audio using pydub (requires ffmpeg)."""
        import os
        from pydub import AudioSegment

        if isinstance(audio_input, (str, Path)):
            audio_segment = AudioSegment.from_file(str(audio_input))
        elif isinstance(audio_input, bytes):
            # Write to temp file so ffmpeg can detect format from extension
            # Try multiple formats
            audio_segment = None
            last_error = None
            for ext in [".webm", ".ogg", ".wav", ".mp3"]:
                try:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(audio_input)
                        tmp.flush()
                        tmp_path = tmp.name
                    try:
                        audio_segment = AudioSegment.from_file(tmp_path)
                        break
                    finally:
                        os.unlink(tmp_path)
                except Exception as e:
                    last_error = e
                    continue
            if audio_segment is None:
                raise last_error or AudioLoadError("Failed to load audio with pydub")
        else:
            # Read from file-like object
            data = audio_input.read()
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(data)
                tmp.flush()
                tmp_path = tmp.name
            try:
                audio_segment = AudioSegment.from_file(tmp_path)
            finally:
                os.unlink(tmp_path)

        # Convert to mono
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        # Resample if needed
        if audio_segment.frame_rate != target_sr:
            audio_segment = audio_segment.set_frame_rate(target_sr)

        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())

        # Normalize to float32 [-1, 1]
        if audio_segment.sample_width == 1:
            samples = samples.astype(np.float32) / 128.0 - 1.0
        elif audio_segment.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            samples = samples.astype(np.float32)

        return samples, target_sr

    def _load_with_ffmpeg(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO],
        target_sr: int,
    ) -> Tuple[np.ndarray, int]:
        """Load audio directly using ffmpeg subprocess."""
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=True) as tmp_out:
            if isinstance(audio_input, bytes):
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
                    tmp_in.write(audio_input)
                    tmp_in.flush()
                    input_path = tmp_in.name
            elif isinstance(audio_input, (str, Path)):
                input_path = str(audio_input)
            else:
                # Read from file-like object
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
                    tmp_in.write(audio_input.read())
                    tmp_in.flush()
                    input_path = tmp_in.name

            # Run ffmpeg to convert to raw PCM
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-f", "f32le",  # 32-bit float little-endian
                "-acodec", "pcm_f32le",
                "-ac", "1",  # mono
                "-ar", str(target_sr),  # target sample rate
                tmp_out.name
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise AudioLoadError(f"ffmpeg failed: {result.stderr.decode()}")

            # Read raw PCM data
            with open(tmp_out.name, "rb") as f:
                raw_data = f.read()

            # Convert to numpy array (float32)
            audio = np.frombuffer(raw_data, dtype=np.float32)

            return audio, target_sr

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
