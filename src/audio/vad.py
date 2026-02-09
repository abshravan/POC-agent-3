"""
Voice Activity Detection (VAD) for speaker identification.

Detects and extracts speech regions from audio.
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)


class VoiceActivityDetector:
    """
    Voice Activity Detection using WebRTC VAD or energy-based detection.

    Identifies speech regions in audio for more accurate speaker
    embedding extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2,
        min_speech_duration: float = 0.5,
        min_silence_duration: float = 0.3,
        use_webrtc: bool = True,
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000 for WebRTC)
            frame_duration_ms: Frame duration (must be 10, 20, or 30ms for WebRTC)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive filtering)
            min_speech_duration: Minimum speech segment duration in seconds
            min_silence_duration: Minimum silence duration to split segments
            use_webrtc: Use WebRTC VAD (falls back to energy-based if unavailable)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # Try to initialize WebRTC VAD
        self.vad = None
        if use_webrtc:
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(aggressiveness)
                logger.info("vad_initialized", type="webrtc", aggressiveness=aggressiveness)
            except ImportError:
                logger.warning("webrtc_vad_unavailable", fallback="energy-based")
            except Exception as e:
                logger.warning("webrtc_vad_error", error=str(e), fallback="energy-based")

    def detect_speech_regions(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Detect speech regions in audio.

        Args:
            audio: Audio array
            sample_rate: Sample rate (uses default if None)

        Returns:
            List of (start_sample, end_sample) tuples for speech regions
        """
        sr = sample_rate or self.sample_rate

        # Get frame-level speech decisions
        if self.vad is not None and sr in (8000, 16000, 32000, 48000):
            speech_frames = self._detect_with_webrtc(audio, sr)
        else:
            speech_frames = self._detect_with_energy(audio, sr)

        # Convert frame decisions to sample regions
        regions = self._frames_to_regions(speech_frames, sr)

        return regions

    def _detect_with_webrtc(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[bool]:
        """
        Detect speech using WebRTC VAD.

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            List of boolean decisions per frame
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32768).astype(np.int16)
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        speech_frames = []
        for i in range(0, len(audio_int16) - frame_size + 1, frame_size):
            frame = audio_int16[i:i + frame_size]
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), sample_rate)
                speech_frames.append(is_speech)
            except Exception:
                speech_frames.append(False)

        return speech_frames

    def _detect_with_energy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        energy_threshold_percentile: int = 30,
    ) -> List[bool]:
        """
        Detect speech using energy-based method.

        Args:
            audio: Audio array
            sample_rate: Sample rate
            energy_threshold_percentile: Percentile for threshold

        Returns:
            List of boolean decisions per frame
        """
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        # Calculate frame energies
        frame_energies = []
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(energy)

        if not frame_energies:
            return []

        # Dynamic threshold based on energy distribution
        energies = np.array(frame_energies)
        threshold = np.percentile(energies, energy_threshold_percentile)

        # Also consider zero-crossing rate for better accuracy
        zcr_threshold = 0.02
        speech_frames = []
        for i, energy in enumerate(frame_energies):
            start = i * frame_size
            end = start + frame_size
            frame = audio[start:end]

            # Zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

            # Combined decision: high energy AND not pure noise (low ZCR)
            is_speech = energy > threshold and zcr < 0.5
            speech_frames.append(is_speech)

        return speech_frames

    def _frames_to_regions(
        self,
        speech_frames: List[bool],
        sample_rate: int,
    ) -> List[Tuple[int, int]]:
        """
        Convert frame-level decisions to sample-level regions.

        Applies minimum duration filtering and merging of close regions.

        Args:
            speech_frames: List of speech/non-speech decisions
            sample_rate: Sample rate

        Returns:
            List of (start_sample, end_sample) tuples
        """
        if not speech_frames:
            return []

        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        min_speech_frames = int(self.min_speech_duration * 1000 / self.frame_duration_ms)
        min_silence_frames = int(self.min_silence_duration * 1000 / self.frame_duration_ms)

        # Apply smoothing - fill short gaps
        smoothed = list(speech_frames)
        for i in range(len(smoothed)):
            if not smoothed[i]:
                # Check if this is a short silence gap
                gap_start = i
                gap_end = i
                while gap_end < len(smoothed) and not smoothed[gap_end]:
                    gap_end += 1
                gap_length = gap_end - gap_start

                if gap_length < min_silence_frames:
                    # Fill the gap
                    for j in range(gap_start, gap_end):
                        smoothed[j] = True

        # Find contiguous speech regions
        regions = []
        in_speech = False
        start_frame = 0

        for i, is_speech in enumerate(smoothed):
            if is_speech and not in_speech:
                # Start of speech
                in_speech = True
                start_frame = i
            elif not is_speech and in_speech:
                # End of speech
                in_speech = False
                end_frame = i

                # Check minimum duration
                if end_frame - start_frame >= min_speech_frames:
                    start_sample = start_frame * frame_size
                    end_sample = end_frame * frame_size
                    regions.append((start_sample, end_sample))

        # Handle speech at the end
        if in_speech:
            end_frame = len(smoothed)
            if end_frame - start_frame >= min_speech_frames:
                start_sample = start_frame * frame_size
                end_sample = end_frame * frame_size
                regions.append((start_sample, end_sample))

        return regions

    def extract_speech(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract speech regions from audio.

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Concatenated speech regions
        """
        regions = self.detect_speech_regions(audio, sample_rate)

        if not regions:
            return np.array([], dtype=np.float32)

        # Concatenate speech regions
        speech_parts = []
        for start, end in regions:
            speech_parts.append(audio[start:end])

        return np.concatenate(speech_parts).astype(np.float32)

    def get_speech_ratio(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> float:
        """
        Get the ratio of speech to total audio duration.

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Speech ratio (0 to 1)
        """
        regions = self.detect_speech_regions(audio, sample_rate)

        if not regions:
            return 0.0

        total_speech_samples = sum(end - start for start, end in regions)
        return total_speech_samples / len(audio)
