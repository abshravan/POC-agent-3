"""
Audio quality assessment for speaker identification.

Evaluates audio segments to ensure they meet quality thresholds
for reliable speaker embedding extraction.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QualityResult:
    """Audio quality assessment result."""

    is_acceptable: bool
    rms_energy: float
    snr_db: float
    duration: float
    clipping_ratio: float
    silence_ratio: float
    quality_score: float  # 0-1 overall quality score
    rejection_reasons: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_acceptable": self.is_acceptable,
            "rms_energy": round(self.rms_energy, 4),
            "snr_db": round(self.snr_db, 2),
            "duration": round(self.duration, 2),
            "clipping_ratio": round(self.clipping_ratio, 4),
            "silence_ratio": round(self.silence_ratio, 4),
            "quality_score": round(self.quality_score, 3),
            "rejection_reasons": self.rejection_reasons,
        }


class AudioQualityAssessor:
    """
    Assess audio quality for speaker identification.

    Checks various quality metrics and determines if audio
    is suitable for reliable embedding extraction.
    """

    def __init__(
        self,
        min_snr_db: float = 10.0,
        min_rms_energy: float = 0.005,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        max_clipping_ratio: float = 0.01,
        max_silence_ratio: float = 0.8,
        silence_threshold: float = 0.01,
    ):
        """
        Initialize the quality assessor.

        Args:
            min_snr_db: Minimum signal-to-noise ratio in dB
            min_rms_energy: Minimum RMS energy level
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            max_clipping_ratio: Maximum allowed clipping ratio
            max_silence_ratio: Maximum allowed silence ratio
            silence_threshold: Threshold for detecting silence
        """
        self.min_snr_db = min_snr_db
        self.min_rms_energy = min_rms_energy
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_clipping_ratio = max_clipping_ratio
        self.max_silence_ratio = max_silence_ratio
        self.silence_threshold = silence_threshold

    def assess(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> QualityResult:
        """
        Assess audio quality.

        Args:
            audio: Audio array
            sample_rate: Sample rate in Hz

        Returns:
            QualityResult with all metrics and acceptance decision
        """
        rejection_reasons = []

        # Calculate duration
        duration = len(audio) / sample_rate

        # Check duration bounds
        if duration < self.min_duration:
            rejection_reasons.append(
                f"Duration too short: {duration:.2f}s < {self.min_duration}s"
            )
        if duration > self.max_duration:
            rejection_reasons.append(
                f"Duration too long: {duration:.2f}s > {self.max_duration}s"
            )

        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < self.min_rms_energy:
            rejection_reasons.append(
                f"Audio too quiet: RMS={rms_energy:.4f} < {self.min_rms_energy}"
            )

        # Estimate SNR
        snr_db = self._estimate_snr(audio, sample_rate)
        if snr_db < self.min_snr_db:
            rejection_reasons.append(
                f"SNR too low: {snr_db:.1f}dB < {self.min_snr_db}dB"
            )

        # Check for clipping
        clipping_ratio = self._calculate_clipping_ratio(audio)
        if clipping_ratio > self.max_clipping_ratio:
            rejection_reasons.append(
                f"Too much clipping: {clipping_ratio:.2%} > {self.max_clipping_ratio:.2%}"
            )

        # Check for silence ratio
        silence_ratio = self._calculate_silence_ratio(audio)
        if silence_ratio > self.max_silence_ratio:
            rejection_reasons.append(
                f"Too much silence: {silence_ratio:.2%} > {self.max_silence_ratio:.2%}"
            )

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            rms_energy, snr_db, clipping_ratio, silence_ratio, duration
        )

        is_acceptable = len(rejection_reasons) == 0

        return QualityResult(
            is_acceptable=is_acceptable,
            rms_energy=rms_energy,
            snr_db=snr_db,
            duration=duration,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
            quality_score=quality_score,
            rejection_reasons=rejection_reasons,
        )

    def _estimate_snr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_size: int = 512,
    ) -> float:
        """
        Estimate Signal-to-Noise Ratio using percentile method.

        Compares high-energy frames (signal) to low-energy frames (noise).

        Args:
            audio: Audio array
            sample_rate: Sample rate
            frame_size: Frame size for energy calculation

        Returns:
            Estimated SNR in dB
        """
        num_frames = len(audio) // frame_size
        if num_frames < 2:
            return 0.0

        # Calculate frame energies
        frame_energies = np.array([
            np.sqrt(np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2))
            for i in range(num_frames)
        ])

        # Signal is 90th percentile, noise is 10th percentile
        signal_energy = np.percentile(frame_energies, 90)
        noise_energy = np.percentile(frame_energies, 10) + 1e-10

        # Calculate SNR in dB
        snr_db = 20 * np.log10(signal_energy / noise_energy)

        return float(snr_db)

    def _calculate_clipping_ratio(
        self,
        audio: np.ndarray,
        clip_threshold: float = 0.99,
    ) -> float:
        """
        Calculate the ratio of clipped samples.

        Args:
            audio: Audio array (assumed normalized to [-1, 1])
            clip_threshold: Threshold for detecting clipping

        Returns:
            Ratio of clipped samples (0 to 1)
        """
        clipped_samples = np.sum(np.abs(audio) > clip_threshold)
        return float(clipped_samples / len(audio))

    def _calculate_silence_ratio(
        self,
        audio: np.ndarray,
        frame_size: int = 512,
    ) -> float:
        """
        Calculate the ratio of silent frames.

        Args:
            audio: Audio array
            frame_size: Frame size for analysis

        Returns:
            Ratio of silent frames (0 to 1)
        """
        num_frames = len(audio) // frame_size
        if num_frames == 0:
            return 1.0

        silent_frames = 0
        for i in range(num_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < self.silence_threshold:
                silent_frames += 1

        return float(silent_frames / num_frames)

    def _calculate_quality_score(
        self,
        rms_energy: float,
        snr_db: float,
        clipping_ratio: float,
        silence_ratio: float,
        duration: float,
    ) -> float:
        """
        Calculate overall quality score (0 to 1).

        Higher score = better quality audio.

        Args:
            rms_energy: RMS energy level
            snr_db: Signal-to-noise ratio in dB
            clipping_ratio: Ratio of clipped samples
            silence_ratio: Ratio of silent frames
            duration: Audio duration in seconds

        Returns:
            Quality score from 0 to 1
        """
        scores = []

        # RMS energy score (0.005 to 0.5 mapped to 0-1)
        rms_score = min(1.0, max(0.0, (rms_energy - 0.005) / 0.495))
        scores.append(rms_score * 0.2)

        # SNR score (10 to 40 dB mapped to 0-1)
        snr_score = min(1.0, max(0.0, (snr_db - 10) / 30))
        scores.append(snr_score * 0.3)

        # Clipping score (inverted, 0% to 1% mapped to 1-0)
        clip_score = 1.0 - min(1.0, clipping_ratio / 0.01)
        scores.append(clip_score * 0.2)

        # Silence score (inverted, 0% to 80% mapped to 1-0)
        silence_score = 1.0 - min(1.0, silence_ratio / 0.8)
        scores.append(silence_score * 0.2)

        # Duration score (1s to 10s is optimal)
        if duration < 1:
            dur_score = duration
        elif duration <= 10:
            dur_score = 1.0
        else:
            dur_score = max(0.5, 1.0 - (duration - 10) / 50)
        scores.append(dur_score * 0.1)

        return sum(scores)
