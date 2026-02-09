"""Tests for audio quality assessment."""

import numpy as np
import pytest

from src.audio.quality import AudioQualityAssessor


class TestAudioQualityAssessor:
    """Tests for AudioQualityAssessor class."""

    @pytest.fixture
    def assessor(self):
        """Create assessor instance."""
        return AudioQualityAssessor()

    def test_assess_good_audio(self, assessor, sample_audio, sample_rate):
        """Test quality assessment of good audio."""
        result = assessor.assess(sample_audio, sample_rate)

        assert result.is_acceptable
        assert result.duration > 0
        assert result.quality_score > 0

    def test_assess_rejects_quiet_audio(self, assessor, sample_rate):
        """Test that very quiet audio is rejected."""
        quiet_audio = np.random.randn(sample_rate * 3).astype(np.float32) * 0.0001

        result = assessor.assess(quiet_audio, sample_rate)

        assert not result.is_acceptable
        assert "quiet" in " ".join(result.rejection_reasons).lower()

    def test_assess_rejects_short_audio(self, assessor, sample_rate):
        """Test that short audio is rejected."""
        short_audio = np.random.randn(sample_rate // 2).astype(np.float32)  # 0.5s

        result = assessor.assess(short_audio, sample_rate)

        assert not result.is_acceptable
        assert "short" in " ".join(result.rejection_reasons).lower()

    def test_quality_score_in_range(self, assessor, sample_audio, sample_rate):
        """Test that quality score is in valid range."""
        result = assessor.assess(sample_audio, sample_rate)

        assert 0 <= result.quality_score <= 1
