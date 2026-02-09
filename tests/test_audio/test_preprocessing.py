"""Tests for audio preprocessing module."""

import numpy as np
import pytest

from src.audio.preprocessing import AudioPreprocessor


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return AudioPreprocessor()

    def test_preprocess_removes_dc_offset(self, preprocessor, sample_audio):
        """Test that preprocessing removes DC offset."""
        # Add DC offset
        audio_with_dc = sample_audio + 0.5

        processed = preprocessor.preprocess(audio_with_dc, 16000)

        # Mean should be close to 0
        assert abs(np.mean(processed)) < 0.01

    def test_preprocess_normalizes_amplitude(self, preprocessor, sample_audio):
        """Test that preprocessing normalizes amplitude."""
        # Quiet audio
        quiet_audio = sample_audio * 0.1

        processed = preprocessor.preprocess(quiet_audio, 16000)

        # Should be normalized
        assert np.max(np.abs(processed)) > 0.5

    def test_chunk_audio_creates_overlapping_chunks(self, preprocessor, sample_audio, sample_rate):
        """Test audio chunking with overlap."""
        # 3 second audio, 1 second chunks, 50% overlap
        chunks = preprocessor.chunk_audio(
            sample_audio,
            sample_rate,
            chunk_duration=1.0,
            overlap=0.5,
        )

        # Should have multiple chunks
        assert len(chunks) >= 2

        # Each chunk should be approximately 1 second
        for chunk in chunks:
            assert len(chunk) == sample_rate

    def test_get_duration(self, preprocessor, sample_audio, sample_rate):
        """Test duration calculation."""
        duration = preprocessor.get_duration(sample_audio, sample_rate)

        assert duration == pytest.approx(3.0, rel=0.01)
