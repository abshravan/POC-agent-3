"""
Pytest fixtures for speaker identification tests.
"""

import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# Configure pytest-asyncio mode
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate a sample audio array for testing."""
    # Generate 3 seconds of audio at 16kHz
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Mix of frequencies to simulate speech-like signal
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Fundamental
        0.2 * np.sin(2 * np.pi * 400 * t) +  # Harmonic
        0.1 * np.sin(2 * np.pi * 800 * t) +  # Harmonic
        0.05 * np.random.randn(len(t))       # Noise
    )

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Generate a sample speaker embedding."""
    # Random 192-dim embedding (normalized)
    embedding = np.random.randn(192).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 16000


@pytest.fixture
def test_speaker_data():
    """Sample speaker data for enrollment tests."""
    return {
        "employee_id": "TEST001",
        "name": "Test Speaker",
        "department": "Testing",
    }
