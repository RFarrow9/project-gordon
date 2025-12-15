"""Pytest configuration and shared fixtures for VAD tests."""
from __future__ import annotations

import numpy as np
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "audio: tests that require audio hardware (microphone)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take a long time to run",
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--with-audio",
        action="store_true",
        default=False,
        help="Run tests that require audio hardware",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--with-audio"):
        skip_audio = pytest.mark.skip(reason="need --with-audio option to run")
        for item in items:
            if "audio" in item.keywords:
                item.add_marker(skip_audio)


@pytest.fixture
def sample_audio_segment():
    """Generate a sample audio segment for testing."""
    from vad.core import AudioSegment, SAMPLE_RATE

    # Generate 1 second of sine wave audio (440 Hz tone)
    duration_s = 1.0
    sample_rate = SAMPLE_RATE
    num_samples = int(sample_rate * duration_s)

    # Create sine wave
    frequency = 440.0  # A4 note
    t = np.linspace(0, duration_s, num_samples, False)
    audio_array = np.sin(2 * np.pi * frequency * t) * 0.5

    # Convert to int16
    audio_int16 = (audio_array * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    return AudioSegment(
        data=audio_bytes,
        sample_rate=sample_rate,
        duration_ms=int(duration_s * 1000),
    )


@pytest.fixture
def silent_audio_segment():
    """Generate a silent audio segment for testing."""
    from vad.core import AudioSegment, SAMPLE_RATE

    duration_s = 1.0
    sample_rate = SAMPLE_RATE
    num_samples = int(sample_rate * duration_s)

    # Silent audio (all zeros)
    audio_int16 = np.zeros(num_samples, dtype=np.int16)
    audio_bytes = audio_int16.tobytes()

    return AudioSegment(
        data=audio_bytes,
        sample_rate=sample_rate,
        duration_ms=int(duration_s * 1000),
    )


@pytest.fixture
def mock_audio_frames():
    """Generate mock audio frames for VAD testing."""
    from vad.core import SAMPLE_RATE, FRAME_DURATION_MS

    frame_size = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    num_frames = 100

    frames = []
    for i in range(num_frames):
        # Alternate between speech-like (noisy) and silent frames
        if i % 4 < 2:  # Speech-like frames
            frame = np.random.randn(frame_size) * 0.3
        else:  # Silent frames
            frame = np.random.randn(frame_size) * 0.01

        audio_int16 = (frame * 32767).astype(np.int16)
        frames.append(audio_int16.tobytes())

    return frames
