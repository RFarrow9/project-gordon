"""Tests for VAD core functionality."""
from __future__ import annotations

import numpy as np
import pytest

from vad.core import (
    SAMPLE_RATE,
    AudioSegment,
    SimpleVAD,
    VoiceActivityDetector,
    WEBRTCVAD_AVAILABLE,
)


class TestAudioSegment:
    """Tests for AudioSegment class."""

    def test_create_segment(self, sample_audio_segment):
        """Test creating an audio segment."""
        segment = sample_audio_segment

        assert segment.data is not None
        assert segment.sample_rate == SAMPLE_RATE
        assert segment.duration_ms > 0

    def test_to_numpy(self, sample_audio_segment):
        """Test converting audio segment to numpy array."""
        segment = sample_audio_segment
        audio_array = segment.to_numpy()

        assert isinstance(audio_array, np.ndarray)
        assert audio_array.dtype == np.int16
        assert len(audio_array) == len(segment.data) // 2  # 2 bytes per sample

    def test_save_wav(self, sample_audio_segment, tmp_path):
        """Test saving audio segment to WAV file."""
        segment = sample_audio_segment
        wav_file = tmp_path / "test.wav"

        segment.save_wav(str(wav_file))

        assert wav_file.exists()
        assert wav_file.stat().st_size > 0

    def test_silent_audio(self, silent_audio_segment):
        """Test silent audio segment."""
        segment = silent_audio_segment
        audio_array = segment.to_numpy()

        # Check that audio is mostly silent
        max_amplitude = np.max(np.abs(audio_array))
        assert max_amplitude < 100  # Very quiet


@pytest.mark.skipif(not WEBRTCVAD_AVAILABLE, reason="webrtcvad not available")
class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    def test_initialization(self):
        """Test VAD initialization."""
        vad = VoiceActivityDetector(aggressiveness=2)

        assert vad.aggressiveness == 2
        assert vad.sample_rate == SAMPLE_RATE
        assert vad.vad is not None

    def test_invalid_sample_rate(self):
        """Test that invalid sample rate raises error."""
        with pytest.raises(ValueError, match="Sample rate must be"):
            VoiceActivityDetector(sample_rate=22050)  # Invalid rate

    def test_invalid_frame_duration(self):
        """Test that invalid frame duration raises error."""
        with pytest.raises(ValueError, match="Frame duration must be"):
            VoiceActivityDetector(frame_duration_ms=25)  # Invalid duration

    def test_list_devices(self):
        """Test listing audio devices."""
        vad = VoiceActivityDetector()
        devices = vad.list_devices()

        assert isinstance(devices, list)
        # May be empty in CI environment, but should be a list
        if devices:
            assert "index" in devices[0]
            assert "name" in devices[0]

    @pytest.mark.audio
    def test_microphone_test(self):
        """Test microphone testing functionality."""
        vad = VoiceActivityDetector()
        # This test requires actual audio hardware
        result = vad.test_microphone(duration_s=1.0)
        assert isinstance(result, bool)

    @pytest.mark.audio
    @pytest.mark.slow
    def test_detect_speech(self):
        """Test speech detection (requires microphone)."""
        vad = VoiceActivityDetector(aggressiveness=3)

        # This would require actual speech input
        # Just test that the generator is created
        generator = vad.detect_speech(max_duration_s=1.0)
        assert generator is not None

    def test_aggressiveness_levels(self):
        """Test different aggressiveness levels."""
        for level in [0, 1, 2, 3]:
            vad = VoiceActivityDetector(aggressiveness=level)
            assert vad.aggressiveness == level


class TestSimpleVAD:
    """Tests for SimpleVAD class."""

    def test_initialization(self):
        """Test SimpleVAD initialization."""
        vad = SimpleVAD()

        assert vad.sample_rate == SAMPLE_RATE
        assert vad.energy_threshold > 0

    def test_list_devices(self):
        """Test listing audio devices."""
        vad = SimpleVAD()
        devices = vad.list_devices()

        assert isinstance(devices, list)
        if devices:
            assert "index" in devices[0]
            assert "name" in devices[0]

    def test_energy_calculation(self):
        """Test energy calculation for audio frames."""
        vad = SimpleVAD()

        # Create loud frame
        loud_frame = np.random.randn(1000) * 0.5
        loud_energy = vad._calculate_energy(loud_frame)

        # Create quiet frame
        quiet_frame = np.random.randn(1000) * 0.001
        quiet_energy = vad._calculate_energy(quiet_frame)

        assert loud_energy > quiet_energy
        assert loud_energy > vad.energy_threshold
        assert quiet_energy < vad.energy_threshold

    @pytest.mark.audio
    @pytest.mark.slow
    def test_detect_speech(self):
        """Test simple VAD speech detection (requires microphone)."""
        vad = SimpleVAD()

        # Test that the generator is created
        generator = vad.detect_speech(max_duration_s=1.0)
        assert generator is not None

    @pytest.mark.audio
    def test_microphone_test(self):
        """Test microphone testing functionality."""
        vad = SimpleVAD()
        # This test requires actual audio hardware
        result = vad.test_microphone(duration_s=1.0)
        assert isinstance(result, bool)


class TestIntegration:
    """Integration tests for VAD module."""

    def test_module_imports(self):
        """Test that all expected symbols are importable."""
        from vad import AudioSegment, SimpleVAD, VoiceActivityDetector, SAMPLE_RATE

        assert AudioSegment is not None
        assert SimpleVAD is not None
        assert VoiceActivityDetector is not None
        assert SAMPLE_RATE == 16000

    def test_sample_rate_constant(self):
        """Test that SAMPLE_RATE is set correctly."""
        assert SAMPLE_RATE in [8000, 16000, 32000, 48000]

    @pytest.mark.skipif(not WEBRTCVAD_AVAILABLE, reason="webrtcvad not available")
    def test_webrtc_vad_works(self):
        """Test that WebRTC VAD can process audio."""
        import webrtcvad

        vad = webrtcvad.Vad(2)

        # Create a test frame (30ms at 16kHz = 480 samples)
        frame_size = int(16000 * 30 / 1000)
        frame = np.random.randn(frame_size) * 0.3
        audio_int16 = (frame * 32767).astype(np.int16)
        frame_bytes = audio_int16.tobytes()

        # Should not raise
        is_speech = vad.is_speech(frame_bytes, 16000)
        assert isinstance(is_speech, bool)
