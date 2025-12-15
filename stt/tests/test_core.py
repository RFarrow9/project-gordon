"""Unit tests for STT core module."""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestSpeechToText:
    """Tests for SpeechToText class."""

    @pytest.fixture
    def mock_whisper(self):
        """Mock whisper module to avoid loading actual model in tests."""
        with patch("stt.core.whisper") as mock:
            mock_model = Mock()
            mock.load_model.return_value = mock_model
            yield mock, mock_model

    def test_init_default(self, mock_whisper):
        """Test initialization with default parameters."""
        from stt.core import SpeechToText

        _, mock_model = mock_whisper
        stt = SpeechToText()

        assert stt.model_name == "base.en"
        assert stt.language == "en"
        assert stt.device in ["cuda", "mps", "cpu"]

    def test_init_custom_model(self, mock_whisper):
        """Test initialization with custom model."""
        from stt.core import SpeechToText

        stt = SpeechToText(model_name="tiny.en", device="cpu")

        assert stt.model_name == "tiny.en"
        assert stt.device == "cpu"

    def test_transcribe_success(self, mock_whisper):
        """Test successful transcription."""
        from stt.core import SpeechToText

        _, mock_model = mock_whisper
        mock_model.transcribe.return_value = {"text": "  Hello world  "}

        stt = SpeechToText()

        # Create mock AudioSegment
        mock_segment = Mock()
        mock_segment.data = np.array([1000, 2000, -1000], dtype=np.int16).tobytes()
        mock_segment.sample_rate = 16000

        text = stt.transcribe(mock_segment)

        assert text == "Hello world"  # Should be stripped
        assert mock_model.transcribe.called

    def test_transcribe_empty_result(self, mock_whisper):
        """Test transcription with empty result."""
        from stt.core import SpeechToText

        _, mock_model = mock_whisper
        mock_model.transcribe.return_value = {"text": "   "}

        stt = SpeechToText()

        mock_segment = Mock()
        mock_segment.data = np.array([0, 0, 0], dtype=np.int16).tobytes()
        mock_segment.sample_rate = 16000

        text = stt.transcribe(mock_segment)

        assert text == ""

    def test_transcribe_exception(self, mock_whisper):
        """Test transcription handles exceptions gracefully."""
        from stt.core import SpeechToText

        _, mock_model = mock_whisper
        mock_model.transcribe.side_effect = Exception("Transcription failed")

        stt = SpeechToText()

        mock_segment = Mock()
        mock_segment.data = b"invalid"
        mock_segment.sample_rate = 16000

        text = stt.transcribe(mock_segment)

        assert text == ""  # Should return empty string on error

    def test_transcribe_wrong_sample_rate(self, mock_whisper):
        """Test warning when sample rate is not 16000."""
        from stt.core import SpeechToText

        _, mock_model = mock_whisper
        mock_model.transcribe.return_value = {"text": "test"}

        stt = SpeechToText()

        mock_segment = Mock()
        mock_segment.data = np.array([1000], dtype=np.int16).tobytes()
        mock_segment.sample_rate = 8000  # Wrong rate

        # Should still work but log warning
        text = stt.transcribe(mock_segment)
        assert text == "test"

    def test_transcribe_file(self, mock_whisper):
        """Test file transcription."""
        from stt.core import SpeechToText

        _, mock_model = mock_whisper
        mock_model.transcribe.return_value = {"text": "File transcription"}

        stt = SpeechToText()

        text = stt.transcribe_file("test.wav")

        assert text == "File transcription"
        mock_model.transcribe.assert_called_once()

    def test_info(self, mock_whisper):
        """Test info method."""
        from stt.core import SpeechToText

        stt = SpeechToText(model_name="small.en", device="cuda", language="en")

        info = stt.info()

        assert info["model"] == "small.en"
        assert info["device"] == "cuda"
        assert info["language"] == "en"
