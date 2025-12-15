"""Speech-to-Text (STT) for Project Gordon.

This module transcribes audio segments to text using OpenAI Whisper.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import whisper

logger = logging.getLogger(__name__)


class SpeechToText:
    """Speech-to-text transcription using OpenAI Whisper.

    Uses English-optimized Whisper model for faster, smaller inference.
    Designed to accept AudioSegment objects from VAD module.

    Example:
        >>> from vad import VoiceActivityDetector
        >>> from stt import SpeechToText
        >>>
        >>> vad = VoiceActivityDetector()
        >>> stt = SpeechToText()
        >>>
        >>> for segment in vad.detect_speech(max_duration_s=10):
        ...     text = stt.transcribe(segment)
        ...     print(f"You said: {text}")
    """

    def __init__(
        self,
        model_name: str = "base.en",
        device: Optional[str] = None,
        language: str = "en",
    ):
        """Initialize Whisper STT.

        Args:
            model_name: Whisper model name. Options:
                - tiny.en (~40MB, fastest, least accurate)
                - base.en (~140MB, balanced - default)
                - small.en (~460MB, better accuracy)
                - medium.en (~1.5GB, high accuracy, slower)
                Note: .en models are English-only and faster than multilingual
            device: Device to use ("cuda", "mps", "cpu"). Auto-detects if None.
            language: Language code (default: "en" for English)
        """
        self.model_name = model_name
        self.language = language

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(
            "Loading Whisper model '%s' on device '%s'...",
            model_name,
            self.device,
        )

        # Load Whisper model
        self.model = whisper.load_model(model_name, device=self.device)

        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio_segment) -> str:
        """Transcribe an audio segment to text.

        Args:
            audio_segment: AudioSegment from VAD module (or compatible object)
                Must have:
                - data: bytes of int16 PCM audio
                - sample_rate: int (should be 16000 for Whisper)

        Returns:
            Transcribed text string. Empty string if transcription fails or
            no speech detected.

        Example:
            >>> segment = vad.detect_speech()
            >>> text = stt.transcribe(segment)
        """
        try:
            # Convert AudioSegment to numpy array
            audio_int16 = np.frombuffer(audio_segment.data, dtype=np.int16)

            # Convert int16 to float32 normalized to [-1, 1]
            # Whisper expects this format
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Whisper expects 16kHz - validate we're getting that
            if audio_segment.sample_rate != 16000:
                logger.warning(
                    "Audio sample rate is %d, Whisper expects 16000. "
                    "Quality may be degraded.",
                    audio_segment.sample_rate,
                )

            # Transcribe using Whisper
            logger.debug("Transcribing audio segment (%d samples)...", len(audio_float32))

            result = self.model.transcribe(
                audio_float32,
                language=self.language,
                fp16=(self.device == "cuda"),  # Use FP16 on CUDA for speed
            )

            text = result["text"].strip()

            logger.info("Transcription: '%s'", text)

            return text

        except Exception as e:
            logger.error("Transcription failed: %s", e, exc_info=True)
            return ""

    def transcribe_file(self, audio_file_path: str) -> str:
        """Transcribe an audio file to text.

        Useful for testing with WAV files or other audio formats.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text string

        Example:
            >>> text = stt.transcribe_file("recording.wav")
        """
        try:
            logger.info("Transcribing file: %s", audio_file_path)

            result = self.model.transcribe(
                audio_file_path,
                language=self.language,
                fp16=(self.device == "cuda"),
            )

            text = result["text"].strip()

            logger.info("Transcription: '%s'", text)

            return text

        except Exception as e:
            logger.error("File transcription failed: %s", e, exc_info=True)
            return ""

    def info(self) -> dict:
        """Get information about the STT configuration.

        Returns:
            Dictionary with model info
        """
        return {
            "model": self.model_name,
            "device": self.device,
            "language": self.language,
        }
