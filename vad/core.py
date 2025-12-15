"""Voice Activity Detection (VAD) for Project Gordon.

This module provides OS-agnostic voice activity detection using WebRTC VAD.
It captures audio from the microphone and detects speech segments.
"""
from __future__ import annotations

import logging
import queue
import struct
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

try:
    import webrtcvad

    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

logger = logging.getLogger(__name__)

# VAD constants
SAMPLE_RATE = 16000  # WebRTC VAD requires 8000, 16000, 32000, or 48000 Hz
FRAME_DURATION_MS = 30  # WebRTC VAD supports 10, 20, or 30 ms
BYTES_PER_SAMPLE = 2  # 16-bit audio


@dataclass
class AudioSegment:
    """Represents a segment of audio data."""

    data: bytes
    sample_rate: int
    duration_ms: int

    def to_numpy(self) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        return np.frombuffer(self.data, dtype=np.int16)

    def save_wav(self, filename: str) -> None:
        """Save audio segment to WAV file."""
        import wave

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.data)


class VoiceActivityDetector:
    """Real-time voice activity detection using WebRTC VAD.

    This class captures audio from the microphone and detects speech segments
    using the WebRTC VAD algorithm. It's designed to be OS-agnostic and work
    in Docker containers.

    Example:
        >>> vad = VoiceActivityDetector(aggressiveness=3)
        >>> for segment in vad.detect_speech(max_duration_s=10):
        ...     print(f"Speech detected: {len(segment.data)} bytes")
    """

    def __init__(
        self,
        aggressiveness: int = 3,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = FRAME_DURATION_MS,
        padding_duration_ms: int = 300,
        device: Optional[int] = None,
    ):
        """Initialize VAD detector.

        Args:
            aggressiveness: VAD aggressiveness (0-3). Higher = less sensitive.
                0: Most permissive, detects more speech
                3: Most aggressive, filters more non-speech
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in ms (must be 10, 20, or 30)
            padding_duration_ms: Amount of silence to keep after speech ends
            device: Audio input device index (None = default)
        """
        if not WEBRTCVAD_AVAILABLE:
            raise ImportError(
                "webrtcvad is required for VAD. Install with: pip install webrtcvad"
            )

        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000")

        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms")

        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.device = device

        # Calculate frame size in samples and bytes
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_size * BYTES_PER_SAMPLE

        # Calculate padding frames
        self.padding_frames = int(padding_duration_ms / frame_duration_ms)

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)

        logger.info(
            "VAD initialized: rate=%d, frame_ms=%d, aggr=%d",
            sample_rate,
            frame_duration_ms,
            aggressiveness,
        )

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Callback for audio input stream."""
        if status:
            logger.warning("Audio stream status: %s", status)

        # Convert float32 to int16
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Put in queue for processing
        self._audio_queue.put(audio_bytes)

    def detect_speech(
        self,
        max_duration_s: float = 30.0,
        min_speech_frames: int = 5,
    ) -> Iterator[AudioSegment]:
        """Detect and yield speech segments from microphone.

        Args:
            max_duration_s: Maximum recording duration in seconds
            min_speech_frames: Minimum consecutive speech frames to trigger

        Yields:
            AudioSegment objects containing detected speech

        Example:
            >>> vad = VoiceActivityDetector()
            >>> for segment in vad.detect_speech(max_duration_s=10):
            ...     segment.save_wav("speech.wav")
            ...     break  # Process first segment
        """
        self._audio_queue = queue.Queue()

        logger.info("Starting speech detection (max %ds)", max_duration_s)

        # Calculate max frames to process
        max_frames = int(max_duration_s * 1000 / self.frame_duration_ms)

        # State tracking
        triggered = False  # Currently in speech segment
        speech_frames = []  # Frames of current speech segment
        padding_frames_buffer = []  # Buffer for post-speech padding
        num_consecutive_speech = 0  # Consecutive speech frames

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.frame_size,
                device=self.device,
                callback=self._audio_callback,
            ):
                logger.info("Listening for speech...")

                frames_processed = 0

                while frames_processed < max_frames:
                    try:
                        # Get audio data with timeout
                        audio_data = self._audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    # Process in frame-sized chunks
                    for i in range(0, len(audio_data), self.frame_bytes):
                        frame = audio_data[i : i + self.frame_bytes]

                        # Skip incomplete frames
                        if len(frame) != self.frame_bytes:
                            continue

                        frames_processed += 1

                        # Run VAD on frame
                        is_speech = self.vad.is_speech(frame, self.sample_rate)

                        if not triggered:
                            # Waiting for speech to start
                            if is_speech:
                                num_consecutive_speech += 1
                                speech_frames.append(frame)

                                if num_consecutive_speech >= min_speech_frames:
                                    triggered = True
                                    logger.info("Speech started")
                            else:
                                num_consecutive_speech = 0
                                speech_frames = []

                        else:
                            # In speech segment
                            speech_frames.append(frame)

                            if not is_speech:
                                padding_frames_buffer.append(frame)

                                if len(padding_frames_buffer) > self.padding_frames:
                                    # End of speech segment
                                    logger.info("Speech ended")

                                    # Combine all frames into segment
                                    segment_bytes = b"".join(speech_frames)
                                    duration_ms = (
                                        len(speech_frames) * self.frame_duration_ms
                                    )

                                    segment = AudioSegment(
                                        data=segment_bytes,
                                        sample_rate=self.sample_rate,
                                        duration_ms=duration_ms,
                                    )

                                    yield segment

                                    # Reset state
                                    triggered = False
                                    speech_frames = []
                                    padding_frames_buffer = []
                                    num_consecutive_speech = 0
                            else:
                                # Still speech, clear padding buffer
                                padding_frames_buffer = []

                        # Check timeout
                        if frames_processed >= max_frames:
                            logger.info("Max duration reached")
                            break

        except Exception as e:
            logger.error("VAD error: %s", e)
            raise

        finally:
            # If we have a partial speech segment, yield it
            if speech_frames:
                logger.info("Yielding final segment")
                segment_bytes = b"".join(speech_frames)
                duration_ms = len(speech_frames) * self.frame_duration_ms

                segment = AudioSegment(
                    data=segment_bytes,
                    sample_rate=self.sample_rate,
                    duration_ms=duration_ms,
                )

                yield segment

    def list_devices(self) -> list[dict]:
        """List available audio input devices.

        Returns:
            List of device info dictionaries

        Example:
            >>> vad = VoiceActivityDetector()
            >>> devices = vad.list_devices()
            >>> for dev in devices:
            ...     print(f"{dev['index']}: {dev['name']}")
        """
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_input_channels"],
                        "sample_rate": dev["default_samplerate"],
                    }
                )
        return devices

    def test_microphone(self, duration_s: float = 3.0) -> bool:
        """Test microphone by recording and analyzing audio.

        Args:
            duration_s: Test duration in seconds

        Returns:
            True if microphone is working and detecting audio

        Example:
            >>> vad = VoiceActivityDetector()
            >>> if vad.test_microphone():
            ...     print("Microphone OK")
        """
        logger.info("Testing microphone for %ds...", duration_s)

        try:
            recording = sd.rec(
                int(duration_s * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.device,
            )
            sd.wait()

            # Check if we got audio data
            max_amplitude = np.max(np.abs(recording))
            logger.info("Max amplitude: %.3f", max_amplitude)

            if max_amplitude > 0.001:  # Some threshold
                logger.info("Microphone working")
                return True
            else:
                logger.warning("Microphone not detecting audio")
                return False

        except Exception as e:
            logger.error("Microphone test failed: %s", e)
            return False


class SimpleVAD:
    """Simplified VAD for environments without WebRTC VAD.

    Uses energy-based detection instead of ML-based detection.
    Less accurate but works everywhere.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = FRAME_DURATION_MS,
        energy_threshold: float = 0.01,
        device: Optional[int] = None,
    ):
        """Initialize simple VAD.

        Args:
            sample_rate: Audio sample rate
            frame_duration_ms: Frame duration in ms
            energy_threshold: Energy threshold for speech detection
            device: Audio input device index
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.device = device

        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        logger.info("Simple VAD initialized (energy-based)")

    def _calculate_energy(self, frame: np.ndarray) -> float:
        """Calculate energy of audio frame."""
        return np.sqrt(np.mean(frame**2))

    def detect_speech(
        self, max_duration_s: float = 30.0, min_speech_frames: int = 5
    ) -> Iterator[AudioSegment]:
        """Detect speech using energy-based method.

        Similar to VoiceActivityDetector.detect_speech() but uses
        simple energy threshold instead of WebRTC VAD.
        """
        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio stream status: %s", status)
            audio_queue.put(indata.copy())

        max_frames = int(max_duration_s * 1000 / self.frame_duration_ms)

        triggered = False
        speech_frames = []
        num_consecutive_speech = 0
        padding_frames = int(300 / self.frame_duration_ms)  # 300ms padding
        padding_buffer = []

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.frame_size,
                device=self.device,
                callback=callback,
            ):
                logger.info("Listening for speech...")

                frames_processed = 0

                while frames_processed < max_frames:
                    try:
                        frame = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    frames_processed += 1

                    # Calculate energy
                    energy = self._calculate_energy(frame)
                    is_speech = energy > self.energy_threshold

                    if not triggered:
                        if is_speech:
                            num_consecutive_speech += 1
                            speech_frames.append(frame)

                            if num_consecutive_speech >= min_speech_frames:
                                triggered = True
                                logger.info("🗣️ Speech started")
                        else:
                            num_consecutive_speech = 0
                            speech_frames = []
                    else:
                        speech_frames.append(frame)

                        if not is_speech:
                            padding_buffer.append(frame)

                            if len(padding_buffer) > padding_frames:
                                logger.info("🔇 Speech ended")

                                # Convert to bytes
                                audio_array = np.concatenate(speech_frames)
                                audio_int16 = (audio_array * 32767).astype(np.int16)
                                segment_bytes = audio_int16.tobytes()

                                duration_ms = (
                                    len(speech_frames) * self.frame_duration_ms
                                )

                                segment = AudioSegment(
                                    data=segment_bytes,
                                    sample_rate=self.sample_rate,
                                    duration_ms=duration_ms,
                                )

                                yield segment

                                triggered = False
                                speech_frames = []
                                padding_buffer = []
                                num_consecutive_speech = 0
                        else:
                            padding_buffer = []

        except Exception as e:
            logger.error("Simple VAD error: %s", e)
            raise

        finally:
            if speech_frames:
                logger.info("Yielding final segment")
                audio_array = np.concatenate(speech_frames)
                audio_int16 = (audio_array * 32767).astype(np.int16)
                segment_bytes = audio_int16.tobytes()
                duration_ms = len(speech_frames) * self.frame_duration_ms

                segment = AudioSegment(
                    data=segment_bytes,
                    sample_rate=self.sample_rate,
                    duration_ms=duration_ms,
                )

                yield segment

    def list_devices(self) -> list[dict]:
        """List available audio input devices."""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_input_channels"],
                        "sample_rate": dev["default_samplerate"],
                    }
                )
        return devices

    def test_microphone(self, duration_s: float = 3.0) -> bool:
        """Test microphone by recording and analyzing audio.

        Args:
            duration_s: Test duration in seconds

        Returns:
            True if microphone is working and detecting audio

        Example:
            >>> vad = SimpleVAD()
            >>> if vad.test_microphone():
            ...     print("Microphone OK")
        """
        logger.info("Testing microphone for %ds...", duration_s)

        try:
            recording = sd.rec(
                int(duration_s * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.device,
            )
            sd.wait()

            # Check if we got audio data
            max_amplitude = np.max(np.abs(recording))
            logger.info("Max amplitude: %.3f", max_amplitude)

            if max_amplitude > 0.001:  # Some threshold
                logger.info("Microphone working")
                return True
            else:
                logger.warning("Microphone not detecting audio")
                return False

        except Exception as e:
            logger.error("Microphone test failed: %s", e)
            return False
