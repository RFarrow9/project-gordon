# Voice Activity Detection (VAD)

**Purpose:** Captures audio from the microphone and detects when speech starts and stops.

## Overview

The VAD module provides OS-agnostic voice activity detection using WebRTC VAD algorithm. It captures audio from any microphone and intelligently detects speech segments while filtering out silence and background noise.

## Features

- **OS-Agnostic**: Works on Linux, Windows, macOS, and Docker containers
- **WebRTC VAD**: Industry-standard voice activity detection algorithm
- **Simple Fallback**: Energy-based VAD for environments without WebRTC
- **Configurable Sensitivity**: Adjustable aggressiveness (0-3)
- **Audio Segments**: Returns clean audio segments ready for transcription
- **Device Selection**: List and select specific microphone devices

## Architecture

### Core Components

1. **`VoiceActivityDetector`** - Main VAD class using WebRTC algorithm
2. **`SimpleVAD`** - Fallback energy-based detection
3. **`AudioSegment`** - Data class representing audio segments
4. **CLI** - Command-line interface for testing

### Dependencies

- `sounddevice`: Cross-platform audio I/O
- `webrtcvad`: WebRTC voice activity detection (optional but recommended)
- `numpy`: Audio processing

## Usage

### Basic Usage

```python
from vad import VoiceActivityDetector

# Initialize VAD with aggressiveness level 3 (most aggressive)
vad = VoiceActivityDetector(aggressiveness=3)

# Detect speech segments
for segment in vad.detect_speech(max_duration_s=30):
    print(f"Speech detected: {segment.duration_ms}ms")
    segment.save_wav("speech.wav")
    break  # Process first segment
```

### List Available Microphones

```python
vad = VoiceActivityDetector()
devices = vad.list_devices()

for dev in devices:
    print(f"[{dev['index']}] {dev['name']}")
```

### Select Specific Device

```python
# Use device index from list_devices()
vad = VoiceActivityDetector(device=1)
```

### Simple VAD (No WebRTC)

```python
from vad import SimpleVAD

# Energy-based detection
vad = SimpleVAD(energy_threshold=0.01)

for segment in vad.detect_speech(max_duration_s=30):
    print(f"Speech detected: {segment.duration_ms}ms")
```

### Working with Audio Segments

```python
# Save to file
segment.save_wav("output.wav")

# Convert to numpy array
audio_array = segment.to_numpy()

# Access properties
print(f"Duration: {segment.duration_ms}ms")
print(f"Sample Rate: {segment.sample_rate}Hz")
print(f"Size: {len(segment.data)} bytes")
```

## Command Line Interface

### Test Microphone

```bash
python -m vad.cli --test-mic
```

### List Devices

```bash
python -m vad.cli --list-devices
```

### Record Speech

```bash
# Basic recording with default settings
python -m vad.cli

# Save to file
python -m vad.cli --save speech.wav

# Adjust sensitivity (0=permissive, 3=aggressive)
python -m vad.cli --aggressiveness 2

# Use specific device
python -m vad.cli --device 1

# Use simple VAD
python -m vad.cli --simple
```

## Configuration

### Aggressiveness Levels

- **0**: Most permissive - detects more speech, more false positives
- **1**: Moderate - balanced detection
- **2**: Balanced - good for most scenarios
- **3**: Most aggressive - filters more noise, may miss quiet speech

### Sample Rates

Supported sample rates for WebRTC VAD:
- 8000 Hz - Phone quality
- 16000 Hz - Default, good balance
- 32000 Hz - High quality
- 48000 Hz - Studio quality

### Frame Duration

Supported frame durations for WebRTC VAD:
- 10 ms - Fastest response
- 20 ms - Balanced
- 30 ms - Default, most stable

## Testing

### Run All Tests

```bash
cd vad
pytest
```

### Run with Audio Hardware Tests

```bash
pytest --with-audio
```

### Run Specific Tests

```bash
pytest tests/test_core.py::TestAudioSegment
```

### Skip Slow Tests

```bash
pytest -m "not slow"
```

## Performance

### Latency Targets

- VAD Detection: ~30-100ms (frame-based)
- End-to-end: ~100-200ms (including buffering)

### Resource Usage

- CPU: Minimal (<5% single core)
- Memory: ~10MB base + audio buffers
- Disk: Only if saving WAV files

## Docker Support

The VAD module is designed to work in Docker containers. For audio access in Docker:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install audio dependencies
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install sounddevice webrtcvad numpy

# Allow access to audio devices
# Run with: docker run --device /dev/snd:/dev/snd
```

```bash
# Run container with audio access
docker run --device /dev/snd:/dev/snd your-image
```

## Troubleshooting

### No Audio Detected

1. Check microphone permissions
2. List devices and verify correct device selected
3. Test microphone: `python -m vad.cli --test-mic`
4. Try lower aggressiveness: `--aggressiveness 0`
5. Try SimpleVAD: `--simple`

### WebRTC VAD Import Error

```bash
pip install webrtcvad
```

If compilation fails, use SimpleVAD as fallback.

### PortAudio/SoundDevice Errors

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install sounddevice
```

**macOS:**
```bash
brew install portaudio
pip install sounddevice
```

**Windows:**
```bash
pip install sounddevice
```

## Integration with STT

The VAD module outputs `AudioSegment` objects that can be directly passed to the STT module:

```python
from vad import VoiceActivityDetector
from stt import SpeechToText  # Future implementation

vad = VoiceActivityDetector()
stt = SpeechToText()

for segment in vad.detect_speech(max_duration_s=30):
    text = stt.transcribe(segment)
    print(f"Transcribed: {text}")
```

## API Reference

### VoiceActivityDetector

**Constructor:**
- `aggressiveness` (int): 0-3, higher = more aggressive filtering
- `sample_rate` (int): 8000, 16000, 32000, or 48000 Hz
- `frame_duration_ms` (int): 10, 20, or 30 ms
- `padding_duration_ms` (int): Silence padding after speech
- `device` (int | None): Audio device index

**Methods:**
- `detect_speech(max_duration_s, min_speech_frames)`: Yield speech segments
- `list_devices()`: List available microphones
- `test_microphone(duration_s)`: Test microphone functionality

### AudioSegment

**Attributes:**
- `data` (bytes): Raw audio data
- `sample_rate` (int): Sample rate in Hz
- `duration_ms` (int): Duration in milliseconds

**Methods:**
- `to_numpy()`: Convert to numpy array
- `save_wav(filename)`: Save as WAV file

## Performance Goals

- ✅ Minimize false positives (detecting silence as speech)
- ✅ Minimize false negatives (missing speech)
- ✅ Low latency detection (~100-200ms)
- ✅ OS-agnostic implementation
- ✅ Docker-compatible
