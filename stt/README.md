# STT (Speech-to-Text) Module

**Purpose:** Transcribes audio segments to text using OpenAI Whisper.

## Overview

The STT module converts speech audio into text using OpenAI's Whisper model. It's optimized for English and uses GPU acceleration when available. Designed to work seamlessly with AudioSegment objects from the VAD module.

## Features

- **English-Optimized**: Uses `base.en` model (smaller and faster than multilingual)
- **GPU-Accelerated**: Auto-detects CUDA/MPS/CPU
- **Swappable Design**: Easy to replace Whisper with alternative STT engines
- **File & Live Support**: Transcribe WAV files or live audio from VAD
- **Error Handling**: Graceful degradation on failures

## Quick Start

### Basic Usage

```python
from stt import SpeechToText
from vad import VoiceActivityDetector

# Initialize
stt = SpeechToText()  # Uses base.en by default
vad = VoiceActivityDetector()

# Transcribe live speech
for segment in vad.detect_speech(max_duration_s=30):
    text = stt.transcribe(segment)
    print(f"You said: {text}")
```

### Transcribe Audio File

```python
from stt import SpeechToText

stt = SpeechToText()
text = stt.transcribe_file("recording.wav")
print(text)
```

### Custom Model Selection

```python
from stt import SpeechToText

# Use tiny.en for faster (but less accurate) transcription
stt = SpeechToText(model_name="tiny.en", device="cuda")

# Or use small.en for better accuracy (slower)
stt = SpeechToText(model_name="small.en")
```

## Command Line Interface

### Transcribe a File

```bash
python -m stt.cli --file recording.wav
```

### Live Transcription with Microphone

```bash
python -m stt.cli --live
```

### Model Options

```bash
# Use tiny model for speed
python -m stt.cli --live --model tiny.en

# Use small model for accuracy
python -m stt.cli --live --model small.en

# Force CPU
python -m stt.cli --live --device cpu
```

### Show Model Info

```bash
python -m stt.cli --info
```

## Configuration

### Model Selection

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny.en` | ~40MB | Fastest | Low | Quick testing |
| `base.en` | ~140MB | Fast | Good | Default, balanced |
| `small.en` | ~460MB | Medium | Better | Quality over speed |
| `medium.en` | ~1.5GB | Slow | High | Maximum accuracy |

**Note:** `.en` models are English-only and significantly faster than multilingual models.

### Device Selection

- **Auto-detect** (default): Picks CUDA > MPS > CPU
- **CUDA**: For NVIDIA GPUs - fastest with FP16
- **MPS**: For Apple Silicon (M1/M2) - good performance
- **CPU**: Works everywhere but slower

## Performance

### Expected Latency

- **tiny.en on GPU**: ~0.3-0.5s
- **base.en on GPU**: ~0.5-1.5s (target)
- **small.en on GPU**: ~1-3s
- **base.en on CPU**: ~2-5s (not recommended)

Actual latency depends on:
- Audio segment length
- Hardware (GPU model, CPU speed)
- Model size
- System load

### VRAM Requirements

| Model | VRAM (FP32) | VRAM (FP16) |
|-------|-------------|-------------|
| tiny.en | ~1GB | ~500MB |
| base.en | ~1.5GB | ~800MB |
| small.en | ~2.5GB | ~1.3GB |
| medium.en | ~5GB | ~2.5GB |

## Integration with VAD

The STT module is designed to work with VAD's `AudioSegment`:

```python
from vad import VoiceActivityDetector, AudioSegment
from stt import SpeechToText

vad = VoiceActivityDetector(aggressiveness=3)
stt = SpeechToText(model_name="base.en")

for segment in vad.detect_speech(max_duration_s=30):
    # segment is AudioSegment (data, sample_rate, duration_ms)
    text = stt.transcribe(segment)
    
    if text:
        print(f"Transcribed ({segment.duration_ms}ms): {text}")
    else:
        print("Failed to transcribe or empty audio")
```

## Testing

### Run Unit Tests

```bash
cd stt
pytest
```

### Skip Integration Tests

```bash
pytest -m "not integration"
```

### Test with Actual Model (Slow)

```bash
pytest -m integration
```

## Troubleshooting

### Whisper Not Installed

```bash
pip install openai-whisper
```

### No GPU Detected

Whisper will fall back to CPU automatically. To force GPU:

```python
stt = SpeechToText(device="cuda")  # Will fail if CUDA unavailable
```

### Transcription Returns Empty String

Possible causes:
- Audio segment too short or silent
- Audio quality too poor
- Model hallucinating (returns non-speech)
- Sample rate mismatch (should be 16kHz)

### Slow Transcription on CPU

Use smaller model:
```python
stt = SpeechToText(model_name="tiny.en")
```

Or consider cloud STT APIs for CPU-only environments.

### VRAM Out of Memory

Use smaller model or enable CPU offload:
```python
# Use tiny.en instead of base.en
stt = SpeechToText(model_name="tiny.en")

# Or force CPU
stt = SpeechToText(device="cpu")
```

## Design Tradeoffs

### English-Only Models
**Choice:** Use `.en` models instead of multilingual

**Pros:**
- Smaller model size (~50% reduction)
- Faster inference
- Better accuracy for English

**Cons:**
- Cannot handle other languages
- Users speaking English with heavy accents may get worse results
- No language auto-detection

### GPU-First Design
**Choice:** Load model on GPU by default

**Pros:**
- Significantly faster inference (3-10x)
- Enables FP16 for additional speedup
- Better UX for voice assistant (lower latency)

**Cons:**
- Requires VRAM (competes with LLM module)
- Falls back to CPU without validation
- May select wrong GPU in multi-GPU systems

### Synchronous Transcription
**Choice:** Block until transcription completes

**Pros:**
- Simple API - easier to reason about
- Matches sequential pipeline design

**Cons:**
- Cannot process multiple segments in parallel
- Cannot start LLM processing while still transcribing
- Higher total latency in pipeline

**Alternative:** Async/streaming transcription (not implemented)

### Whisper Over Alternatives
**Choice:** Use OpenAI Whisper

**Pros:**
- State-of-the-art accuracy
- Well-maintained, popular library
- Handles poor audio quality reasonably well

**Cons:**
- Relatively slow compared to streaming models
- Large model sizes
- Not optimized for voice assistant use case (designed for long-form transcription)

**Alternatives Not Chosen:**
- Wav2Vec2: Faster but lower accuracy
- Streaming models (Vosk, Coqui): Lower latency but worse quality
- Cloud APIs: Better accuracy but violates privacy-first principle

## API Reference

### SpeechToText

**Constructor:**
- `model_name` (str): Whisper model name (default: "base.en")
- `device` (str | None): Device to use - "cuda", "mps", "cpu" (default: auto-detect)
- `language` (str): Language code (default: "en")

**Methods:**
- `transcribe(audio_segment) -> str`: Transcribe AudioSegment to text
- `transcribe_file(file_path) -> str`: Transcribe audio file to text
- `info() -> dict`: Get model configuration info

## Future Optimizations

Potential improvements (not currently implemented):

1. **Streaming Transcription**: Process audio as it arrives instead of waiting for full segment
2. **Batch Processing**: Transcribe multiple segments in parallel
3. **Model Quantization**: Use INT8 quantization for faster inference
4. **Sentence Chunking**: Return partial results as sentences complete
5. **VAD Integration**: Whisper has built-in VAD - could skip external VAD module
6. **Caching**: Cache model in memory across multiple instances

These are tradeoffs - each adds complexity and may not improve end-to-end latency.
