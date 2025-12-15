# Project Gordon - Agent Documentation

**For AI Agents/LLMs working on this codebase**

## Overview

Project Gordon is a low-latency voice-enabled local LLM assistant. The codebase is structured as a modular pipeline with four independent components. Whether this modularity is the right choice depends on how much inter-component optimization we're sacrificing for testability.

## Architecture Pipeline

```
Microphone → VAD → STT → LLM → TTS → Speakers
```

## Component Structure

### 1. `vad/` - Voice Activity Detection
- **Purpose:** Audio capture and speech detection
- **Input:** Raw microphone audio
- **Output:** `AudioSegment` objects (when speech detected)
- **Performance target:** ~100-200ms detection latency (not validated under real conditions)
- **Implementation:** WebRTC VAD algorithm with SimpleVAD fallback
- **Status:** Implemented but not tested with actual hardware or validated for latency
- **Dependencies:** `sounddevice`, `webrtcvad` (optional), `numpy`
- **Tradeoffs:** WebRTC VAD is solid but we're accepting its ~30ms frame processing limitation. SimpleVAD fallback may have different latency characteristics - untested.

### 2. `stt/` - Speech-to-Text
- **Purpose:** Transcribe audio to text using OpenAI Whisper
- **Input:** `AudioSegment` objects from VAD
- **Output:** Transcribed text string (empty string on failure)
- **Performance target:** 0.5-1.5s for base.en (not validated on actual hardware)
- **Implementation:** Whisper base.en model (English-only optimization)
- **Status:** Implemented but not integration tested with VAD or validated for accuracy/latency
- **Dependencies:** `openai-whisper`, `torch`, `numpy`
- **Tradeoffs:** Whisper is accurate but slow. We chose base.en (English-only, ~140MB) over multilingual for speed but locked ourselves to English. GPU required for acceptable latency - CPU fallback is 3-5x slower. Synchronous processing means we can't start LLM until transcription completes.

### 3. `llm/` - Language Model
- **Purpose:** Generate conversational responses using Ollama
- **Input:** User text from STT
- **Output:** Assistant response text
- **Performance target:** 1-3 seconds (model-dependent)
- **Default model:** `llama3.2:3b`
- **Framework:** LangChain for conversation management

### 4. `tts/` - Text-to-Speech
- **Purpose:** Convert text responses to speech using pyttsx3
- **Input:** LLM response text
- **Output:** Audio playback to speakers
- **Performance target:** 0.5-1 second

## Design Principles

1. **Modularity:** Each component is independently testable and swappable - but this means we can't do cross-component optimizations like starting TTS before LLM finishes. Question whether this tradeoff makes sense.
2. **Low Latency:** Total pipeline target is 2-6 seconds end-to-end - this is slow for a voice assistant. We haven't validated if this is acceptable UX.
3. **Privacy-First:** Everything runs locally, no external API calls - at the cost of worse model quality and higher hardware requirements.
4. **Performance Optimization:** Each stage can be profiled and optimized independently - assumes the bottleneck isn't in the handoffs between components.

## Code Organization

- Each component folder contains:
  - `__init__.py` - Module initialization and exports
  - `README.md` - Human-readable documentation
  - `core.py` - Core implementation (main service classes)
  - `cli.py` - Command-line interface for testing
  - `tests/` - Unit and integration tests
  - `pytest.ini` - Pytest configuration

- Root files:
  - `voice_assistant.py` - Main orchestrator that connects all components
  - `test_ollama.py` - Setup verification script
  - `requirements.txt` - Python dependencies

## Module Architecture Details

### VAD Module (`vad/`)

**Files:**
- `core.py` - Main implementation
  - `VoiceActivityDetector`: WebRTC-based VAD (primary)
  - `SimpleVAD`: Energy-based fallback
  - `AudioSegment`: Audio data container
- `cli.py` - CLI for testing microphone, listing devices, recording
- `tests/` - Comprehensive test suite
  - `test_core.py`: Unit tests for VAD classes
  - `conftest.py`: Fixtures for audio testing
  - Audio tests require `--with-audio` flag

**Key Design Decisions & Tradeoffs:**
- **OS-Agnostic**: Uses `sounddevice` instead of PyAudio for better cross-platform support - but adds another dependency and sounddevice has its own quirks on Windows. Not tested comprehensively on all platforms.
- **Docker-First**: Optimized for Linux/Docker but "works everywhere" is a claim that needs validation. Audio device passthrough in Docker is notoriously finicky.
- **WebRTC VAD**: Industry-standard doesn't mean optimal for our use case. We're locked into its frame size constraints (10/20/30ms) and aggressiveness levels. Alternative: train custom VAD on voice assistant data.
- **Generator Pattern**: `detect_speech()` yields segments as they're detected - clean API but forces sequential processing. Prevents batching or pipelining optimizations.
- **No File I/O in Core**: Audio segments are in-memory - assumes we have enough RAM for speech segments. Could be an issue with long utterances or memory-constrained devices.

**Integration Pattern:**
```python
from vad import VoiceActivityDetector, AudioSegment

vad = VoiceActivityDetector(aggressiveness=3)
for segment in vad.detect_speech(max_duration_s=30):
    # segment is AudioSegment ready for STT
    pass
```

### STT Module (`stt/`)

**Files:**
- `core.py` - Main implementation
  - `SpeechToText`: Whisper-based transcription with GPU acceleration
- `cli.py` - CLI for testing file and live transcription
- `tests/` - Unit tests (mocked to avoid slow model loading)
  - `test_core.py`: Unit tests for SpeechToText class
- `pytest.ini` - Test configuration

**Key Design Decisions & Tradeoffs:**
- **English-Only Optimization**: Uses `base.en` instead of multilingual base - 50% smaller and faster but cannot handle other languages. Accents and non-native speakers may see degraded accuracy.
- **Whisper Choice**: Chose accuracy over speed. Whisper is slow (0.5-2s) compared to streaming models (Vosk ~100ms). Alternative: Use Wav2Vec2 or cloud APIs, but quality/privacy tradeoffs.
- **GPU-First**: Auto-detects CUDA/MPS/CPU - same issues as LLM module (doesn't handle multi-GPU, may pick wrong device, no user override).
- **Synchronous API**: `transcribe()` blocks until complete - prevents pipeline parallelism. Can't start LLM processing while still transcribing. Alternative: Async API or streaming transcription.
- **VRAM Competition**: Loads another model into VRAM alongside LLM - may cause OOM on smaller GPUs. Base.en needs ~800MB FP16. No memory sharing between models.
- **Error Handling**: Returns empty string on failure - caller can't distinguish between "no speech detected" vs "transcription failed" vs "model error". Silent failures.
- **No Batching**: Processes one segment at a time - can't batch multiple segments for throughput optimization.

**Integration Pattern:**
```python
from vad import VoiceActivityDetector
from stt import SpeechToText

vad = VoiceActivityDetector(aggressiveness=3)
stt = SpeechToText(model_name="base.en")  # GPU auto-detected

for segment in vad.detect_speech(max_duration_s=30):
    text = stt.transcribe(segment)
    if text:
        print(f"Transcribed: {text}")
    # Empty string = failure or no speech (ambiguous)
```

### LLM Module (`llm/`)

**Files:**
- `core.py` - GPU-first local LLM with transformers
- `cli.py` - Interactive chat interface
- `tests/` - Performance and functional tests

**Key Design Decisions & Tradeoffs:**
- **GPU Preference**: Auto-detects CUDA/MPS/CPU - but doesn't handle multi-GPU scenarios or let user override. May pick wrong device.
- **4-bit Quantization**: Optional memory optimization - with quality degradation that's not quantified. No A/B testing of quantized vs full precision.
- **Small Models**: Phi-3, TinyLlama, Llama-3.2-1B support - trading quality for speed. These models will give worse responses than larger models. Have we validated the quality is acceptable?

## Development Guidelines

### When Making Changes

**IMPORTANT:** When committing code changes, refactors, or architectural updates:
1. Update `README.md` with human-friendly explanations
2. Update `agent.md` (this file) with technical details for AI agents
3. Keep both files in sync

### Adding New Features

- Identify which component(s) the feature belongs to
- Keep components loosely coupled
- Update component README if responsibilities change
- Add tests that can run in isolation

### OS-Agnostic Development

All components should work across Linux, Windows, and macOS - but "OS-agnostic" is aspirational, not current reality. We haven't tested comprehensively on all platforms.

**Critical thinking required:**
- Cross-platform libraries like `sounddevice` and `pathlib` are better than OS-specific ones, but they're not magic. They have their own bugs and platform-specific behavior.
- Fallback implementations (e.g., SimpleVAD) sound good but double the test surface area. Are we actually maintaining both paths?
- Docker/Linux priority means Windows and macOS are second-class citizens. Is that acceptable given Windows is a common dev environment?
- Audio device passthrough in Docker (`--device /dev/snd`) works in theory. Practice is messier - PulseAudio, ALSA, permissions, etc.

**When making cross-platform changes:**
- Test on the actual platform or clearly document it's untested
- Don't assume environment detection works - it usually has edge cases
- Question whether we need true cross-platform or if we should just pick Linux and document it

### Performance Optimization

- Profile each component separately
- Document latency improvements in both README.md and agent.md
- Consider model size vs speed tradeoffs

## Current State

**Status:** Refactoring in progress (branch: `refactor`)

**What exists:**
- Folder structure created
- Component READMEs written (documentation != working code)
- LLM module implemented (GPU-first local text generation) - not integration tested
- VAD module implemented (OS-agnostic voice activity detection claim) - not validated on actual hardware or different OS platforms
- STT module implemented (Whisper base.en with GPU acceleration) - not integration tested with VAD, accuracy/latency not validated
- TTS module - not started

**What's missing:**
- End-to-end integration testing of the full pipeline
- Performance validation against stated targets
- Real hardware testing (we've been coding in a vacuum)
- User testing to validate if 2-6 second latency is acceptable
- Error handling between component boundaries
- Production considerations (logging, monitoring, recovery)

## Technology Stack

### Current Modules
- **VAD:** WebRTC VAD + sounddevice (OS-agnostic audio)
- **STT:** OpenAI Whisper base.en (English-only, GPU-accelerated)
- **LLM:** Transformers + PyTorch (GPU-accelerated local models)

### Planned Modules
- **TTS:** pyttsx3 (offline)

### Infrastructure
- **Audio I/O:** sounddevice (cross-platform)
- **ML Framework:** PyTorch with CUDA/MPS support
- **Testing:** pytest with custom markers
- **Language:** Python 3.11+

## Future Enhancements

Planned improvements - though adding features before validating current ones is questionable:
- Conversation memory/history - will increase latency and memory usage
- Wake word detection - another dependency, another thing to tune, more latency
- Streaming TTS for lower latency - should this have been the initial design?
- Custom voice profiles - nice-to-have that distracts from core functionality
- Docker containerization - already claiming Docker-first, so what does this mean?

**Reality check:** Before adding features, validate the current architecture actually works and meets performance requirements.

---

## Working with This Codebase

**Expectations:**
- **Challenge assumptions:** If something seems suboptimal, it probably is. Question design decisions and propose alternatives.
- **Tradeoffs over perfection:** There are no perfect solutions. Every choice has costs. Make those costs explicit.
- **Validate claims:** "Works on all platforms", "low latency", "production-ready" - these need proof, not documentation.
- **Working code over documentation:** READMEs and tests mean nothing if the actual implementation doesn't work.
- **Be critical:** If you see technical debt, poor abstractions, or questionable patterns - call them out. Don't be diplomatic.

**Anti-patterns to avoid:**
- Implementing features before validating existing ones work
- Writing tests that mock everything (they prove nothing)
- Optimizing for "clean code" over actual performance
- Adding abstraction layers "for future flexibility" that never get used
- Claiming something is done when it's only implemented but not validated

**Last Updated:** 2025-12-15 (STT module implemented - Whisper base.en with GPU, not integration tested)
