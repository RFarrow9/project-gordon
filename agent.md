# Project Gordon - Agent Documentation

**For AI Agents/LLMs working on this codebase**

## Overview

Project Gordon is a low-latency voice-enabled local LLM assistant. The codebase is structured as a modular pipeline with four independent, testable components.

## Architecture Pipeline

```
Microphone → VAD → STT → LLM → TTS → Speakers
```

## Component Structure

### 1. `vad/` - Voice Activity Detection
- **Purpose:** Audio capture and speech detection
- **Input:** Raw microphone audio
- **Output:** `AudioSegment` objects (when speech detected)
- **Performance target:** ~100-200ms detection latency
- **Implementation:** WebRTC VAD algorithm with SimpleVAD fallback
- **Status:** ✅ Complete
- **Dependencies:** `sounddevice`, `webrtcvad` (optional), `numpy`

### 2. `stt/` - Speech-to-Text
- **Purpose:** Transcribe audio to text using OpenAI Whisper
- **Input:** Audio segments from VAD
- **Output:** Transcribed text
- **Performance target:** 0.5-2 seconds (model-dependent)
- **Default model:** `base` (balance of speed/accuracy)

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

1. **Modularity:** Each component is independently testable and swappable
2. **Low Latency:** Total pipeline target is 2-6 seconds end-to-end
3. **Privacy-First:** Everything runs locally, no external API calls
4. **Performance Optimization:** Each stage can be profiled and optimized independently

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

**Key Design Decisions:**
- **OS-Agnostic**: Uses `sounddevice` instead of PyAudio for better cross-platform support
- **Docker-First**: Optimized for Linux/Docker but works everywhere
- **WebRTC VAD**: Industry-standard algorithm, optional SimpleVAD fallback
- **Generator Pattern**: `detect_speech()` yields segments as they're detected
- **No File I/O in Core**: Audio segments are in-memory, saving is optional

**Integration Pattern:**
```python
from vad import VoiceActivityDetector, AudioSegment

vad = VoiceActivityDetector(aggressiveness=3)
for segment in vad.detect_speech(max_duration_s=30):
    # segment is AudioSegment ready for STT
    pass
```

### LLM Module (`llm/`)

**Files:**
- `core.py` - GPU-first local LLM with transformers
- `cli.py` - Interactive chat interface
- `tests/` - Performance and functional tests

**Key Design Decisions:**
- **GPU Preference**: Auto-detects CUDA/MPS/CPU
- **4-bit Quantization**: Optional memory optimization
- **Small Models**: Phi-3, TinyLlama, Llama-3.2-1B support

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

**IMPORTANT:** All components must work across Linux, Windows, and macOS. When Docker/Linux is the primary target:

**Do:**
- ✅ Use cross-platform libraries (`sounddevice` over PyAudio, `pathlib` over `os.path`)
- ✅ Test on multiple platforms when possible
- ✅ Provide fallback implementations (e.g., SimpleVAD when WebRTC unavailable)
- ✅ Document Docker-specific setup in README
- ✅ Use environment detection sparingly and only when necessary

**Don't:**
- ❌ Hardcode OS-specific paths or commands in core logic
- ❌ Use Windows-only or Linux-only libraries without fallbacks
- ❌ Assume specific audio device names/indices
- ❌ Rely on OS-specific features without graceful degradation

**Docker Priority:**
- When choosing between equivalent libraries, prefer the one that works best in Docker/Linux
- Ensure audio devices can be passed through (`--device /dev/snd`)
- Document required system packages (e.g., `portaudio19-dev`)

### Performance Optimization

- Profile each component separately
- Document latency improvements in both README.md and agent.md
- Consider model size vs speed tradeoffs

## Current State

**Status:** Refactoring in progress (branch: `refactor`)
- ✅ Folder structure created
- ✅ Component READMEs written
- ✅ LLM module complete (GPU-first local text generation)
- ✅ VAD module complete (OS-agnostic voice activity detection)
- ⏳ STT module (Speech-to-Text with Whisper)
- ⏳ TTS module (Text-to-Speech)

## Technology Stack

### Current Modules
- **VAD:** WebRTC VAD + sounddevice (OS-agnostic audio)
- **LLM:** Transformers + PyTorch (GPU-accelerated local models)

### Planned Modules
- **STT:** OpenAI Whisper (local)
- **TTS:** pyttsx3 (offline)

### Infrastructure
- **Audio I/O:** sounddevice (cross-platform)
- **ML Framework:** PyTorch with CUDA/MPS support
- **Testing:** pytest with custom markers
- **Language:** Python 3.11+

## Future Enhancements

Planned improvements (from README.md):
- Conversation memory/history
- Wake word detection
- Streaming TTS for lower latency
- Custom voice profiles
- Docker containerization

---

**Last Updated:** 2025-12-14 (VAD module completed - OS-agnostic implementation)
