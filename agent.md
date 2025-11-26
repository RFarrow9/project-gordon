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
- **Output:** Audio segments (when speech detected)
- **Performance target:** ~100-200ms detection latency

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
  - `__init__.py` - Module initialization
  - `README.md` - Human-readable documentation
  - Implementation files (to be added during refactor)

- Root files:
  - `voice_assistant.py` - Main orchestrator that connects all components
  - `test_ollama.py` - Setup verification script
  - `requirements.txt` - Python dependencies

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

### Performance Optimization

- Profile each component separately
- Document latency improvements in both README.md and agent.md
- Consider model size vs speed tradeoffs

## Current State

**Status:** Refactoring in progress (branch: `refactor`)
- ✅ Folder structure created
- ✅ Component READMEs written
- ⏳ Moving existing code into components
- ⏳ Adding independent tests for each component

## Technology Stack

- **STT:** OpenAI Whisper (local)
- **LLM:** Ollama (local) with LangChain
- **TTS:** pyttsx3 (offline)
- **Audio:** PyAudio for capture
- **Language:** Python 3.x

## Future Enhancements

Planned improvements (from README.md):
- Conversation memory/history
- Wake word detection
- Streaming TTS for lower latency
- Custom voice profiles
- Docker containerization

---

**Last Updated:** 2025-11-26 (Initial creation during modular refactor)
