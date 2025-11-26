# Voice Activity Detection (VAD)

**Purpose:** Captures audio from the microphone and detects when speech starts and stops.

## Components

- Audio capture from microphone
- Real-time voice activity detection
- Silence detection and threshold management
- Audio buffer management

## Key Responsibilities

- Listen to microphone input
- Detect when user starts speaking
- Detect when user stops speaking (silence detection)
- Provide clean audio segments for transcription

## Performance Goals

- Minimize false positives (detecting silence as speech)
- Minimize false negatives (missing speech)
- Low latency detection (~100-200ms)
