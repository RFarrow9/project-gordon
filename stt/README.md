# Speech-to-Text (STT)

**Purpose:** Converts audio recordings into text using OpenAI Whisper.

## Components

- Whisper model loading and management
- Audio transcription engine
- Model selection (tiny, base, small, medium, large)

## Key Responsibilities

- Load and initialize Whisper models
- Transcribe audio segments to text
- Handle different audio formats and quality levels
- Provide fast, accurate transcriptions

## Performance Goals

- **Target latency:** 0.5-2 seconds (depending on model size)
- Balance accuracy vs speed based on model selection
- Efficient memory usage for local processing

## Models Available

- `tiny`: Fastest, lowest accuracy
- `base`: Good balance (default)
- `small`: Better accuracy, slower
- `medium`: High accuracy, much slower
- `large`: Best accuracy, very slow
