# Project Gordon 🎤🤖

**A low-latency voice-enabled local LLM assistant**

Project Gordon is a locally-running voice assistant that combines speech recognition, a local language model, and text-to-speech to create a seamless conversational AI experience. Built with Python, LangChain, and optimized for minimal latency.

## ✨ Features

- 🎤 **Voice Input**: Real-time speech recognition using OpenAI Whisper
- 🧠 **Local LLM**: Powered by Ollama with LangChain integration
- 🔊 **Voice Output**: Natural text-to-speech responses
- ⚡ **Low Latency**: Optimized for fast response times
- 🔒 **Privacy First**: Everything runs locally - no data sent to external services
- 🎯 **Voice Activity Detection**: Automatically detects when you stop speaking
- 📱 **Simple Interface**: Just speak and listen - no typing required

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   # Visit https://ollama.ai and install for your OS
   # Then pull the recommended model:
   ollama pull llama3.2:3b
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Your Setup**
   ```bash
   python test_ollama.py
   ```

4. **Run Project Gordon**
   ```bash
   python voice_assistant.py
   ```

## 📋 Installation

### System Dependencies

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### Python Dependencies

Install all required packages:
```bash
pip install openai-whisper pyttsx3 pyaudio langchain-community ollama wave
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. **Start the assistant:**
   ```bash
   python voice_assistant.py
   ```

2. **Speak naturally** - Gordon will automatically detect when you start and stop talking

3. **Commands:**
   - Say "quit", "exit", "stop", or "goodbye" to end the conversation
   - Speak clearly and wait for the response
   - No wake word needed - just start talking!

## ⚙️ Configuration

### Model Selection

Edit the model settings in `voice_assistant.py`:

```python
assistant = VoiceLLMAssistant(
    model_name="llama3.2:3b",    # Try: mistral:7b, codellama:7b
    whisper_model="base",        # Options: tiny, base, small, medium, large
    tts_rate=200                 # Speech speed (words per minute)
)
```

### Performance Tuning

For **faster responses**:
- Use smaller models: `llama3.2:1b` or `mistral:7b`
- Reduce `whisper_model` to "tiny" or "base"
- Increase `tts_rate` for faster speech

For **better accuracy**:
- Use larger models: `llama3.1:8b` or `mixtral:8x7b`
- Upgrade `whisper_model` to "small" or "medium"
- Fine-tune `silence_threshold` for your microphone

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│   Whisper STT    │───▶│   User Text     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Speakers     │◀───│   pyttsx3 TTS    │◀───│  Ollama + LC    │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Components:**
- **Speech-to-Text**: OpenAI Whisper (runs locally)
- **LLM Backend**: Ollama with LangChain integration
- **Text-to-Speech**: pyttsx3 (offline TTS engine)
- **Voice Activity Detection**: Volume-based silence detection

## 🛠️ Development

### Project Structure

```
project-gordon/
├── voice_assistant.py      # Main assistant class
├── test_ollama.py         # Setup verification script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Running Tests

Verify your setup:
```bash
python test_ollama.py
```

### Adding Features

Common extensions:
- **Memory**: Add conversation history with LangChain memory
- **Wake Word**: Integrate wake word detection
- **Streaming TTS**: Use faster streaming text-to-speech
- **Custom Voices**: Add voice customization options

## 🔧 Troubleshooting

### Common Issues

**"Ollama connection failed"**
```bash
# Start Ollama service
ollama serve
# Or start the Ollama desktop app
```

**"No speech detected"**
- Check microphone permissions
- Adjust `silence_threshold` in the code
- Test with `python -m pyaudio` to verify audio setup

**"Model not found"**
```bash
# Install the default model
ollama pull llama3.2:3b
```

**Slow responses**
- Try a smaller model: `ollama pull llama3.2:1b`
- Reduce max tokens in the LLM config
- Use Whisper "tiny" model for faster transcription

### Performance Benchmarks

**Expected response times with llama3.2:3b:**
- Speech recognition: ~0.5-2 seconds
- LLM generation: ~1-3 seconds  
- Text-to-speech: ~0.5-1 second
- **Total latency: ~2-6 seconds**

## 🤝 Contributing

Ideas for contributions:
- Voice activity detection improvements
- Additional LLM backend support
- Streaming audio processing
- Mobile app integration
- Docker containerization

## 📄 License

MIT License - Feel free to modify and distribute!

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Ollama** for local LLM serving
- **LangChain** for LLM orchestration
- **pyttsx3** for text-to-speech

---

**Built with ❤️ for privacy-focused voice AI**