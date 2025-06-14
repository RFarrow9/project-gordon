import asyncio
import threading
import queue
import time
import os
from typing import Optional

import pyaudio
import wave
import whisper
import pyttsx3
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class VoiceLLMAssistant:
    def __init__(self,
                 model_name: str = "llama3.2:3b",  # Fast local model
                 whisper_model: str = "base",  # base, small, medium, large
                 tts_rate: int = 200):

        # Initialize components
        self.llm = self._init_llm(model_name)
        self.whisper_model = whisper.load_model(whisper_model)
        self.tts_engine = self._init_tts(tts_rate)

        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works best with 16kHz
        self.record_seconds = 10  # Max recording time
        self.silence_threshold = 500
        self.silence_chunks = 20  # Number of silent chunks before stopping

        # Threading
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False

        print("🎤 Voice LLM Assistant initialized!")
        print("📋 Available commands: 'quit', 'exit', 'stop'")

    def _init_llm(self, model_name: str) -> Ollama:
        """Initialize local LLM with Ollama"""
        try:
            llm = Ollama(
                model=model_name,
                temperature=0.7,
                # Enable streaming for lower perceived latency
                callbacks=[StreamingStdOutCallbackHandler()],
                # Reduce max tokens for faster responses
                num_predict=150,
            )
            # Test the connection
            llm.invoke("Hello")
            print(f"✅ LLM '{model_name}' loaded successfully")
            return llm
        except Exception as e:
            print(f"❌ Error loading LLM: {e}")
            print("Make sure Ollama is running and the model is installed:")
            print(f"   ollama pull {model_name}")
            raise

    def _init_tts(self, rate: int) -> pyttsx3.Engine:
        """Initialize text-to-speech engine"""
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', 0.9)

        # Get available voices and set a pleasant one
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a female voice for variety
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break

        print("🔊 TTS engine initialized")
        return engine

    def record_audio(self) -> Optional[str]:
        """Record audio from microphone until silence is detected"""
        audio = pyaudio.PyAudio()

        # Start recording
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print("🎤 Listening... (speak now)")
        frames = []
        silent_chunks = 0
        started_talking = False

        try:
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)

                # Simple volume-based voice activity detection
                volume = max(data)

                if volume > self.silence_threshold:
                    started_talking = True
                    silent_chunks = 0
                elif started_talking:
                    silent_chunks += 1
                    if silent_chunks > self.silence_chunks:
                        print("🔇 Silence detected, processing...")
                        break

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        if not frames or not started_talking:
            print("❌ No speech detected")
            return None

        # Save audio to temporary file
        temp_filename = "temp_audio.wav"
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))

        return temp_filename

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Convert speech to text using Whisper"""
        try:
            print("🔄 Transcribing...")
            result = self.whisper_model.transcribe(audio_file)
            text = result["text"].strip()

            if text:
                print(f"👤 You said: {text}")
                return text
            else:
                print("❌ No speech recognized")
                return None

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    def generate_response(self, text: str) -> str:
        """Generate response using local LLM"""
        try:
            print("🧠 Thinking...")

            # Create a prompt that encourages concise responses
            prompt = f"""You are a helpful voice assistant. Keep your responses conversational and concise (1-2 sentences max unless more detail is specifically requested).

User: {text}
Assistant:"""

            response = self.llm.invoke(prompt)
            print(f"🤖 Assistant: {response}")
            return response

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"❌ LLM error: {e}")
            return error_msg

    def speak_text(self, text: str):
        """Convert text to speech"""
        try:
            self.is_speaking = True
            print("🔊 Speaking...")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.is_speaking = False
        except Exception as e:
            print(f"❌ TTS error: {e}")
            self.is_speaking = False

    def process_conversation_turn(self):
        """Handle one complete conversation turn"""
        # Record audio
        audio_file = self.record_audio()
        if not audio_file:
            return True  # Continue listening

        # Transcribe speech
        user_text = self.transcribe_audio(audio_file)

        # Clean up temp file
        try:
            os.remove(audio_file)
        except:
            pass

        if not user_text:
            return True  # Continue listening

        # Check for quit commands
        if any(cmd in user_text.lower() for cmd in ['quit', 'exit', 'stop', 'goodbye']):
            self.speak_text("Goodbye! Have a great day!")
            return False  # Stop the conversation

        # Generate and speak response
        response = self.generate_response(user_text)
        if response:
            self.speak_text(response)

        return True  # Continue listening

    def run(self):
        """Main conversation loop"""
        print("\n🚀 Starting voice conversation...")
        print("💡 Tip: Speak clearly and wait for the beep before talking")

        # Welcome message
        welcome_msg = "Hello! I'm your voice assistant. How can I help you today?"
        print(f"🤖 {welcome_msg}")
        self.speak_text(welcome_msg)

        try:
            while True:
                print("\n" + "=" * 50)
                if not self.process_conversation_turn():
                    break

        except KeyboardInterrupt:
            print("\n👋 Conversation ended by user")
            self.speak_text("Goodbye!")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


# Installation and setup helper
def setup_instructions():
    """Print setup instructions"""
    print("""
🔧 SETUP INSTRUCTIONS:

1. Install Ollama:
   - Visit: https://ollama.ai
   - Download and install for your OS
   - Run: ollama pull llama3.2:3b

2. Install Python dependencies:
   pip install openai-whisper pyttsx3 pyaudio langchain-community ollama wave

3. For PyAudio installation issues:
   - Windows: pip install pipwin && pipwin install pyaudio
   - macOS: brew install portaudio && pip install pyaudio
   - Linux: sudo apt-get install portaudio19-dev python3-pyaudio

4. Run the assistant:
   python voice_assistant.py
""")


# Main execution
if __name__ == "__main__":
    try:
        # Initialize and run the assistant
        assistant = VoiceLLMAssistant(
            model_name="llama3.2:3b",  # Use a fast, small model
            whisper_model="base",  # Balance between speed and accuracy
            tts_rate=200  # Adjust speech speed
        )
        assistant.run()

    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
        print("\n" + "=" * 50)
        setup_instructions()