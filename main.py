import asyncio
import threading
import queue
import time
import os
import subprocess
import platform
from typing import Optional

import whisper

# Try new langchain import first, fallback to old
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

    print("⚠️  Consider upgrading: pip install -U langchain-ollama")

from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Conditional imports for audio - only import if not in WSL or if available
try:
    import pyaudio
    import wave

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not available - using text input mode")

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️  pyttsx3 not available - using alternative TTS")


class AudioHandler:
    """Handles audio I/O with automatic environment detection"""

    def __init__(self, tts_rate: int = 200):
        self.environment = self._detect_environment()
        self.tts_method = self._init_tts_method(tts_rate)
        self.recording_method = self._init_recording_method()

        print(f"🌍 Environment: {self.environment}")
        print(f"🔊 TTS Method: {self.tts_method}")
        print(f"🎤 Recording Method: {self.recording_method}")

    def _detect_environment(self) -> str:
        """Detect the current environment"""
        try:
            # Check if WSL
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return 'wsl'
        except:
            pass

        # Check if Linux
        if platform.system() == 'Linux':
            return 'linux'
        elif platform.system() == 'Darwin':
            return 'macos'
        elif platform.system() == 'Windows':
            return 'windows'

        return 'unknown'

    def _init_tts_method(self, rate: int) -> str:
        """Initialize the best available TTS method"""
        if self.environment == 'wsl':
            # Try Windows SAPI first
            if self._test_windows_sapi():
                return 'windows_sapi'

        # Try espeak-ng
        if self._test_espeak():
            return 'espeak'

        # Try pyttsx3
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', rate)
                self.pyttsx3_engine.setProperty('volume', 0.9)

                # Set voice preference
                voices = self.pyttsx3_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.pyttsx3_engine.setProperty('voice', voice.id)
                            break

                return 'pyttsx3'
            except:
                pass

        return 'text_only'

    def _init_recording_method(self) -> str:
        """Initialize the best available recording method"""
        if self.environment == 'wsl':
            # For now, use text input in WSL
            return 'text_input'

        if PYAUDIO_AVAILABLE:
            try:
                # Test PyAudio
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                p.terminate()
                if device_count > 0:
                    return 'pyaudio'
            except:
                pass

        return 'text_input'

    def _test_windows_sapi(self) -> bool:
        """Test if Windows SAPI is available"""
        try:
            result = subprocess.run([
                'powershell.exe', '-Command',
                'Add-Type -AssemblyName System.Speech; exit 0'
            ], capture_output=True, timeout=5, check=True)
            return True
        except:
            return False

    def _test_espeak(self) -> bool:
        """Test if espeak-ng is available"""
        try:
            subprocess.run(['espeak-ng', '--version'],
                           capture_output=True, check=True)
            return True
        except:
            return False

    def speak(self, text: str) -> bool:
        """Speak text using the best available method"""
        try:
            if self.tts_method == 'windows_sapi':
                return self._speak_windows_sapi(text)
            elif self.tts_method == 'espeak':
                return self._speak_espeak(text)
            elif self.tts_method == 'pyttsx3':
                return self._speak_pyttsx3(text)
            else:
                return self._speak_text_only(text)
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return self._speak_text_only(text)

    def _speak_windows_sapi(self, text: str) -> bool:
        """Use Windows SAPI via PowerShell"""
        escaped_text = text.replace("'", "''").replace('"', '`"')
        cmd = [
            'powershell.exe', '-Command',
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = 1; "
            f"$synth.Speak('{escaped_text}')"
        ]
        subprocess.run(cmd, check=True, timeout=30)
        return True

    def _speak_espeak(self, text: str) -> bool:
        """Use espeak-ng"""
        subprocess.run([
            'espeak-ng', text,
            '--speed=160', '--pitch=50', '--amplitude=100'
        ], check=True)
        return True

    def _speak_pyttsx3(self, text: str) -> bool:
        """Use pyttsx3"""
        self.pyttsx3_engine.say(text)
        self.pyttsx3_engine.runAndWait()
        return True

    def _speak_text_only(self, text: str) -> bool:
        """Text-only fallback"""
        print(f"🔊 [AUDIO]: {text}")
        return True

    def record_audio(self, chunk=1024, format_type=None, channels=1,
                     rate=16000, record_seconds=10, silence_threshold=500,
                     silence_chunks=20) -> Optional[str]:
        """Record audio using the best available method"""
        if self.recording_method == 'pyaudio':
            return self._record_pyaudio(chunk, format_type or pyaudio.paInt16,
                                        channels, rate, record_seconds,
                                        silence_threshold, silence_chunks)
        else:
            return self._record_text_input()

    def _record_pyaudio(self, chunk, format_type, channels, rate,
                        record_seconds, silence_threshold, silence_chunks) -> Optional[str]:
        """Record using PyAudio (original method)"""
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=format_type,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk
        )

        print("🎤 Listening... (speak now)")
        frames = []
        silent_chunks = 0
        started_talking = False

        try:
            for _ in range(0, int(rate / chunk * record_seconds)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

                volume = max(data)

                if volume > silence_threshold:
                    started_talking = True
                    silent_chunks = 0
                elif started_talking:
                    silent_chunks += 1
                    if silent_chunks > silence_chunks:
                        print("🔇 Silence detected, processing...")
                        break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        if not frames or not started_talking:
            print("❌ No speech detected")
            return None

        temp_filename = "temp_audio.wav"
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format_type))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        return temp_filename

    def _record_text_input(self) -> str:
        """Text input fallback"""
        print("💬 Voice input not available - using text input")
        return input("👤 You: ").strip()


class VoiceLLMAssistant:
    def __init__(self,
                 model_name: str = "llama3.2:3b",
                 whisper_model: str = "base",
                 tts_rate: int = 200):

        # Initialize audio handler first
        self.audio_handler = AudioHandler(tts_rate)

        # Initialize LLM
        self.llm = self._init_llm(model_name)

        # Initialize Whisper only if we might need it
        if self.audio_handler.recording_method == 'pyaudio':
            print("🔄 Loading Whisper model...")
            self.whisper_model = whisper.load_model(whisper_model)
        else:
            self.whisper_model = None
            print("⚠️  Skipping Whisper - using text input")

        # Audio settings (kept for compatibility)
        self.chunk = 1024
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 10
        self.silence_threshold = 500
        self.silence_chunks = 20

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
                callbacks=[StreamingStdOutCallbackHandler()],
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

    def record_audio(self) -> Optional[str]:
        """Record audio using the audio handler"""
        return self.audio_handler.record_audio(
            self.chunk, self.format, self.channels, self.rate,
            self.record_seconds, self.silence_threshold, self.silence_chunks
        )

    def transcribe_audio(self, audio_input) -> Optional[str]:
        """Convert speech to text using Whisper, or handle text input"""
        # If it's already text (from text input), return it
        if isinstance(audio_input, str) and not audio_input.endswith('.wav'):
            print(f"👤 You typed: {audio_input}")
            return audio_input

        # If we don't have Whisper loaded, we can't transcribe
        if not self.whisper_model:
            print("❌ No Whisper model available for transcription")
            return None

        # It's an audio file, transcribe it
        try:
            print("🔄 Transcribing...")
            result = self.whisper_model.transcribe(audio_input)
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
        """Convert text to speech using the audio handler"""
        try:
            self.is_speaking = True
            print("🔊 Speaking...")
            self.audio_handler.speak(text)
            self.is_speaking = False
        except Exception as e:
            print(f"❌ TTS error: {e}")
            self.is_speaking = False

    def process_conversation_turn(self):
        """Handle one complete conversation turn"""
        # Get user input (audio or text)
        user_input = self.record_audio()
        if not user_input:
            return True  # Continue listening

        # Transcribe if needed
        user_text = self.transcribe_audio(user_input)

        # Clean up temp file if it was an audio file
        if isinstance(user_input, str) and user_input.endswith('.wav'):
            try:
                os.remove(user_input)
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

        if self.audio_handler.recording_method == 'text_input':
            print("💡 Tip: Type your messages and press Enter")
        else:
            print("💡 Tip: Speak clearly and wait for the silence detection")

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
   pip install openai-whisper langchain-community ollama

3. Optional (for full audio support):
   - For PyAudio: pip install pyaudio
   - For pyttsx3: pip install pyttsx3
   - WSL: Audio will use Windows TTS automatically
   - Linux: sudo apt install espeak-ng

4. Run the assistant:
   python voice_assistant.py
""")


# Main execution
if __name__ == "__main__":
    try:
        # Initialize and run the assistant
        assistant = VoiceLLMAssistant(
            model_name="llama3.2:3b",
            whisper_model="base",
            tts_rate=200
        )
        assistant.run()

    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
        print("\n" + "=" * 50)
        setup_instructions()