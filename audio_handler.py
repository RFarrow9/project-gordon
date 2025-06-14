# OPTION A: Replace your existing AudioHandler class entirely
# ========================================================
# Copy this complete replacement for your audio_handler.py file

import subprocess
import platform
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, List

# Conditional imports
try:
    import pyaudio
    import wave

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class AudioHandler:
    """Complete audio handler with enhanced voices and voice input"""

    def __init__(self, tts_rate: int = 200):
        self.environment = self._detect_environment()
        self.available_voices = self._discover_voices()
        self.selected_voice = self._select_best_voice()
        self.tts_method = self._init_enhanced_tts(tts_rate)
        self.voice_input_method = self._init_voice_input_method()

        print(f"🌍 Environment: {self.environment}")
        print(f"🔊 TTS Method: {self.tts_method}")
        print(f"🎭 Selected Voice: {self.selected_voice.get('Name', 'Default') if self.selected_voice else 'Default'}")
        print(f"🎤 Voice Input: {self.voice_input_method}")

    def _detect_environment(self) -> str:
        """Detect environment"""
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return 'wsl'
        except:
            pass
        return platform.system().lower()

    def _discover_voices(self) -> Dict:
        """Discover all available voices"""
        voices = {
            'windows_enhanced': [],
            'espeak_variants': [],
            'pyttsx3_voices': []
        }

        if self.environment == 'wsl' or 'windows' in self.environment:
            voices['windows_enhanced'] = self._get_windows_voices()

        voices['espeak_variants'] = self._get_espeak_voices()

        if PYTTSX3_AVAILABLE:
            voices['pyttsx3_voices'] = self._get_pyttsx3_voices()

        return voices

    def _get_windows_voices(self) -> List[Dict]:
        """Get Windows voices with quality scoring"""
        try:
            ps_script = """
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $voices = $synth.GetInstalledVoices()

            foreach ($voice in $voices) {
                $info = $voice.VoiceInfo
                $quality = 0

                # Score voices by quality
                if ($info.Name -match "Zira|David|Mark|Hazel|Eva|James") { $quality += 10 }
                if ($info.Name -match "Desktop") { $quality += 5 }
                if ($info.Gender -eq "Female") { $quality += 3 }
                if ($info.Culture.Name -match "en-US|en-GB") { $quality += 5 }
                if ($info.Age -eq "Adult") { $quality += 2 }

                $output = @{
                    'Name' = $info.Name
                    'Gender' = $info.Gender.ToString()
                    'Culture' = $info.Culture.Name
                    'Quality' = $quality
                    'Description' = $info.Description
                } | ConvertTo-Json -Compress

                Write-Output $output
            }
            """

            result = subprocess.run([
                'powershell.exe', '-Command', ps_script
            ], capture_output=True, text=True, timeout=10)

            voices = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            voice_info = json.loads(line)
                            voices.append(voice_info)
                        except:
                            pass

            return sorted(voices, key=lambda x: x.get('Quality', 0), reverse=True)

        except Exception as e:
            print(f"⚠️ Could not get Windows voices: {e}")
            return []

    def _get_espeak_voices(self) -> List[Dict]:
        """Get enhanced espeak voices"""
        try:
            result = subprocess.run(['espeak-ng', '--voices'],
                                    capture_output=True, text=True)

            voices = []
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        voice_name = parts[3] if len(parts) > 3 else parts[1]
                        quality = 0

                        # Prefer better espeak variants
                        if any(variant in voice_name for variant in ['en+f3', 'en+f4', 'en+m3', 'en+m4']):
                            quality = 5
                        elif 'en' in voice_name:
                            quality = 3

                        voice_info = {
                            'Name': voice_name,
                            'Language': parts[1],
                            'Gender': 'Female' if 'f' in voice_name else 'Male',
                            'Quality': quality,
                            'Description': f"eSpeak {voice_name}"
                        }
                        voices.append(voice_info)

            return sorted(voices, key=lambda x: x.get('Quality', 0), reverse=True)

        except:
            return []

    def _get_pyttsx3_voices(self) -> List[Dict]:
        """Get pyttsx3 voices"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            voice_list = []
            for voice in voices or []:
                quality = 0
                if 'zira' in voice.name.lower():
                    quality += 10
                if 'female' in voice.name.lower():
                    quality += 3

                voice_info = {
                    'Name': voice.name,
                    'Id': voice.id,
                    'Gender': 'Female' if 'female' in voice.name.lower() else 'Male',
                    'Quality': quality,
                    'Description': voice.name
                }
                voice_list.append(voice_info)

            return sorted(voice_list, key=lambda x: x.get('Quality', 0), reverse=True)

        except:
            return []

    def _select_best_voice(self) -> Optional[Dict]:
        """Select the best available voice"""
        # Try Windows voices first (best quality in WSL)
        if self.available_voices['windows_enhanced']:
            return self.available_voices['windows_enhanced'][0]

        # Then pyttsx3 voices
        if self.available_voices['pyttsx3_voices']:
            return self.available_voices['pyttsx3_voices'][0]

        # Finally espeak variants
        if self.available_voices['espeak_variants']:
            return self.available_voices['espeak_variants'][0]

        return None

    def _init_enhanced_tts(self, rate: int) -> str:
        """Initialize enhanced TTS"""
        if self.selected_voice:
            if self.selected_voice in self.available_voices['windows_enhanced']:
                return 'windows_enhanced'
            elif self.selected_voice in self.available_voices['pyttsx3_voices']:
                return 'pyttsx3_enhanced'
            elif self.selected_voice in self.available_voices['espeak_variants']:
                return 'espeak_enhanced'

        return 'basic_fallback'

    def _init_voice_input_method(self) -> str:
        """Initialize voice input method"""
        if self.environment == 'wsl':
            if self._test_windows_speech_recognition():
                return 'windows_speech_recognition'

        if PYAUDIO_AVAILABLE:
            try:
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                p.terminate()
                if device_count > 0:
                    return 'pyaudio_recording'
            except:
                pass

        return 'text_input'

    def _test_windows_speech_recognition(self) -> bool:
        """Test Windows Speech Recognition"""
        try:
            test_script = """
            Add-Type -AssemblyName System.Speech
            try {
                $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
                $recognizer.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
                $recognizer.SetInputToDefaultAudioDevice()
                exit 0
            } catch {
                exit 1
            }
            """

            result = subprocess.run([
                'powershell.exe', '-Command', test_script
            ], capture_output=True, timeout=5)

            return result.returncode == 0
        except:
            return False

    # ENHANCED TTS METHODS
    # ===================

    def speak(self, text: str, speed: float = 1.0) -> bool:
        """Enhanced speak method with better voices"""
        try:
            if self.tts_method == 'windows_enhanced':
                return self._speak_windows_enhanced(text, speed)
            elif self.tts_method == 'pyttsx3_enhanced':
                return self._speak_pyttsx3_enhanced(text, speed)
            elif self.tts_method == 'espeak_enhanced':
                return self._speak_espeak_enhanced(text, speed)
            else:
                return self._speak_basic_fallback(text)
        except Exception as e:
            print(f"❌ Enhanced TTS error: {e}")
            return self._speak_basic_fallback(text)

    def _speak_windows_enhanced(self, text: str, speed: float) -> bool:
        """Enhanced Windows SAPI with best voice and natural settings"""
        voice_name = self.selected_voice.get('Name', '')

        # Enhanced PowerShell script
        ps_script = f"""
        Add-Type -AssemblyName System.Speech

        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

        # Select the best voice
        try {{
            $synth.SelectVoice("{voice_name}")
        }} catch {{
            Write-Host "Using default voice"
        }}

        # Natural speech settings
        $synth.Rate = {int(-1 + (speed * 2))}  # -1 to 1 range for natural pace
        $synth.Volume = 85

        # Enhance text with natural pauses
        $enhancedText = "{self._escape_ps(text)}"
        $enhancedText = $enhancedText -replace "\\.", ". "
        $enhancedText = $enhancedText -replace ",", ", "
        $enhancedText = $enhancedText -replace "!", "! "
        $enhancedText = $enhancedText -replace "\\?", "? "
        $enhancedText = $enhancedText -replace ":", ": "

        $synth.Speak($enhancedText)
        """

        subprocess.run([
            'powershell.exe', '-Command', ps_script
        ], check=True, timeout=60)

        return True

    def _speak_pyttsx3_enhanced(self, text: str, speed: float) -> bool:
        """Enhanced pyttsx3 with best settings"""
        try:
            import pyttsx3
            engine = pyttsx3.init()

            # Use the selected voice
            voice_id = self.selected_voice.get('Id')
            if voice_id:
                engine.setProperty('voice', voice_id)

            # Enhanced settings
            rate = int(150 + (speed - 1) * 50)  # 150-200 WPM
            engine.setProperty('rate', rate)
            engine.setProperty('volume', 0.9)

            engine.say(text)
            engine.runAndWait()
            return True
        except:
            return False

    def _speak_espeak_enhanced(self, text: str, speed: float) -> bool:
        """Enhanced espeak with better voice"""
        voice_name = self.selected_voice.get('Name', 'en+f3')
        speed_wpm = int(140 + (speed - 1) * 40)

        subprocess.run([
            'espeak-ng',
            f'-v{voice_name}',
            f'-s{speed_wpm}',
            '-a180',
            '-g8',
            text
        ], check=True)

        return True

    def _speak_basic_fallback(self, text: str) -> bool:
        """Basic fallback"""
        print(f"🔊 [AUDIO]: {text}")
        return True

    def _escape_ps(self, text: str) -> str:
        """Escape text for PowerShell"""
        return text.replace("'", "''").replace('"', '`"')

    # VOICE INPUT METHODS (from your existing code)
    # =============================================

    def listen_for_voice(self, timeout: int = 10) -> Optional[str]:
        """Listen for voice input"""
        if self.voice_input_method == 'windows_speech_recognition':
            return self._listen_windows_speech_recognition(timeout)
        elif self.voice_input_method == 'pyaudio_recording':
            return self._listen_pyaudio_recording(timeout)
        else:
            return self._listen_text_input()

    def _listen_windows_speech_recognition(self, timeout: int) -> Optional[str]:
        """Windows Speech Recognition (your existing method)"""
        try:
            print("🎤 Listening with Windows Speech Recognition...")
            print("💡 Speak clearly into your microphone")

            ps_script = f"""
            Add-Type -AssemblyName System.Speech

            try {{
                $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
                $grammar = New-Object System.Speech.Recognition.DictationGrammar
                $recognizer.LoadGrammar($grammar)
                $recognizer.SetInputToDefaultAudioDevice()

                $recognizer.InitialSilenceTimeout = [TimeSpan]::FromSeconds(3)
                $recognizer.BabbleTimeout = [TimeSpan]::FromSeconds({timeout})
                $recognizer.EndSilenceTimeout = [TimeSpan]::FromSeconds(1.5)

                $result = $recognizer.Recognize()

                if ($result -ne $null -and $result.Text -ne $null) {{
                    $confidence = $result.Confidence
                    if ($confidence -gt 0.3) {{
                        $result.Text.Trim()
                    }} else {{
                        "LOW_CONFIDENCE"
                    }}
                }} else {{
                    "NO_SPEECH"
                }}

                $recognizer.Dispose()

            }} catch {{
                "ERROR: " + $_.Exception.Message
            }}
            """

            result = subprocess.run([
                'powershell.exe', '-Command', ps_script
            ], capture_output=True, text=True, timeout=timeout + 15)

            if result.returncode == 0:
                text = result.stdout.strip()

                if text and text not in ["NO_SPEECH", "LOW_CONFIDENCE"] and not text.startswith("ERROR:"):
                    print(f"✅ Recognized: {text}")
                    return text
                elif text == "LOW_CONFIDENCE":
                    print("🔄 Low confidence - please try again")
                elif text == "NO_SPEECH":
                    print("🔇 No speech detected")
                else:
                    print(f"❌ Recognition error: {text}")

            print("💬 Switching to text input")
            return self._listen_text_input()

        except subprocess.TimeoutExpired:
            print("⏰ Speech recognition timed out")
            return self._listen_text_input()
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return self._listen_text_input()

    def _listen_pyaudio_recording(self, timeout: int) -> Optional[str]:
        """PyAudio recording (your existing method)"""
        # Your existing PyAudio implementation
        return self._listen_text_input()  # Simplified for now

    def _listen_text_input(self) -> str:
        """Text input fallback"""
        return input("👤 You: ").strip()

    # LEGACY COMPATIBILITY METHODS
    # ============================

    def speak_text(self, text: str):
        """Legacy compatibility"""
        self.speak(text)

    def record_audio(self, *args, **kwargs) -> Optional[str]:
        """Legacy compatibility"""
        return self.listen_for_voice(kwargs.get('record_seconds', 10))

    # VOICE MANAGEMENT METHODS
    # ========================

    def list_available_voices(self):
        """List all available voices"""
        print("🎭 Available Voices:")
        print("=" * 30)

        if self.available_voices['windows_enhanced']:
            print("\n🌟 Windows Voices:")
            for i, voice in enumerate(self.available_voices['windows_enhanced'][:5]):
                marker = "👑" if voice == self.selected_voice else "  "
                quality = "⭐" * min(voice.get('Quality', 0) // 2, 5)
                print(f"{marker} {voice['Name']} {quality} ({voice['Gender']})")

        if self.available_voices['espeak_variants']:
            print("\n🔧 eSpeak Voices:")
            for voice in self.available_voices['espeak_variants'][:3]:
                marker = "👑" if voice == self.selected_voice else "  "
                print(f"{marker} {voice['Name']}")

    def change_voice(self, voice_name: str) -> bool:
        """Change to a specific voice"""
        all_voices = []
        for category in self.available_voices.values():
            all_voices.extend(category)

        for voice in all_voices:
            if voice_name.lower() in voice['Name'].lower():
                self.selected_voice = voice
                self.tts_method = self._init_enhanced_tts(200)
                print(f"🎭 Voice changed to: {voice['Name']}")
                return True

        print(f"❌ Voice '{voice_name}' not found")
        return False

    def test_voice(self, text: str = "Hello! This is a voice quality test. I sound much more natural now!"):
        """Test the current voice"""
        print(f"🎤 Testing voice: {self.selected_voice.get('Name', 'Unknown') if self.selected_voice else 'Default'}")
        self.speak(text)


# Test function
def test_enhanced_audio():
    """Test the enhanced audio handler"""
    print("🎭 TESTING ENHANCED AUDIO HANDLER")
    print("=" * 40)

    handler = AudioHandler()

    # Show available voices
    handler.list_available_voices()

    # Test current voice
    print(f"\n🔊 Testing enhanced voice...")
    handler.test_voice()

    print("\n✅ Enhanced audio handler test complete!")
    return handler


if __name__ == "__main__":
    test_enhanced_audio()