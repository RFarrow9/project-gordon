"""Interactive CLI for testing Speech-to-Text."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Speech-to-Text (STT) with Whisper."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base.en",
        choices=["tiny.en", "base.en", "small.en", "medium.en"],
        help="Whisper model to use (default: base.en)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detect if not specified)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Audio file to transcribe (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live microphone with VAD",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness for live mode (default: 3)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    print("\n🎤 Speech-to-Text CLI")
    print("=" * 70)

    # Import here to avoid slow startup if just showing help
    from stt.core import SpeechToText

    # Initialize STT
    print(f"\n⏳ Loading Whisper model '{args.model}'...")
    stt = SpeechToText(model_name=args.model, device=args.device)

    info = stt.info()
    print(f"✅ Model loaded: {info['model']} on {info['device']}")

    # Show info and exit
    if args.info:
        print(f"\nModel: {info['model']}")
        print(f"Device: {info['device']}")
        print(f"Language: {info['language']}")
        return

    # Transcribe file
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"\n❌ Error: File not found: {args.file}")
            sys.exit(1)

        print(f"\n📄 Transcribing file: {args.file}")
        print("-" * 70)

        text = stt.transcribe_file(str(file_path))

        if text:
            print(f"\n💬 Transcription:\n{text}\n")
        else:
            print("\n❌ Transcription failed or returned empty\n")

        return

    # Live microphone with VAD
    if args.live:
        try:
            from vad import VoiceActivityDetector
        except ImportError:
            print("\n❌ Error: VAD module not found. Cannot use live mode.")
            print("Make sure the vad module is available.")
            sys.exit(1)

        print(f"\n🎙️  Starting live transcription with VAD...")
        print(f"VAD aggressiveness: {args.vad_aggressiveness}")
        print("\nSpeak into your microphone. Press Ctrl+C to stop.\n")
        print("-" * 70)

        try:
            vad = VoiceActivityDetector(aggressiveness=args.vad_aggressiveness)

            for segment in vad.detect_speech(max_duration_s=300):  # 5 min max
                print(f"\n🗣️  Speech detected ({segment.duration_ms}ms)")
                print("⏳ Transcribing...")

                text = stt.transcribe(segment)

                if text:
                    print(f"💬 You said: {text}\n")
                else:
                    print("❌ Could not transcribe (empty or failed)\n")

                print("-" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 Stopped by user")

        return

    # No action specified
    print("\n❌ Error: No action specified")
    print("\nUsage:")
    print("  --file <path>   Transcribe an audio file")
    print("  --live          Use live microphone with VAD")
    print("  --info          Show model information")
    print("\nRun with --help for more options")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
