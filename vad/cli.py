"""Interactive CLI for testing Voice Activity Detection."""
from __future__ import annotations

import argparse
import logging

from vad.core import SAMPLE_RATE, SimpleVAD, VoiceActivityDetector, WEBRTCVAD_AVAILABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Voice Activity Detection (VAD) with microphone input."
    )
    parser.add_argument(
        "--aggressiveness",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness level (0=permissive, 3=aggressive)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum recording duration in seconds",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Test microphone and exit",
    )
    parser.add_argument(
        "--test-all-devices",
        action="store_true",
        help="Test all devices to find which ones are working",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple energy-based VAD instead of WebRTC VAD",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save detected speech to WAV file (e.g., speech.wav)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def list_devices(vad) -> None:
    """List available audio input devices."""
    devices = vad.list_devices()

    if not devices:
        print("No audio input devices found")
        return

    print("Available Audio Input Devices:")
    print("=" * 60)
    for dev in devices:
        print(
            f"  [{dev['index']}] {dev['name']}\n"
            f"      Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']:.0f} Hz"
        )
    print("=" * 60)
    print(f"\nUse --device INDEX to select a specific device")


def test_microphone(vad) -> None:
    """Test microphone functionality."""
    print("Testing microphone...")
    print("Please speak or make noise during the test\n")

    if vad.test_microphone(duration_s=3.0):
        print("\nMicrophone is working correctly!")
    else:
        print("\nMicrophone test failed - check your device")


def test_all_devices() -> None:
    """Test all available devices to find working ones."""
    from vad.core import SimpleVAD
    import sounddevice as sd
    import numpy as np

    print("Testing all audio input devices...")
    print("Please make some noise (speak, tap mic, etc.) during the test")
    print("=" * 60)

    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append((i, dev))

    if not devices:
        print("No audio input devices found")
        return

    working_devices = []

    for idx, dev in devices:
        device_name = dev["name"][:50]  # Truncate long names
        print(f"\n[{idx}] {device_name}")
        print(f"    Testing...", end=" ", flush=True)

        try:
            # Quick 2-second test
            vad = SimpleVAD(device=idx)

            # Record for 2 seconds
            recording = sd.rec(
                int(2.0 * 16000),
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                device=idx,
            )
            sd.wait()

            # Check if we got audio data
            max_amplitude = np.max(np.abs(recording))

            if max_amplitude > 0.001:  # Threshold for "working"
                status = f"WORKING (amplitude: {max_amplitude:.4f})"
                working_devices.append((idx, device_name, max_amplitude))
                print(status)
            else:
                print(f"Silent (amplitude: {max_amplitude:.4f})")

        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")

    # Summary
    print("\n" + "=" * 60)
    if working_devices:
        print(f"\nFound {len(working_devices)} working device(s):\n")
        for idx, name, amp in sorted(working_devices, key=lambda x: x[2], reverse=True):
            print(f"  [{idx}] {name[:50]}")
            print(f"      Amplitude: {amp:.4f}")

        # Recommend the best one
        best_idx, best_name, best_amp = max(working_devices, key=lambda x: x[2])
        print(f"\nRECOMMENDED: Device [{best_idx}] (highest amplitude)")
        print(f"Use with: python -m vad.cli --device {best_idx}")
    else:
        print("\nNo working devices found.")
        print("Make sure you're making noise during the test!")
        print("Try speaking, tapping the microphone, or playing audio.")


def record_speech(vad, args: argparse.Namespace) -> None:
    """Record speech segments using VAD."""
    print(f"Voice Activity Detection Active")
    print(f"Aggressiveness: {args.aggressiveness if not args.simple else 'Simple VAD'}")
    print(f"Max Duration: {args.max_duration}s")
    print("=" * 60)
    print("Speak into your microphone...")
    print("The system will detect when you start and stop speaking")
    print("Press Ctrl+C to stop\n")

    try:
        segment_count = 0

        for segment in vad.detect_speech(max_duration_s=args.max_duration):
            segment_count += 1

            duration_s = segment.duration_ms / 1000
            size_kb = len(segment.data) / 1024

            print(
                f"\nSpeech segment #{segment_count} detected: "
                f"{duration_s:.2f}s, {size_kb:.1f} KB"
            )

            if args.save:
                # Add segment number to filename if multiple segments
                if segment_count == 1:
                    filename = args.save
                else:
                    base, ext = args.save.rsplit(".", 1) if "." in args.save else (args.save, "wav")
                    filename = f"{base}_{segment_count}.{ext}"

                segment.save_wav(filename)
                print(f"Saved to: {filename}")

            # Ask if user wants to continue
            try:
                choice = input("\nContinue listening? (y/n): ").strip().lower()
                if choice not in ["y", "yes"]:
                    print("Stopping VAD")
                    break
            except (EOFError, KeyboardInterrupt):
                print("\nStopping VAD")
                break

    except KeyboardInterrupt:
        print("\n\nVAD stopped by user")

    if segment_count == 0:
        print("\nNo speech detected. Try:")
        print("   - Speaking louder")
        print("   - Moving closer to microphone")
        print("   - Using --simple for less sensitive detection")
        print("   - Using --aggressiveness 0 for more permissive detection")


def main() -> None:
    args = parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Handle test-all-devices early (doesn't need VAD instance)
    if args.test_all_devices:
        test_all_devices()
        return

    # Create VAD instance
    if args.simple:
        vad = SimpleVAD(device=args.device)
    else:
        if not WEBRTCVAD_AVAILABLE:
            print("WARNING: WebRTC VAD not available, falling back to SimpleVAD")
            print("   Install with: pip install webrtcvad\n")
            vad = SimpleVAD(device=args.device)
        else:
            vad = VoiceActivityDetector(
                aggressiveness=args.aggressiveness,
                device=args.device,
            )

    # Handle commands
    if args.list_devices:
        list_devices(vad)
        return

    if args.test_mic:
        test_microphone(vad)
        return

    # Default: record speech
    record_speech(vad, args)


if __name__ == "__main__":
    main()
