"""Live test for VAD speech detection with real microphone."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vad import SimpleVAD

def test_live_vad():
    """Test VAD with real microphone input."""
    print("=" * 60)
    print("Live VAD Speech Detection Test")
    print("=" * 60)
    print("\nThis test will:")
    print("1. Listen to your microphone for up to 10 seconds")
    print("2. Detect when you start speaking")
    print("3. Detect when you stop speaking")
    print("4. Save the audio segment")
    print("\nInstructions:")
    print("- Wait 2 seconds, then speak a sentence")
    print("- Wait 2 seconds after speaking")
    print("- The system will detect your speech and stop")
    print("\nStarting in 3 seconds...")

    import time
    time.sleep(3)

    print("\n" + "=" * 60)
    print("LISTENING... Speak now!")
    print("=" * 60 + "\n")

    vad = SimpleVAD()
    segment_count = 0

    try:
        for segment in vad.detect_speech(max_duration_s=10.0):
            segment_count += 1
            duration_s = segment.duration_ms / 1000
            size_kb = len(segment.data) / 1024

            print(f"\nSpeech segment #{segment_count} detected!")
            print(f"  Duration: {duration_s:.2f}s")
            print(f"  Size: {size_kb:.1f} KB")
            print(f"  Sample rate: {segment.sample_rate} Hz")

            # Save the segment
            filename = f"test_speech_{segment_count}.wav"
            segment.save_wav(filename)
            print(f"  Saved: {filename}")

            # Only capture first segment for this test
            break

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False

    print("\n" + "=" * 60)
    if segment_count > 0:
        print("SUCCESS! VAD detected speech")
        print(f"Captured {segment_count} segment(s)")
        print("\nYou can play the audio file to verify:")
        print(f"  test_speech_1.wav")
        return True
    else:
        print("WARNING: No speech detected")
        print("\nTroubleshooting:")
        print("- Make sure microphone is working (test with --test-mic)")
        print("- Speak louder or closer to microphone")
        print("- Check microphone isn't muted in Windows settings")
        return False

if __name__ == "__main__":
    success = test_live_vad()
    sys.exit(0 if success else 1)
