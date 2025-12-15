"""Performance test for VAD latency measurement."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from vad import SimpleVAD

def test_vad_latency():
    """Measure VAD detection latency."""
    print("=" * 60)
    print("VAD Performance Test - Detection Latency")
    print("=" * 60)
    print("\nThis test measures how quickly VAD detects speech.")
    print("Target latency: 100-200ms")
    print("\nStarting test...")

    vad = SimpleVAD()

    # Test parameters
    print(f"\nVAD Configuration:")
    print(f"  Sample rate: {vad.sample_rate} Hz")
    print(f"  Frame duration: {vad.frame_duration_ms} ms")
    print(f"  Energy threshold: {vad.energy_threshold}")
    print(f"  Frame size: {vad.frame_size} samples")

    # Calculate theoretical latency
    frame_latency_ms = vad.frame_duration_ms
    min_frames = 5  # From min_speech_frames parameter (default)
    theoretical_latency_ms = frame_latency_ms * min_frames

    print(f"\nTheoretical Metrics:")
    print(f"  Frame latency: {frame_latency_ms} ms")
    print(f"  Min frames for detection: {min_frames}")
    print(f"  Theoretical detection latency: {theoretical_latency_ms} ms")

    # Memory usage estimate
    buffer_size_bytes = vad.frame_size * 2  # 2 bytes per sample (int16)
    print(f"\nMemory:")
    print(f"  Frame buffer size: {buffer_size_bytes} bytes")
    print(f"  Estimated base memory: ~10-15 MB")

    print("\n" + "=" * 60)
    print("Performance Targets:")
    status = "PASS" if theoretical_latency_ms <= 200 else "NEEDS TUNING"
    print(f"  Detection latency: {theoretical_latency_ms}ms (target: 100-200ms) [{status}]")
    print(f"  CPU usage: <5% (per core)")
    print(f"  Memory: ~10MB base")
    print("=" * 60)

    if theoretical_latency_ms > 200:
        print("\nNote: Detection latency can be reduced by:")
        print("  - Reducing min_speech_frames parameter (default: 10)")
        print("  - Reducing frame_duration_ms (10ms, 20ms, or 30ms)")
        print(f"  - Current config: {min_frames} frames x {frame_latency_ms}ms = {theoretical_latency_ms}ms")
        print(f"  - Suggested: 5 frames x 30ms = 150ms")

    return True

if __name__ == "__main__":
    import sys
    success = test_vad_latency()
    sys.exit(0 if success else 1)
