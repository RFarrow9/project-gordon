#!/usr/bin/env python3
"""
Quick Local LLM Performance Test
Tests the local LLM and measures response latency
Run this directly without pytest for a quick check
"""

import sys
import time
from typing import List, Tuple

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass  # If it fails, emojis will be skipped but script will run


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_result(label: str, success: bool, details: str = ""):
    """Print a test result."""
    icon = "✅" if success else "❌"
    print(f"{icon} {label}")
    if details:
        print(f"   {details}")


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")

    deps = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
    }

    all_installed = True
    for module, name in deps.items():
        try:
            __import__(module)
            print_result(f"{name} installed", True)
        except ImportError:
            print_result(f"{name} not found", False, f"Install with: pip install {module}")
            all_installed = False

    return all_installed


def check_gpu() -> str:
    """Check GPU availability."""
    print_header("Checking GPU")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_result("CUDA GPU available", True, f"{device_name} ({memory:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print_result("Apple MPS GPU available", True, "Metal Performance Shaders")
            return "mps"
        else:
            print_result("No GPU found", False, "Will use CPU (slower)")
            return "cpu"
    except Exception as e:
        print_result("GPU check failed", False, str(e))
        return "cpu"


def test_model_loading(model_id: str, device: str) -> Tuple[bool, object]:
    """Test loading the model."""
    print_header(f"Loading Model: {model_id}")

    try:
        from llm.local_llm import LocalLLM

        print(f"⏳ Loading model (this may take a minute)...")
        start = time.perf_counter()

        llm = LocalLLM(model_id=model_id, device_preference=device)

        load_time = time.perf_counter() - start

        info = llm.info()
        print_result(
            "Model loaded successfully",
            True,
            f"Device: {info['device']}, Load time: {load_time:.2f}s",
        )

        return True, llm

    except Exception as e:
        print_result("Model loading failed", False, str(e))
        return False, None


def measure_latency(llm, prompt: str, max_tokens: int = 50) -> Tuple[float, str]:
    """Measure generation latency."""
    start = time.perf_counter()
    response = llm.generate(prompt, max_new_tokens=max_tokens, temperature=0.7)
    latency = time.perf_counter() - start
    return latency, response


def test_generation_latency(llm) -> bool:
    """Test generation with latency measurements."""
    print_header("Testing Generation Latency")

    test_cases = [
        ("Hello", 20, "Minimal prompt"),
        ("What is 2+2?", 30, "Simple question"),
        ("Explain Python in one sentence.", 50, "Medium prompt"),
    ]

    latencies: List[float] = []
    all_passed = True

    for i, (prompt, max_tokens, description) in enumerate(test_cases, 1):
        print(f"\n{i}. {description}: '{prompt}'")

        try:
            latency, response = measure_latency(llm, prompt, max_tokens)
            latencies.append(latency)

            # Count tokens
            tokens = len(response.split())
            speed = tokens / latency if latency > 0 else 0

            print(f"   ⏱️  Latency: {latency:.3f}s")
            print(f"   🔢 Tokens: {tokens} ({speed:.1f} tok/s)")
            print(f"   💬 Response: {response[:80]}...")

            # Check if acceptable
            if latency < 0.5:
                print(f"   ✅ EXCELLENT - Instant response!")
            elif latency < 2.0:
                print(f"   👍 GOOD - Acceptable for voice assistant")
            elif latency < 5.0:
                print(f"   ⚠️  ACCEPTABLE - May feel slightly slow")
            else:
                print(f"   ❌ SLOW - Too slow for interactive use")
                all_passed = False

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_passed = False

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n{'─' * 70}")
        print(f"📊 Average Latency: {avg_latency:.3f}s")

        if avg_latency < 0.5:
            print("🚀 Performance: EXCELLENT for voice assistant!")
        elif avg_latency < 2.0:
            print("👍 Performance: GOOD for voice assistant")
        elif avg_latency < 5.0:
            print("⚠️  Performance: ACCEPTABLE but may be slow")
        else:
            print("❌ Performance: TOO SLOW for voice assistant")
            print("💡 Try: TinyLlama model or enable --quantize")

    return all_passed


def test_conversation_flow(llm) -> bool:
    """Test a realistic conversation with latency tracking."""
    print_header("Voice Assistant Conversation Simulation")

    conversation = [
        ("Hi there!", 30),
        ("What's 15 times 7?", 20),
        ("Thank you!", 20),
    ]

    total_time = 0
    all_passed = True

    for i, (prompt, max_tokens) in enumerate(conversation, 1):
        print(f"\n💬 Turn {i}")
        print(f"   User: {prompt}")

        try:
            latency, response = measure_latency(llm, prompt, max_tokens)
            total_time += latency

            print(f"   Assistant: {response[:100]}")
            print(f"   ⏱️  Response time: {latency:.3f}s")

            if latency > 2.0:
                print(f"   ⚠️  Slow response for voice interaction")
                all_passed = False

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_passed = False

    avg_latency = total_time / len(conversation) if conversation else 0
    print(f"\n{'─' * 70}")
    print(f"Total conversation time: {total_time:.3f}s")
    print(f"Average response latency: {avg_latency:.3f}s")

    return all_passed


def main():
    """Main test routine."""
    print("\n🧪 LOCAL LLM PERFORMANCE TEST")
    print("=" * 70)

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ CRITICAL: Missing dependencies!")
        print("Install with: pip install torch transformers")
        return False

    # Step 2: Check GPU
    device = check_gpu()

    # Step 3: Determine model to test
    print_header("Model Selection")
    print("Using TinyLlama for fast testing")
    print("(You can edit this script to test other models)")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Step 4: Load model
    model_loaded, llm = test_model_loading(model_id, device)
    if not model_loaded:
        print("\n❌ CRITICAL: Model failed to load!")
        return False

    # Step 5: Test generation latency
    latency_passed = test_generation_latency(llm)

    # Step 6: Test conversation flow
    conversation_passed = test_conversation_flow(llm)

    # Final summary
    print_header("TEST SUMMARY")

    print_result("Dependencies installed", True)
    print_result(f"GPU available ({device})", device != "cpu")
    print_result("Model loading", model_loaded)
    print_result("Generation latency", latency_passed)
    print_result("Conversation flow", conversation_passed)

    if model_loaded and latency_passed and conversation_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print(f"Your {model_id} is ready for use!")

        if device == "cpu":
            print("\n💡 TIP: GPU would significantly improve performance")
        elif not latency_passed:
            print("\n💡 TIP: Try a smaller model or enable quantization for faster responses")

        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Review the issues above.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
