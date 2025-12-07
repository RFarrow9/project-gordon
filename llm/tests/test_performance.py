"""Performance and latency benchmarks for LocalLLM.

These tests load real models and measure actual latency.
Run with: pytest llm/test_performance.py -v -s

Use markers to run specific tests:
  pytest llm/test_performance.py -m "not slow"  # Skip slow tests
  pytest llm/test_performance.py -m benchmark   # Only benchmarks
"""
from __future__ import annotations

import time
from typing import Dict, List

import pytest

from llm.core import LocalLLM

# Latency thresholds for different scenarios (in seconds)
THRESHOLDS = {
    "instant_response": 0.5,  # Almost instant for voice assistant
    "acceptable_response": 2.0,  # Good for interactive use
    "slow_response": 5.0,  # Starting to feel slow
}

# Test prompts of varying complexity
TEST_PROMPTS = {
    "minimal": "Hi",
    "short": "What is 2+2?",
    "medium": "Explain what Python is in one sentence.",
    "long": "Write a brief paragraph about the importance of testing in software development.",
}


class LatencyResult:
    """Container for latency measurement results."""

    def __init__(self, prompt: str, response: str, latency: float):
        self.prompt = prompt
        self.response = response
        self.latency = latency
        self.tokens = len(response.split())
        self.tokens_per_second = self.tokens / latency if latency > 0 else 0

    def __str__(self):
        return (
            f"Latency: {self.latency:.3f}s | "
            f"Tokens: {self.tokens} | "
            f"Speed: {self.tokens_per_second:.1f} tok/s | "
            f"Prompt: '{self.prompt[:30]}...'"
        )


def measure_latency(llm: LocalLLM, prompt: str, **kwargs) -> LatencyResult:
    """Measure latency of a single generation."""
    start = time.perf_counter()
    response = llm.generate(prompt, **kwargs)
    latency = time.perf_counter() - start

    return LatencyResult(prompt, response, latency)


def print_latency_report(results: List[LatencyResult], test_name: str):
    """Print a formatted latency report."""
    print(f"\n{'=' * 70}")
    print(f"📊 {test_name}")
    print(f"{'=' * 70}")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")

    if results:
        avg_latency = sum(r.latency for r in results) / len(results)
        avg_speed = sum(r.tokens_per_second for r in results) / len(results)

        print(f"\n{'─' * 70}")
        print(f"Average Latency: {avg_latency:.3f}s")
        print(f"Average Speed: {avg_speed:.1f} tokens/sec")

        # Performance assessment
        if avg_latency < THRESHOLDS["instant_response"]:
            print("✅ Performance: EXCELLENT - Instant responses!")
        elif avg_latency < THRESHOLDS["acceptable_response"]:
            print("👍 Performance: GOOD - Suitable for interactive use")
        elif avg_latency < THRESHOLDS["slow_response"]:
            print("⚠️  Performance: ACCEPTABLE - May feel slightly slow")
        else:
            print("❌ Performance: SLOW - Consider smaller model or quantization")

    print(f"{'=' * 70}\n")


@pytest.fixture(scope="module")
def tiny_llm():
    """Load the smallest model for fast testing (module-scoped for reuse)."""
    print("\n🔄 Loading TinyLlama model (this may take a moment)...")
    llm = LocalLLM(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_preference="cuda",
    )
    print(f"✅ Model loaded on {llm.info()['device']}")
    return llm


@pytest.fixture(scope="module")
def phi_llm():
    """Load Phi-3 model for realistic testing (module-scoped for reuse)."""
    print("\n🔄 Loading Phi-3-mini model (this may take a moment)...")
    llm = LocalLLM(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        device_preference="cuda",
    )
    print(f"✅ Model loaded on {llm.info()['device']}")
    return llm


@pytest.mark.benchmark
def test_cold_start_latency(tiny_llm):
    """Measure first-generation latency (includes any model warmup)."""
    result = measure_latency(
        tiny_llm,
        "Hello",
        max_new_tokens=20,
        temperature=0.7,
    )

    print_latency_report([result], "Cold Start Latency Test")

    # Cold start can be slower, but should still be reasonable
    assert result.latency < 10.0, f"Cold start too slow: {result.latency:.3f}s"


@pytest.mark.benchmark
def test_warm_generation_latency(tiny_llm):
    """Measure latency after model warmup (realistic use case)."""
    # Warmup generation
    tiny_llm.generate("Warmup", max_new_tokens=10)

    # Measure multiple warm generations
    results = []
    for name, prompt in TEST_PROMPTS.items():
        result = measure_latency(
            tiny_llm,
            prompt,
            max_new_tokens=50,
            temperature=0.7,
        )
        results.append(result)

    print_latency_report(results, "Warm Generation Latency Test (TinyLlama)")

    # Average warm generation should be fast
    avg_latency = sum(r.latency for r in results) / len(results)
    assert avg_latency < THRESHOLDS["acceptable_response"], \
        f"Average latency too high: {avg_latency:.3f}s"


@pytest.mark.benchmark
def test_short_response_latency(tiny_llm):
    """Test latency for very short responses (voice assistant use case)."""
    results = []

    # Test quick questions that should have short answers
    quick_prompts = [
        "Say yes or no: Is the sky blue?",
        "What is 5+3?",
        "Name one color.",
    ]

    for prompt in quick_prompts:
        result = measure_latency(
            tiny_llm,
            prompt,
            max_new_tokens=20,  # Force short responses
            temperature=0.7,
        )
        results.append(result)

    print_latency_report(results, "Short Response Latency Test")

    # Short responses should be very fast
    avg_latency = sum(r.latency for r in results) / len(results)
    assert avg_latency < THRESHOLDS["instant_response"], \
        f"Short responses too slow for voice assistant: {avg_latency:.3f}s"


@pytest.mark.benchmark
def test_batch_latency(tiny_llm):
    """Measure latency over a batch of requests."""
    results = []

    for i in range(5):
        result = measure_latency(
            tiny_llm,
            f"Test prompt number {i+1}",
            max_new_tokens=30,
            temperature=0.7,
        )
        results.append(result)

    print_latency_report(results, "Batch Latency Test (5 generations)")

    # Check for consistency
    latencies = [r.latency for r in results]
    avg = sum(latencies) / len(latencies)
    variance = sum((x - avg) ** 2 for x in latencies) / len(latencies)
    std_dev = variance ** 0.5

    print(f"Latency std dev: {std_dev:.3f}s")

    # Latency should be relatively consistent
    assert std_dev < avg * 0.5, f"High latency variance: {std_dev:.3f}s"


@pytest.mark.benchmark
def test_temperature_effect_on_latency(tiny_llm):
    """Test if temperature affects latency."""
    prompt = "What is the capital of France?"

    results = []
    for temp in [0.0, 0.5, 1.0]:
        result = measure_latency(
            tiny_llm,
            prompt,
            max_new_tokens=30,
            temperature=temp,
        )
        results.append(result)
        print(f"Temperature {temp}: {result.latency:.3f}s")

    print_latency_report(results, "Temperature Effect on Latency")


@pytest.mark.benchmark
def test_token_length_scaling(tiny_llm):
    """Test how latency scales with max_new_tokens."""
    prompt = "Count from 1 to 20"

    results = []
    for max_tokens in [20, 50, 100, 200]:
        result = measure_latency(
            tiny_llm,
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
        )
        results.append(result)
        print(f"Max tokens {max_tokens}: {result.latency:.3f}s ({result.tokens} actual tokens)")

    print_latency_report(results, "Token Length Scaling Test")


@pytest.mark.slow
@pytest.mark.benchmark
def test_phi3_model_latency(phi_llm):
    """Benchmark the default Phi-3 model (slower, more realistic)."""
    results = []

    for name, prompt in TEST_PROMPTS.items():
        result = measure_latency(
            phi_llm,
            prompt,
            max_new_tokens=50,
            temperature=0.7,
        )
        results.append(result)

    print_latency_report(results, "Phi-3-mini Latency Test")

    avg_latency = sum(r.latency for r in results) / len(results)
    print(f"\n📈 Comparison: Phi-3-mini avg latency: {avg_latency:.3f}s")


@pytest.mark.benchmark
def test_voice_assistant_simulation(tiny_llm):
    """Simulate a realistic voice assistant interaction."""
    conversation = [
        ("What's the weather like?", 30),
        ("Tell me a joke", 50),
        ("What is 15 times 7?", 20),
        ("Thank you", 20),
    ]

    results = []
    total_time = 0

    print("\n" + "=" * 70)
    print("🎤 Voice Assistant Simulation")
    print("=" * 70)

    for i, (prompt, max_tokens) in enumerate(conversation, 1):
        result = measure_latency(
            tiny_llm,
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
        )
        results.append(result)
        total_time += result.latency

        print(f"\nTurn {i}:")
        print(f"  User: {prompt}")
        print(f"  Assistant: {result.response[:100]}...")
        print(f"  Latency: {result.latency:.3f}s")

    print("\n" + "─" * 70)
    print(f"Total conversation time: {total_time:.3f}s")
    print(f"Average response latency: {total_time/len(results):.3f}s")
    print("=" * 70 + "\n")

    # For a voice assistant, we want fast responses
    avg_latency = total_time / len(results)
    if avg_latency < THRESHOLDS["instant_response"]:
        print("✅ EXCELLENT for voice assistant use!")
    elif avg_latency < THRESHOLDS["acceptable_response"]:
        print("👍 GOOD for voice assistant use")
    else:
        print("⚠️  May be too slow for smooth voice interaction")

    assert avg_latency < THRESHOLDS["acceptable_response"], \
        f"Voice assistant too slow: {avg_latency:.3f}s avg"


@pytest.mark.benchmark
def test_device_info(tiny_llm):
    """Display device and configuration info for context."""
    info = tiny_llm.info()

    print("\n" + "=" * 70)
    print("💻 System Information")
    print("=" * 70)
    print(f"Model: {info['model_id']}")
    print(f"Device: {info['device']}")
    print(f"Data type: {info['dtype']}")
    print(f"Quantized: {info['quantized']}")

    import torch
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70 + "\n")


# Pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "benchmark: performance benchmark tests")
    config.addinivalue_line("markers", "slow: slow tests that load large models")
