#!/usr/bin/env python3
"""
Ollama Test Script
Tests if Ollama is running and models are working correctly
"""

import subprocess
import sys
import time
from typing import List, Dict


def run_command(command: str) -> tuple[bool, str]:
    """Run a shell command and return success status and output"""
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_ollama_service():
    """Check if Ollama service is running"""
    print("🔍 Checking if Ollama service is running...")

    success, output = run_command("ollama list")
    if success:
        print("✅ Ollama service is running")
        return True, output
    else:
        print("❌ Ollama service is not running")
        print("💡 Start it with: ollama serve")
        return False, output


def list_available_models() -> List[str]:
    """Get list of installed models"""
    print("\n📋 Checking installed models...")

    success, output = run_command("ollama list")
    if not success:
        print("❌ Failed to list models")
        return []

    models = []
    lines = output.strip().split('\n')

    if len(lines) <= 1:
        print("❌ No models found")
        print("💡 Install a model with: ollama pull llama3.2:3b")
        return []

    # Parse model list (skip header)
    for line in lines[1:]:
        if line.strip():
            model_name = line.split()[0]
            models.append(model_name)
            print(f"  📦 Found: {model_name}")

    return models


def test_model_response(model_name: str) -> bool:
    """Test if a model can generate responses"""
    print(f"\n🧪 Testing model: {model_name}")

    test_prompt = "Hello! Please respond with exactly: 'Test successful'"
    command = f"ollama run {model_name} \"{test_prompt}\""

    print("⏳ Sending test prompt...")
    start_time = time.time()

    success, output = run_command(command)

    end_time = time.time()
    response_time = end_time - start_time

    if success:
        print(f"✅ Model responded in {response_time:.2f} seconds")
        print(f"📝 Response: {output.strip()}")
        return True
    else:
        print(f"❌ Model failed to respond")
        print(f"📝 Error: {output}")
        return False


def test_langchain_integration():
    """Test LangChain integration with Ollama"""
    print("\n🔗 Testing LangChain integration...")

    try:
        from langchain_community.llms import Ollama
        print("✅ LangChain imports successful")

        # Test with a simple model
        models = list_available_models()
        if not models:
            print("❌ No models available for LangChain test")
            return False

        test_model = models[0]  # Use first available model
        print(f"🧪 Testing LangChain with model: {test_model}")

        llm = Ollama(model=test_model, num_predict=50)

        start_time = time.time()
        response = llm.invoke("Say 'LangChain test successful' and nothing else.")
        end_time = time.time()

        print(f"✅ LangChain test completed in {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {response}")
        return True

    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        print("💡 Install with: pip install langchain-community")
        return False
    except Exception as e:
        print(f"❌ LangChain test failed: {e}")
        return False


def performance_benchmark(model_name: str):
    """Run a simple performance benchmark"""
    print(f"\n⚡ Running performance benchmark for: {model_name}")

    test_prompts = [
        "What is 2+2?",
        "Name three colors.",
        "Complete this sentence: The weather today is",
    ]

    total_time = 0
    successful_tests = 0

    for i, prompt in enumerate(test_prompts, 1):
        print(f"  🧪 Test {i}/3: '{prompt}'")

        command = f"ollama run {model_name} \"{prompt}\""
        start_time = time.time()
        success, output = run_command(command)
        end_time = time.time()

        response_time = end_time - start_time
        total_time += response_time

        if success:
            successful_tests += 1
            print(f"    ✅ Completed in {response_time:.2f}s")
            print(f"    📝 Response: {output.strip()[:100]}...")
        else:
            print(f"    ❌ Failed in {response_time:.2f}s")

    if successful_tests > 0:
        avg_time = total_time / successful_tests
        print(f"\n📊 Benchmark Results:")
        print(f"  Success rate: {successful_tests}/{len(test_prompts)}")
        print(f"  Average response time: {avg_time:.2f} seconds")

        if avg_time < 3:
            print("  🚀 Excellent performance for voice assistant!")
        elif avg_time < 8:
            print("  👍 Good performance for voice assistant")
        else:
            print("  ⚠️  Slow performance - consider a smaller model")


def main():
    """Main test routine"""
    print("🧪 OLLAMA SETUP TEST")
    print("=" * 50)

    # Step 1: Check if Ollama is running
    service_running, _ = check_ollama_service()
    if not service_running:
        print("\n❌ CRITICAL: Ollama service is not running!")
        print("Please start Ollama first:")
        print("  - On most systems: ollama serve")
        print("  - Or start the Ollama desktop app")
        return False

    # Step 2: List available models
    models = list_available_models()
    if not models:
        print("\n❌ CRITICAL: No models installed!")
        print("Install a model with:")
        print("  ollama pull llama3.2:3b")
        return False

    # Step 3: Test basic model functionality
    print(f"\n🎯 Testing recommended model: llama3.2:3b")
    if "llama3.2:3b" in models:
        test_model = "llama3.2:3b"
    else:
        print("⚠️  Recommended model not found, using first available")
        test_model = models[0]

    model_works = test_model_response(test_model)
    if not model_works:
        print(f"\n❌ CRITICAL: Model {test_model} is not working!")
        return False

    # Step 4: Test LangChain integration
    langchain_works = test_langchain_integration()

    # Step 5: Performance benchmark
    performance_benchmark(test_model)

    # Final summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Ollama service: {'Running' if service_running else 'Not running'}")
    print(f"✅ Models installed: {len(models)} found")
    print(f"✅ Model functionality: {'Working' if model_works else 'Failed'}")
    print(f"✅ LangChain integration: {'Working' if langchain_works else 'Failed'}")

    if service_running and models and model_works and langchain_works:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Ollama setup is ready for the voice assistant!")
        print(f"💡 Recommended model for voice assistant: {test_model}")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please fix the issues above before running the voice assistant.")
        return False

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)