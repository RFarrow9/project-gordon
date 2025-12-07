# Local LLM Module

GPU-first local LLM integration for Project Gordon using Hugging Face models.

## Overview

This module provides a simple interface for running small language models locally with GPU acceleration. It's designed for low-latency voice assistant use cases.

## Default Models

- **microsoft/Phi-3-mini-4k-instruct** (default, ~3.8B, solid quality/speed balance)
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (very small, fastest fallback)
- **meta-llama/Llama-3.2-1B-Instruct** (lightweight, good reasoning for size)

## Installation

### Base Requirements

```bash
pip install -r requirements.txt
```

### GPU Support

Install a GPU-matched PyTorch wheel (CUDA example):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Pick the wheel that matches your CUDA version.

### Optional: 4-bit Quantization

For reduced VRAM usage (Linux/macOS CUDA only):

```bash
pip install bitsandbytes --extra-index-url https://pypi.nvidia.com
```

## Quickstart

### Single Prompt

```bash
python -m llm.cli --device cuda --prompt "Say hello from a local GPU LLM"
```

### Interactive Chat

```bash
python -m llm.cli --device cuda --max-new-tokens 128
```

### Quantized 4-bit (saves VRAM)

Requires bitsandbytes:

```bash
python -m llm.cli --device cuda --quantize --model microsoft/Phi-3-mini-4k-instruct
```

### CPU-only Fallback

```bash
python -m llm.cli --device cpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Programmatic Usage

```python
from llm import LocalLLM

# Initialize
llm = LocalLLM(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    device_preference="cuda",
    use_4bit=False
)

# Generate response
response = llm.generate(
    "What is the capital of France?",
    max_new_tokens=50,
    temperature=0.7
)

print(response)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = llm.generate(messages)
messages.append({"role": "assistant", "content": response})

# Next turn
messages.append({"role": "user", "content": "Show me an example."})
response = llm.generate(messages)
```

## Configuration

### Device Selection

The module automatically selects the best available device:
1. CUDA (NVIDIA GPU) - if available and requested
2. MPS (Apple Silicon) - if available and requested
3. CPU - fallback

Override with `--device` flag or `device_preference` parameter.

### Performance Tuning

**For faster responses:**
- Use smaller models: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Reduce `max_new_tokens` (30-50 for voice)
- Enable quantization: `use_4bit=True`

**For better quality:**
- Use larger models: `microsoft/Phi-3-mini-4k-instruct`
- Increase `temperature` (0.8-1.0) for creativity
- Use full precision (don't quantize)

## Notes

- First run downloads model weights to Hugging Face cache (`~/.cache/huggingface`)
- Set `HF_HOME` environment variable to change cache location
- `--history` parameter trims conversation context to control memory usage
- For best GPU performance, match PyTorch build to your CUDA version

---

# Testing

Comprehensive test suite with latency measurements for voice assistant optimization.

## Test Structure

```
tests/
├── test_core.py         # Unit tests for LocalLLM class (mocked, fast)
├── test_cli.py          # Unit tests for CLI interface (mocked, fast)
├── test_performance.py  # Performance benchmarks (real models, slower)
└── conftest.py          # Shared pytest fixtures
```

## Installation

Install test dependencies:

```bash
pip install pytest pytest-mock
```

## Running Tests

### Quick Unit Tests (mocked, no model loading)

```bash
pytest llm/tests/test_core.py llm/tests/test_cli.py -v
```

### Performance Benchmarks (loads real models)

```bash
pytest llm/tests/test_performance.py -v -s
```

The `-s` flag shows latency output in real-time.

### Run All Tests

```bash
pytest llm/ -v
```

### Specific Benchmark Tests

```bash
# Only fast benchmarks (uses TinyLlama)
pytest llm/tests/test_performance.py -m "benchmark and not slow" -v -s

# All benchmarks including slow ones (Phi-3)
pytest llm/tests/test_performance.py -m benchmark -v -s

# Voice assistant simulation
pytest llm/tests/test_performance.py::test_voice_assistant_simulation -v -s

# Short response latency (critical for voice)
pytest llm/tests/test_performance.py::test_short_response_latency -v -s
```

## Performance Benchmarks

The performance tests measure:

- **Cold start latency** - First generation (includes warmup)
- **Warm latency** - Subsequent generations (typical use case)
- **Short responses** - Critical for voice assistant (target: <0.5s)
- **Batch consistency** - Latency variance over multiple requests
- **Token scaling** - How latency increases with response length
- **Voice assistant simulation** - Realistic conversation flow

### Latency Thresholds

Performance grades based on response time:

- **Instant**: < 0.5s (excellent for voice assistant)
- **Acceptable**: < 2.0s (good for interactive use)
- **Slow**: < 5.0s (starting to feel slow)
- **Very Slow**: > 5.0s (not suitable for real-time)

### Performance Tips

If latency tests show slow performance:

1. **Use a smaller model:**
   ```python
   LocalLLM(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
   ```

2. **Enable 4-bit quantization:**
   ```python
   LocalLLM(use_4bit=True)
   ```

3. **Reduce max_new_tokens:**
   ```python
   llm.generate(prompt, max_new_tokens=30)
   ```

4. **Use GPU (CUDA or MPS):**
   ```python
   LocalLLM(device_preference="cuda")
   ```

## Expected Performance

On a typical GPU setup:

- **TinyLlama (1.1B)**: ~0.5-1.5s per response
- **Phi-3-mini (3.8B)**: ~1.5-3.0s per response
- **With quantization**: 20-30% faster, uses less VRAM

CPU-only will be 5-10x slower.

## Example Output

```
📊 Short Response Latency Test
======================================================================
1. Latency: 0.423s | Tokens: 3 | Speed: 7.1 tok/s | Prompt: 'Say yes or no: Is the sky blue?...'
2. Latency: 0.381s | Tokens: 1 | Speed: 2.6 tok/s | Prompt: 'What is 5+3?...'
3. Latency: 0.456s | Tokens: 2 | Speed: 4.4 tok/s | Prompt: 'Name one color....'

──────────────────────────────────────────────────────────────────────
Average Latency: 0.420s
Average Speed: 4.7 tokens/sec
✅ Performance: EXCELLENT - Instant responses!
======================================================================
```

## Interpreting Results

- **Latency**: Time from prompt to complete response
- **Tokens**: Number of tokens in the response
- **Speed**: Tokens generated per second
- **Performance grade**: Based on latency thresholds

If you see "⚠️ SLOW" warnings, your model may not be suitable for real-time voice interaction.
