"""Unit tests for LocalLLM with latency measurements."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from llm.core import DEFAULT_MODEL_ID, LocalLLM


class TestLocalLLM:
    """Test suite for LocalLLM class."""

    @pytest.fixture
    def mock_model(self):
        """Mock the model and tokenizer to avoid loading real weights."""
        with patch("llm.core.AutoTokenizer") as mock_tokenizer, \
             patch("llm.core.AutoModelForCausalLM") as mock_model, \
             patch("llm.core.pipeline") as mock_pipeline:

            # Setup tokenizer mock
            tokenizer_instance = MagicMock()
            tokenizer_instance.eos_token_id = 2
            tokenizer_instance.apply_chat_template.return_value = "Mocked prompt"
            mock_tokenizer.from_pretrained.return_value = tokenizer_instance

            # Setup model mock
            model_instance = MagicMock()
            mock_model.from_pretrained.return_value = model_instance

            # Setup pipeline mock
            pipeline_instance = MagicMock()
            pipeline_instance.return_value = [{"generated_text": "Mocked prompt\nMocked response"}]
            mock_pipeline.return_value = pipeline_instance

            yield {
                "tokenizer": mock_tokenizer,
                "model": mock_model,
                "pipeline": mock_pipeline,
                "tokenizer_instance": tokenizer_instance,
                "pipeline_instance": pipeline_instance,
            }

    def test_device_selection_cuda(self, mock_model):
        """Test CUDA device selection when available."""
        with patch("torch.cuda.is_available", return_value=True):
            llm = LocalLLM(device_preference="cuda")
            assert llm.device == "cuda"
            assert llm.dtype == torch.float16

    def test_device_selection_cpu_fallback(self, mock_model):
        """Test CPU fallback when CUDA not available."""
        with patch("torch.cuda.is_available", return_value=False):
            llm = LocalLLM(device_preference="cuda")
            assert llm.device == "cpu"
            assert llm.dtype == torch.float32

    def test_device_selection_mps(self, mock_model):
        """Test MPS device selection for Apple Silicon."""
        with patch("torch.backends.mps.is_available", return_value=True):
            llm = LocalLLM(device_preference="mps")
            assert llm.device == "mps"
            assert llm.dtype == torch.float16

    def test_model_initialization(self, mock_model):
        """Test model initialization with default settings."""
        llm = LocalLLM()

        assert llm.model_id == DEFAULT_MODEL_ID
        mock_model["tokenizer"].from_pretrained.assert_called_once()
        mock_model["model"].from_pretrained.assert_called_once()
        mock_model["pipeline"].assert_called_once()

    def test_build_prompt_string(self, mock_model):
        """Test prompt building from simple string."""
        llm = LocalLLM()
        result = llm._build_prompt("Hello")

        # Should convert string to message format and use chat template
        mock_model["tokenizer_instance"].apply_chat_template.assert_called_once()

    def test_build_prompt_messages(self, mock_model):
        """Test prompt building from message list."""
        llm = LocalLLM()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = llm._build_prompt(messages)

        mock_model["tokenizer_instance"].apply_chat_template.assert_called_once()

    def test_build_prompt_no_chat_template(self, mock_model):
        """Test prompt building when tokenizer has no chat template."""
        mock_model["tokenizer_instance"].chat_template = None

        llm = LocalLLM()
        messages = [{"role": "user", "content": "Hello"}]
        result = llm._build_prompt(messages)

        assert "User: Hello" in result
        assert "Assistant:" in result

    def test_generate_basic(self, mock_model):
        """Test basic text generation."""
        llm = LocalLLM()
        response = llm.generate("Hello")

        assert response == "Mocked response"
        mock_model["pipeline_instance"].assert_called_once()

    def test_generate_with_params(self, mock_model):
        """Test generation with custom parameters."""
        llm = LocalLLM()
        response = llm.generate(
            "Hello",
            max_new_tokens=100,
            temperature=0.9,
            top_p=0.95,
        )

        # Check that pipeline was called with correct params
        call_kwargs = mock_model["pipeline_instance"].call_args[1]
        assert call_kwargs["max_new_tokens"] == 100
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["top_p"] == 0.95

    def test_generate_latency_measurement(self, mock_model):
        """Test that generation completes within acceptable time (mocked)."""
        llm = LocalLLM()

        start = time.perf_counter()
        response = llm.generate("Hello")
        latency = time.perf_counter() - start

        # Mocked call should be very fast
        assert latency < 0.1, f"Mocked generation too slow: {latency:.3f}s"

    def test_info_method(self, mock_model):
        """Test info() returns correct runtime information."""
        llm = LocalLLM(model_id="test-model", device_preference="cpu")
        info = llm.info()

        assert info["model_id"] == "test-model"
        assert info["device"] == "cpu"
        assert "dtype" in info
        assert "quantized" in info

    def test_4bit_quantization_config(self, mock_model):
        """Test 4-bit quantization configuration."""
        with patch("llm.core.BitsAndBytesConfig") as mock_bnb:
            llm = LocalLLM(use_4bit=True)

            # Should create BitsAndBytesConfig
            mock_bnb.assert_called_once()
            assert llm.info()["quantized"] is True

    def test_4bit_without_bitsandbytes(self, mock_model):
        """Test that 4-bit mode raises error when bitsandbytes not available."""
        with patch("llm.core.BitsAndBytesConfig", None):
            with pytest.raises(ImportError, match="bitsandbytes is required"):
                LocalLLM(use_4bit=True)

    def test_temperature_zero_disables_sampling(self, mock_model):
        """Test that temperature=0 disables sampling."""
        llm = LocalLLM()
        llm.generate("Hello", temperature=0.0)

        call_kwargs = mock_model["pipeline_instance"].call_args[1]
        assert call_kwargs["do_sample"] is False

    def test_temperature_nonzero_enables_sampling(self, mock_model):
        """Test that temperature>0 enables sampling."""
        llm = LocalLLM()
        llm.generate("Hello", temperature=0.7)

        call_kwargs = mock_model["pipeline_instance"].call_args[1]
        assert call_kwargs["do_sample"] is True
