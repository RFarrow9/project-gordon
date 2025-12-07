"""Tests for run_local_llm CLI module."""
from __future__ import annotations

import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from llm.cli import EXIT_COMMANDS, chat_loop, parse_args, trim_history


class TestParseArgs:
    """Test argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        with patch("sys.argv", ["run_local_llm.py"]):
            args = parse_args()

            assert args.model == "microsoft/Phi-3-mini-4k-instruct"
            assert args.device == "cuda"
            assert args.max_new_tokens == 256
            assert args.temperature == 0.7
            assert args.top_p == 0.9
            assert args.history == 6
            assert args.system_prompt == "You are a concise, helpful local assistant."
            assert args.prompt is None
            assert args.quantize is False

    def test_custom_args(self):
        """Test custom argument values."""
        with patch("sys.argv", [
            "run_local_llm.py",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--device", "cpu",
            "--max-new-tokens", "100",
            "--temperature", "0.5",
            "--top-p", "0.8",
            "--history", "4",
            "--system-prompt", "Custom prompt",
            "--prompt", "Test prompt",
            "--quantize",
        ]):
            args = parse_args()

            assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            assert args.device == "cpu"
            assert args.max_new_tokens == 100
            assert args.temperature == 0.5
            assert args.top_p == 0.8
            assert args.history == 4
            assert args.system_prompt == "Custom prompt"
            assert args.prompt == "Test prompt"
            assert args.quantize is True


class TestTrimHistory:
    """Test conversation history trimming."""

    def test_trim_history_basic(self):
        """Test basic history trimming."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "3"},
        ]

        result = trim_history(messages, max_turns=2)

        # Should keep system + last 2 turns (4 messages)
        assert len(result) == 5  # 1 system + 2*2 turns
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "2"  # Last 2 user/assistant pairs

    def test_trim_history_keeps_system(self):
        """Test that system message is always kept."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "1"},
        ]

        result = trim_history(messages, max_turns=1)

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System"

    def test_trim_history_zero_turns(self):
        """Test max_turns=0 returns original."""
        messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "1"},
        ]

        result = trim_history(messages, max_turns=0)
        assert result == messages

    def test_trim_history_more_turns_than_messages(self):
        """Test when max_turns exceeds message count."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "1"},
        ]

        result = trim_history(messages, max_turns=10)
        assert len(result) == 3


class TestChatLoop:
    """Test chat loop functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.generate.return_value = "Mocked response"
        llm.info.return_value = {
            "model_id": "test-model",
            "device": "cpu",
            "quantized": False,
        }
        return llm

    @pytest.fixture
    def default_args(self):
        """Create default args namespace."""
        args = MagicMock()
        args.system_prompt = "Test system prompt"
        args.prompt = None
        args.max_new_tokens = 256
        args.temperature = 0.7
        args.top_p = 0.9
        args.history = 6
        return args

    def test_chat_loop_single_prompt(self, mock_llm, default_args):
        """Test single prompt mode (non-interactive)."""
        default_args.prompt = "Hello"

        with patch("builtins.print") as mock_print:
            chat_loop(mock_llm, default_args)

            mock_llm.generate.assert_called_once()
            mock_print.assert_called_with("assistant> Mocked response")

    def test_chat_loop_single_prompt_latency(self, mock_llm, default_args):
        """Test latency of single prompt execution."""
        default_args.prompt = "Hello"

        start = time.perf_counter()
        chat_loop(mock_llm, default_args)
        latency = time.perf_counter() - start

        # Should be very fast with mocked LLM
        assert latency < 0.1, f"Single prompt too slow: {latency:.3f}s"

    def test_chat_loop_interactive_exit(self, mock_llm, default_args):
        """Test interactive mode with exit command."""
        with patch("builtins.input", side_effect=["exit"]):
            with patch("builtins.print"):
                chat_loop(mock_llm, default_args)

        # Should not call generate for exit command
        mock_llm.generate.assert_not_called()

    def test_chat_loop_interactive_conversation(self, mock_llm, default_args):
        """Test interactive conversation flow."""
        user_inputs = ["Hello", "How are you?", "quit"]

        with patch("builtins.input", side_effect=user_inputs):
            with patch("builtins.print"):
                chat_loop(mock_llm, default_args)

        # Should call generate twice (not for 'quit')
        assert mock_llm.generate.call_count == 2

    def test_chat_loop_empty_input(self, mock_llm, default_args):
        """Test that empty input is skipped."""
        user_inputs = ["", "  ", "exit"]

        with patch("builtins.input", side_effect=user_inputs):
            with patch("builtins.print"):
                chat_loop(mock_llm, default_args)

        # Should not call generate for empty inputs
        mock_llm.generate.assert_not_called()

    def test_chat_loop_eof(self, mock_llm, default_args):
        """Test handling of EOF (Ctrl+D)."""
        with patch("builtins.input", side_effect=EOFError):
            with patch("builtins.print") as mock_print:
                chat_loop(mock_llm, default_args)

        # Should print bye message
        assert any("bye" in str(call).lower() for call in mock_print.call_args_list)

    def test_chat_loop_keyboard_interrupt(self, mock_llm, default_args):
        """Test handling of KeyboardInterrupt (Ctrl+C)."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("builtins.print") as mock_print:
                chat_loop(mock_llm, default_args)

        # Should print bye message
        assert any("bye" in str(call).lower() for call in mock_print.call_args_list)

    def test_exit_commands(self):
        """Test that all exit commands are recognized."""
        expected_exits = {"exit", "quit", "q", "bye"}
        assert EXIT_COMMANDS == expected_exits

    def test_chat_loop_history_trimming(self, mock_llm, default_args):
        """Test that history is trimmed correctly during conversation."""
        default_args.history = 2  # Only keep 2 turns
        user_inputs = ["msg1", "msg2", "msg3", "exit"]

        with patch("builtins.input", side_effect=user_inputs):
            with patch("builtins.print"):
                chat_loop(mock_llm, default_args)

        # Check that generate was called with trimmed history
        assert mock_llm.generate.call_count == 3

        # Last call should have limited history
        last_call_history = mock_llm.generate.call_args_list[-1][0][0]
        # System + 2 turns * 2 messages = 5 messages max (but after response, history grows to 6)
        assert len(last_call_history) <= 6

    def test_chat_loop_passes_generation_params(self, mock_llm, default_args):
        """Test that generation parameters are passed correctly."""
        default_args.prompt = "Test"
        default_args.max_new_tokens = 128
        default_args.temperature = 0.5
        default_args.top_p = 0.8

        chat_loop(mock_llm, default_args)

        # Check kwargs passed to generate
        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 128
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8
