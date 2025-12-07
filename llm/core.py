"""GPU-first local LLM loader for Project Gordon."""
from __future__ import annotations

import logging
from typing import Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # bitsandbytes is optional
    BitsAndBytesConfig = None  # type: ignore

DEFAULT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SUPPORTED_SMALL_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B-Instruct",
]

logger = logging.getLogger(__name__)


class LocalLLM:
    """Simple local text-generation helper with GPU preference."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_preference: str = "cuda",
        use_4bit: bool = False,
    ):
        self.model_id = model_id
        self.device = self._select_device(device_preference)
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.quantization_config = self._maybe_build_4bit(use_4bit)

        logger.info("Loading model %s on %s", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map="auto" if self.device != "cpu" else None,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
        )
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if self.device != "cpu" else None,
            model_kwargs={"torch_dtype": self.dtype},
        )

    def _select_device(self, device_preference: str) -> str:
        choice = device_preference.lower()
        if choice == "cuda" and torch.cuda.is_available():
            return "cuda"
        if choice == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _maybe_build_4bit(self, use_4bit: bool):
        if not use_4bit:
            return None
        if BitsAndBytesConfig is None:
            raise ImportError(
                "bitsandbytes is required for --quantize/4-bit mode. "
                "Install with: pip install bitsandbytes --extra-index-url https://pypi.nvidia.com"
            )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    def _build_prompt(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_lines = []
        for message in messages:
            role = message.get("role", "user").title()
            prompt_lines.append(f"{role}: {message.get('content', '')}")
        prompt_lines.append("Assistant:")
        return "\n".join(prompt_lines)

    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        prompt = self._build_prompt(messages)
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        return generated[len(prompt):].strip()

    def info(self) -> dict:
        """Return basic runtime info for debugging."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "quantized": bool(self.quantization_config),
        }
