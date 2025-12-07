"""Interactive CLI for running the local LLM."""
from __future__ import annotations

import argparse
from typing import Dict, List

from llm.core import DEFAULT_MODEL_ID, LocalLLM

EXIT_COMMANDS = {"exit", "quit", "q", "bye"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small local LLM with GPU preference.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id to load.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"], help="Preferred compute device.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per response.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--history", type=int, default=6, help="Number of user/assistant turns to retain.")
    parser.add_argument("--system-prompt", default="You are a concise, helpful local assistant.", help="System prompt.")
    parser.add_argument("--prompt", help="Optional single prompt; if set the tool runs once and exits.")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization (requires bitsandbytes).")
    return parser.parse_args()


def trim_history(messages: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    """Keep only the latest N user/assistant turns plus the first system message."""
    if max_turns <= 0:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    allowed = max_turns * 2
    if len(convo) > allowed:
        convo = convo[-allowed:]
    return system + convo


def chat_loop(llm: LocalLLM, args: argparse.Namespace) -> None:
    history: List[Dict[str, str]] = [{"role": "system", "content": args.system_prompt}]

    if args.prompt:
        history.append({"role": "user", "content": args.prompt})
        reply = llm.generate(history, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        print(f"assistant> {reply}")
        return

    print("Local LLM ready. Type 'exit' to quit.\n")
    print(f"Loaded {llm.info()['model_id']} on {llm.info()['device']} (quantized={llm.info()['quantized']}).\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("bye!")
            break

        history.append({"role": "user", "content": user_input})
        history = trim_history(history, args.history)

        reply = llm.generate(
            history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"assistant> {reply}")
        history.append({"role": "assistant", "content": reply})


def main() -> None:
    args = parse_args()
    llm = LocalLLM(
        model_id=args.model,
        device_preference=args.device,
        use_4bit=args.quantize,
    )
    chat_loop(llm, args)


if __name__ == "__main__":
    main()
