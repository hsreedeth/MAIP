#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from openai import OpenAI

PROMPT_ROOT = Path("reports/rulecards_rag")
INPUT_DIRS = {
    "high": PROMPT_ROOT / "high",
    "mid":  PROMPT_ROOT / "mid",
    "low":  PROMPT_ROOT / "low",
}
OUTPUT_DIRS = {
    "high": PROMPT_ROOT / "high_out_remote",
    "mid":  PROMPT_ROOT / "mid_out_remote",
    "low":  PROMPT_ROOT / "low_out_remote",
}

MODEL_NAME = "gpt-4.1-mini"  # or any chat-capable model

client = OpenAI()  # reads OPENAI_API_KEY from env


def generate_chat_remote(system_prompt: str, user_prompt: str) -> str:
    """
    Send the RAG-assembled prompts to a remote chat model and
    return the assistant's markdown reply.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.02,
        max_tokens=1400,
    )

    # Single assistant message with markdown content
    return response.choices[0].message.content.strip()


def process_dir(prompt_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(prompt_dir.glob("prompt_*.json")):
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            print(f"[WARN] Empty prompt file: {p} (skipping)")
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[WARN] Could not parse JSON in {p}: {e} (skipping)")
            continue

        system_prompt = payload["system_prompt"]
        user_prompt   = payload["user_prompt"]
        label         = payload["label"]

        print(f"[LLM-REMOTE] Generating rulecard for {label} from {p}...")
        content = generate_chat_remote(system_prompt, user_prompt)

        out_file = out_dir / f"rulecard_{label}.md"
        out_file.write_text(content, encoding="utf-8")
        print(f"[LLM-REMOTE] Wrote {out_file}")


def main():
    print(f"[LLM-REMOTE] Using model: {MODEL_NAME}")

    for key, prompt_dir in INPUT_DIRS.items():
        if not prompt_dir.exists():
            print(f"[WARN] Prompt directory missing: {prompt_dir} (skipping)")
            continue
        process_dir(prompt_dir, OUTPUT_DIRS[key])

    print("[LLM-REMOTE] All done.")


if __name__ == "__main__":
    main()
