#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This was an experiment to generate rulecards using LLMs.
# It is not part of the main pipeline.  It did not produce satisfactory
# results compared to hand-crafted rulecards with RAG STrat, so it is no longer used.

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_ID  = "meta-llama/Meta-Llama-3-8B-Instruct"
# CACHE_DIR = Path("models/Meta-Llama-3-8B-Instruct")

MODEL_ID  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CACHE_DIR = Path("models/TinyLlama-1.1B-Chat-v1.0")

PROMPT_ROOT = Path("reports/rulecards_rag")
INPUT_DIRS = {
    "high": PROMPT_ROOT / "high",
    "mid":  PROMPT_ROOT / "mid",
    "low":  PROMPT_ROOT / "low",
}
OUTPUT_DIRS = {
    "high": PROMPT_ROOT / "high_out",
    "mid":  PROMPT_ROOT / "mid_out",
    "low":  PROMPT_ROOT / "low_out",
}


# def load_model():
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     dtype = torch.float16 if device == "mps" else torch.float32

#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_ID,
#         cache_dir=str(CACHE_DIR),
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         cache_dir=str(CACHE_DIR),
#         torch_dtype=dtype,
#         device_map=None,
#     ).to(device)

#     return tokenizer, model, device

def load_model():
    device = "cpu"  # force CPU, avoid MPS issues

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,   # <- key: don’t go online
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,   # <- key: don’t go online
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )

    return tokenizer, model, device



def generate_chat(tokenizer, model, device, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Stay safely under TinyLlama's 2048 context
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1900,   # prompt tokens
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,                # shorter, avoids rambling
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt and decode only new tokens
    generated_ids = outputs[0, inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()



def process_dir(tokenizer, model, device, prompt_dir: Path, out_dir: Path):
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

        print(f"[LLM] Generating rulecard for {label} from {p}...")
        content = generate_chat(tokenizer, model, device, system_prompt, user_prompt)

        out_file = out_dir / f"rulecard_{label}.md"
        out_file.write_text(content, encoding="utf-8")
        print(f"[LLM] Wrote {out_file}")



def main():
    tokenizer, model, device = load_model()
    print(f"[LLM] Loaded {MODEL_ID} on {device}")

    for key, prompt_dir in INPUT_DIRS.items():
        if not prompt_dir.exists():
            print(f"[WARN] Prompt directory missing: {prompt_dir} (skipping)")
            continue
        process_dir(tokenizer, model, device, prompt_dir, OUTPUT_DIRS[key])

    print("[LLM] All done.")


if __name__ == "__main__":
    main()
