#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Call GPT-4.1-mini on the RAG prompts and produce canonical rulecards.

Usage example:

  python -m src.run_rulecards_remote_canonical \
    --in-root reports/rulecards_rag \
    --out-root reports/rulecards_final \
    --model gpt-4.1-mini \
    --max-tokens 1200
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI

client = OpenAI()


def build_system_prompt(label: str) -> str:
    """
    Canonical system prompt for all phenotypes.
    Enforces headings and bans JSON / style-guide chatter.
    """
    return f"""You are a deterministic translator for ICU phenotyping rules.

Your job is to rewrite JSON-derived decision rules into clinician-facing text
without changing their logical content.

Requirements:
- Preserve every condition (variable, operator, threshold) in an equivalent IF/THEN rule.
- Do NOT introduce new variables, thresholds, or phenotypes.
- Do NOT give management or treatment advice.
- Use only the information contained in the user message (phenotype summary,
  variable dictionary snippets, and rules).

Output format (strict):
1. First line: 'Phenotype {label} – <short clinical title>'
2. Then a blank line.
3. A section starting exactly with 'Key idea:' on its own line,
   followed by 1–2 sentences.
4. A section starting exactly with 'Rulecard:' on its own line,
   followed by a bullet list of IF/THEN rules.
5. A section starting exactly with 'ASCII flowchart:' on its own line,
   followed by a simple ASCII tree representation of the same rules.

Do NOT:
- Include raw JSON.
- Include fenced code blocks of any kind.
- Mention the style guide or quote its text.
- Repeat the instructions back to the user.

If space is limited, prioritise a complete Rulecard and ASCII flowchart
over elaboration in the Key idea.
"""


def call_gpt(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            # We keep all the RAG content (style guide excerpt, phenotype summary,
            # variable dictionary, JSON rules) in the user message:
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def process_dir(model: str, in_dir: Path, out_dir: Path, max_tokens: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(in_dir.glob("prompt_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        label = payload["label"]
        user_prompt = payload["user_prompt"]

        system_prompt = build_system_prompt(label)

        print(f"[REMOTE] Generating rulecard for {label} from {p} ...")
        text = call_gpt(model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens)

        out_file = out_dir / f"rulecard_{label}.md"
        out_file.write_text(text, encoding="utf-8")
        print(f"[REMOTE] Wrote {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=str, required=True,
                    help="Root directory with RAG prompts (e.g. reports/rulecards_rag)")
    ap.add_argument("--out-root", type=str, required=True,
                    help="Root directory for final rulecards (e.g. reports/rulecards_final)")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--max-tokens", type=int, default=1200)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    strata = {
        "high": ("high", "High_MM"),
        "mid":  ("mid",  "Mid_MM"),
        "low":  ("low",  "Low_MM"),
    }

    print(f"[REMOTE] Using model={args.model}, max_tokens={args.max_tokens}")

    for key, _ in strata.values():
        in_dir = in_root / key
        out_dir = out_root / key
        if not in_dir.exists():
            print(f"[WARN] Missing input dir {in_dir}, skipping.")
            continue
        process_dir(args.model, in_dir, out_dir, args.max_tokens)

    print("[REMOTE] All done.")


if __name__ == "__main__":
    main()
