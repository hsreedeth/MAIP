# # src/cli_rulecard.py
# import argparse, subprocess, pathlib, json, hashlib
# from src.llm_prompt import build_user_prompt
# from src.llm_validate import validate_and_write
# from src.flowchart_fallback import ascii_flow_from_rules, rulecard_from_rules
# from src.llm_fallback import generate_translation  

# def sha256_file(path: pathlib.Path, chunk=1024*1024) -> str:
#     h = hashlib.sha256()
#     with open(path, "rb") as f:
#         while True:
#             b = f.read(chunk)
#             if not b:
#                 break
#             h.update(b)
#     return h.hexdigest()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--rules", required=True)                 # tree_rules.json
#     ap.add_argument("--glossary-used", default=None)          # glossary_used.json (preferred)
#     ap.add_argument("--glossary", default=None)               # full glossary (fallback)
#     ap.add_argument("--model", required=True)                 # gguf file
#     ap.add_argument("--runner", required=True)                # llama binary (e.g., /opt/homebrew/bin/llama)
#     ap.add_argument("--out-dir", required=True)

#     ap.add_argument("--seed", type=int, default=123)
#     ap.add_argument("--temp", type=float, default=0.2)
#     ap.add_argument("--top-p", type=float, default=0.9)
#     ap.add_argument("--n-tokens", type=int, default=1024)
#     ap.add_argument("--threads", type=int, default=6)
#     ap.add_argument("--ctx", type=int, default=4096)
#     ap.add_argument("--n-gpu-layers", type=int, default=None)
#     ap.add_argument("--no-cnv", action="store_true",
#                 help="Disable chat template; treat prompt as plain text.")


#     args = ap.parse_args()

#     out_dir = pathlib.Path(args.out_dir)
#     llm_dir = out_dir / "llm"
#     tab_dir = out_dir / "tables"
#     llm_dir.mkdir(parents=True, exist_ok=True)
#     tab_dir.mkdir(parents=True, exist_ok=True)

#     # Prefer rule_ruleset.json if user points at tree_rules.json and sibling exists
#     rules_path = pathlib.Path(args.rules)
#     auto_alt = rules_path.with_name("rule_ruleset.json")
#     if rules_path.name == "tree_rules.json" and auto_alt.exists():
#         print(f"[cli_rulecard] Using {auto_alt.name} instead of {rules_path.name} (LLM-ready).")
#         rules_path = auto_alt


#     # Build prompt
#     prompt, fp = build_user_prompt(
#         rules_json_path=str(rules_path),
#         glossary_used_path=args.glossary_used,
#         glossary_fallback_path=args.glossary
#     )
#     prompt_path = llm_dir / "prompt.txt"
#     prompt_path.write_text(prompt, encoding="utf-8")

#     # Run llama.cpp via the provided runner
#     model_p = pathlib.Path(args.model)
#     model_sha = sha256_file(model_p)
#     out_txt = llm_dir / "llm_out.txt"
#     err_txt = llm_dir / "llm_err.txt"

#     cmd = [
#         args.runner,
#         "-m", str(model_p),
#         "-f", str(prompt_path),        # <- pass prompt file, not full string
#         "-n", str(args.n_tokens),
#         "-c", str(args.ctx),
#         "--seed", str(args.seed),
#         "--temp", str(args.temp),
#         "--top-p", str(args.top_p),
#         "-t", str(args.threads)
#     ]

#     if args.no_cnv:
#         cmd += ["-no-cnv"]


#     if args.n_gpu_layers is not None:
#         cmd += ["--n-gpu-layers", str(args.n_gpu_layers)]


#     proc = subprocess.run(cmd, capture_output=True, text=True)
#     out_txt.write_text(proc.stdout or "", encoding="utf-8")
#     err_txt.write_text(proc.stderr or "", encoding="utf-8")

#     if proc.returncode != 0:
#         # Still continue to write metadata and fallbacks; validator will fail gracefully
#         print(f"[llama] exit code {proc.returncode} — see {err_txt}")

#     # Validate output and write RULECARD / ASCII if valid
#     ok, report = validate_and_write(
#         rules_path,
#         out_txt,
#         out_dir
#     )

#     # Record metadata
#     meta = {
#         "rules_file": args.rules,
#         "glossary_used": args.glossary_used,
#         "glossary_fallback": args.glossary,
#         "model_path": str(model_p),
#         "model_sha256": model_sha,
#         "runner": args.runner,
#         "seed": args.seed,
#         "temp": args.temp,
#         "top_p": args.top_p,
#         "threads": args.threads,
#         "ctx": args.ctx,
#         "valid": ok,
#         "validator_report": report,
#         "stdout_path": str(out_txt),
#         "stderr_path": str(err_txt),
#     }
#     (tab_dir / "translation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

#     # # Deterministic fallbacks if invalid
#     # if not ok:
#     #     (tab_dir / "flow_ascii_fallback.txt").write_text(
#     #         ascii_flow_from_rules(args.rules), encoding="utf-8"
#     #     )
#     #     gloss_path = args.glossary_used or args.glossary
#     #     if gloss_path:
#     #         (tab_dir / "rulecard_fallback.md").write_text(
#     #             rulecard_from_rules(args.rules, gloss_path), encoding="utf-8"
#     #         )
#     #     print("LLM translation invalid per validator. Fallback artifacts written.")
#     # else:
#     #     print("LLM translation valid. Rulecard and ASCII flow saved.")
#          # Deterministic fallbacks if invalid
#     if not ok:
#         # If the LLM output is invalid, build a deterministic translation
#         # directly from the rules JSON.  This ensures that every predicate
#         # appears and that the fingerprint is echoed.  We write the
#         # translation to a separate file and run the validator again to
#         # produce the final artifacts.
#         fallback_txt = generate_translation(
#             rules_json_path=str(rules_path),
#             glossary_path=args.glossary_used or args.glossary,
#         )
#         fallback_path = llm_dir / "llm_fallback_out.txt"
#         fallback_path.write_text(fallback_txt, encoding="utf-8")
#         # Validate the fallback translation and write outputs
#         ok2, report2 = validate_and_write(
#             rules_path,
#             fallback_path,
#             out_dir,
#         )
#         # Record fallback metadata
#         meta["fallback_used"] = True
#         meta["fallback_valid"] = ok2
#         meta["fallback_validator_report"] = report2
#         (tab_dir / "translation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
#         # Create human‑readable fallback flow and rulecard as in prior
#         (tab_dir / "flow_ascii_fallback.txt").write_text(
#             ascii_flow_from_rules(args.rules), encoding="utf-8"
#         )
#         gloss_path = args.glossary_used or args.glossary
#         if gloss_path:
#             (tab_dir / "rulecard_fallback.md").write_text(
#                 rulecard_from_rules(args.rules, gloss_path), encoding="utf-8"
#             )
#         print("LLM translation invalid. Deterministic fallback translation generated and validated.")
#     else:
#         print("LLM translation valid. Rulecard and ASCII flow saved.")

# if __name__ == "__main__":
#     main()


# src/cli_rulecard.py
import argparse
import pathlib
import json
import hashlib

from src.flowchart_fallback import ascii_flow_from_rules, rulecard_from_rules


def sha256_file(path: pathlib.Path, chunk: int = 1024 * 1024) -> str:
    """Utility kept only for possible future use; currently not required."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Deterministic rulecard/flow generator.\n"
            "Reads a rule_ruleset.json and optional glossary.json, "
            "writes ASCII flow + Markdown rulecard, no LLM involved."
        )
    )
    ap.add_argument(
        "--rules",
        required=True,
        help="Path to rule_ruleset.json (or tree_rules.json; will auto-switch if sibling rule_ruleset.json exists).",
    )
    ap.add_argument(
        "--glossary-used",
        default=None,
        help="Path to glossary JSON actually used (if already chosen).",
    )
    ap.add_argument(
        "--glossary",
        default=None,
        help="Fallback glossary JSON (used if --glossary-used is not provided).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (tables/ subdir will be created if needed).",
    )

    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Prefer rule_ruleset.json if user points at tree_rules.json and sibling exists
    rules_path = pathlib.Path(args.rules)
    auto_alt = rules_path.with_name("rule_ruleset.json")
    if rules_path.name == "tree_rules.json" and auto_alt.exists():
        print(f"[cli_rulecard] Using {auto_alt.name} instead of {rules_path.name} (ruleset format).")
        rules_path = auto_alt

    # Deterministic ASCII flow
    flow_txt = ascii_flow_from_rules(str(rules_path))
    (tab_dir / "flow_ascii_fallback.txt").write_text(flow_txt, encoding="utf-8")

    # Deterministic rulecard (if glossary available)
    gloss_path = args.glossary_used or args.glossary
    rulecard_path = None
    if gloss_path:
        rulecard_md = rulecard_from_rules(str(rules_path), gloss_path)
        rulecard_path = tab_dir / "rulecard_fallback.md"
        rulecard_path.write_text(rulecard_md, encoding="utf-8")

    # Minimal meta (documents that this was fallback-only, no LLM)
    meta = {
        "rules_file": str(rules_path),
        "glossary_used": args.glossary_used,
        "glossary_fallback": args.glossary,
        "mode": "deterministic_fallback_only",
        "valid": True,
        "llm_used": False,
    }
    (tab_dir / "translation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[cli_rulecard] ASCII flow → {tab_dir / 'flow_ascii_fallback.txt'}")
    if rulecard_path:
        print(f"[cli_rulecard] Rulecard → {rulecard_path}")
    else:
        print("[cli_rulecard] No glossary provided; only ASCII flow was generated.")
    print("[cli_rulecard] Done (no LLM used).")


if __name__ == "__main__":
    main()
