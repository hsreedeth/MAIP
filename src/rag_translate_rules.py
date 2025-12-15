#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build RAG-ready prompts for translating surrogate tree rules into
clinician-facing rulecards and ASCII flowcharts.

For each phenotype label in rule_ruleset.json, this script:

- Collects all rules whose outcome == label.
- Identifies the variables used in those rules.
- Extracts the corresponding entries from the variable dictionary.
- Loads the phenotype summary markdown and the style guide.
- Assembles a system + user prompt skeleton.

Outputs one JSON file per phenotype, e.g.:

  <out-dir>/prompt_High_MM_0.json

with keys:
  - "stratum"
  - "label"
  - "system_prompt"
  - "user_prompt"
  - "rules"
  - "variables_used"
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_ruleset(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Expect keys: stratum, feature_names, class_names, rules, etc.
    if "rules" not in data:
        raise ValueError(f"{path} does not look like a rule_ruleset.json (missing 'rules').")
    return data


def load_style_guide(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_variable_dict(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_phenotype_summary(phenotype_dir: Path, label: str) -> str:
    """
    Expects files like phenotype_High_MM_0.md in phenotype_dir.
    If missing, returns an empty string so the caller can still proceed.
    """
    fname = f"phenotype_{label}.md"
    fpath = phenotype_dir / fname
    if fpath.exists():
        return fpath.read_text(encoding="utf-8")
    else:
        return ""


def extract_rules_by_label(rules: list[dict]) -> dict[str, list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rules:
        lbl = r.get("outcome")
        if lbl is None:
            continue
        by_label[str(lbl)].append(r)
    return by_label


def variables_used_in_rules(rules: list[dict]) -> set[str]:
    vars_used: set[str] = set()
    for r in rules:
        for cond in r.get("path", []):
            feat = cond.get("feature")
            if feat is not None:
                vars_used.add(str(feat))
    return vars_used


def format_variable_snippets(var_dict: dict, vars_used: set[str]) -> str:
    """
    Build a small markdown section summarising only the variables actually
    used in the rules for this phenotype.
    """
    lines: list[str] = []
    lines.append("## Variables used in rules\n")
    for v in sorted(vars_used):
        if v not in var_dict:
            # Fall back to a minimal line if not in dictionary
            lines.append(f"- `{v}`: no detailed entry in the variable dictionary.\n")
            continue

        entry = var_dict[v]
        # entry may be a string or a dict
        if isinstance(entry, str):
            lines.append(f"- `{v}`: {entry}\n")
        elif isinstance(entry, dict):
            disp = entry.get("display_name", v)
            units = entry.get("units")
            notes = entry.get("notes")
            scale = entry.get("scale")
            vm = entry.get("value_map")

            desc_parts = [disp]
            if units:
                desc_parts.append(f"({units})")
            desc = " ".join(desc_parts)

            # Add notes / scale info if present
            extra_bits = []
            if notes:
                extra_bits.append(notes)
            if scale:
                scale_str = ", ".join(f"{k}: {v2}" for k, v2 in scale.items())
                extra_bits.append(f"Scale: {scale_str}")
            if vm:
                vm_str = ", ".join(f"{k}: {v2}" for k, v2 in vm.items())
                extra_bits.append(f"Value map: {vm_str}")

            if extra_bits:
                lines.append(f"- `{v}`: {desc}. " + " ".join(extra_bits) + "\n")
            else:
                lines.append(f"- `{v}`: {desc}.\n")
        else:
            # Unexpected type
            lines.append(f"- `{v}`: (unrecognised dictionary entry type)\n")

    return "".join(lines)


def build_system_prompt() -> str:
    """
    A compact but strict system prompt that pairs with the style guide.
    """
    return (
        "You are a deterministic translator for ICU phenotyping rules. "
        "Your job is to rewrite JSON decision rules into clinician-facing text "
        "without changing their logical content. "
        "You must preserve every condition (variable, operator, threshold) in "
        "an equivalent IF/THEN form and obey the attached style guide. "
        "Do not add any new criteria or thresholds, and do not give treatment advice."
    )


def build_user_prompt(
    stratum: str | None,
    label: str,
    style_guide_text: str,
    phenotype_summary: str,
    variable_markdown: str,
    rules_for_label: list[dict],
) -> str:
    """
    Compose a single user prompt with all context in markdown-style text.
    The caller can send this as the user message, with the system prompt set separately.
    """
    rules_json = json.dumps(rules_for_label, indent=2, ensure_ascii=False)

    parts: list[str] = []

    header = f"Phenotype {label}"
    if stratum:
        header += f" (stratum: {stratum})"
    parts.append(header + "\n")
    parts.append("\nYou are given:\n")
    parts.append("- A style guide for phrasing rulecards and flowcharts.\n")
    parts.append("- A short phenotype summary for this label.\n")
    parts.append("- A subset of the variable dictionary for variables used in the rules.\n")
    parts.append("- The JSON-encoded rules from a surrogate decision tree.\n\n")

    parts.append("Your task is to produce three sections using ONLY this information:\n")
    parts.append("1. `Key idea` – 1–2 sentences summarising the typical patient in this phenotype.\n")
    parts.append("2. `Rulecard` – a bullet list of IF/THEN rules that exactly preserve the JSON logic.\n")
    parts.append("3. `ASCII flowchart` – a simple text flowchart representing the same branches.\n\n")

    parts.append("Follow all constraints in the style guide. "
                 "Do not introduce any new thresholds or variables, "
                 "and do not give management or treatment advice.\n\n")

    parts.append("---\n\n")
    parts.append("## Style guide\n\n")
    parts.append(style_guide_text.strip() + "\n\n")

    parts.append("---\n\n")
    parts.append("## Phenotype summary\n\n")
    if phenotype_summary.strip():
        parts.append(phenotype_summary.strip() + "\n\n")
    else:
        parts.append("_No phenotype summary file was found; you must rely on the rules and variable dictionary._\n\n")

    parts.append("---\n\n")
    parts.append(variable_markdown.strip() + "\n\n")

    parts.append("---\n\n")
    parts.append("## JSON rules for this phenotype\n\n")
    parts.append("```json\n")
    parts.append(rules_json)
    parts.append("\n```\n")

    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description="Build RAG-ready prompts for translating surrogate tree rules into rulecards."
    )
    ap.add_argument(
        "--rules-json",
        required=True,
        help="Path to rule_ruleset.json from surrogate_tree.py"
    )
    ap.add_argument(
        "--variable-dict",
        default="rag_corpus/variable_dictionary.json",
        help="Path to variable dictionary JSON (default: rag_corpus/variable_dictionary.json)"
    )
    ap.add_argument(
        "--style-guide",
        default="rag_corpus/style_guide.md",
        help="Path to style guide markdown (default: rag_corpus/style_guide.md)"
    )
    ap.add_argument(
        "--phenotype-dir",
        default="rag_corpus",
        help="Directory containing phenotype_*.md summaries (default: rag_corpus)"
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write prompt JSON files into"
    )
    ap.add_argument(
        "--label",
        default=None,
        help="Optional: restrict to a single phenotype label (e.g. High_MM_0). "
             "If omitted, prompts are generated for all labels in rules_json."
    )
    args = ap.parse_args()

    rules_path = Path(args.rules_json)
    var_path = Path(args.variable_dict)
    style_path = Path(args.style_guide)
    pheno_dir = Path(args.phenotype_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    ruleset = load_ruleset(rules_path)
    stratum = ruleset.get("stratum")
    all_rules = ruleset["rules"]

    style_guide_text = load_style_guide(style_path)
    var_dict = load_variable_dict(var_path)

    rules_by_label = extract_rules_by_label(all_rules)

    # If user requested a single label, filter to that
    labels_to_do = [args.label] if args.label else sorted(rules_by_label.keys())

    system_prompt = build_system_prompt()

    for label in labels_to_do:
        if label not in rules_by_label:
            raise ValueError(f"Label '{label}' not found in ruleset outcomes.")

        rules_for_label = rules_by_label[label]
        vars_used = variables_used_in_rules(rules_for_label)
        var_md = format_variable_snippets(var_dict, vars_used)
        pheno_summary = load_phenotype_summary(pheno_dir, label)

        user_prompt = build_user_prompt(
            stratum=stratum,
            label=label,
            style_guide_text=style_guide_text,
            phenotype_summary=pheno_summary,
            variable_markdown=var_md,
            rules_for_label=rules_for_label,
        )

        payload = {
            "stratum": stratum,
            "label": label,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "variables_used": sorted(vars_used),
            "rules": rules_for_label,
        }

        out_file = out_dir / f"prompt_{label}.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[RAG] Wrote prompt for {label} -> {out_file}")

    print(f"[RAG] Done. Prompts written to {out_dir}")


if __name__ == "__main__":
    main()
