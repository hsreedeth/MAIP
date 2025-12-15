#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate LLM-generated rulecards against surrogate tree rules.

Implements:
  1) Rule coverage / consistency checks (4.1)
  2) Variable-dictionary alignment checks (4.2)
  3) Optional synthetic profile verification (4.3)

Example usage (per stratum):

  python -m src.rulecard_validate \
    --stratum High_MM \
    --rules-json reports/surrogate_high/tables/rule_ruleset.json \
    --rulecards-dir reports/rulecards_final/high \
    --var-dict rag_corpus/variable_dictionary.json \
    --out-dir reports/rulecards_final/high

# MID_MM
python -m src.rulecard_validate \
--stratum Mid_MM \
--rules-json reports/surrogate_mid/tables/rule_ruleset.json \
--rulecards-dir reports/rulecards_final/mid \
--var-dict rag_corpus/variable_dictionary.json \
--out-dir reports/rulecards_final/mid \
--model-bundle reports/surrogate_mid/models/surrogate_tree.joblib

# LOW_MM
python -m src.rulecard_validate \
--stratum Low_MM \
--rules-json reports/surrogate_low/tables/rule_ruleset.json \
--rulecards-dir reports/rulecards_final/low \
--var-dict rag_corpus/variable_dictionary.json \
--out-dir reports/rulecards_final/low \
--model-bundle reports/surrogate_low/models/surrogate_tree.joblib

  # with optional synthetic checks:
  python -m src.rulecard_validate \
    --stratum High_MM \
    --rules-json reports/surrogate_high/tables/rule_ruleset.json \
    --rulecards-dir reports/rulecards_final/high \
    --var-dict rag_corpus/variable_dictionary.json \
    --model-bundle reports/surrogate_high/models/surrogate_tree.joblib \
    --out-dir reports/rulecards_final/high
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from joblib import load as joblib_load
except ImportError:
    joblib_load = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_ruleset(rules_json: Path):
    payload = json.loads(rules_json.read_text(encoding="utf-8"))
    rules = payload["rules"]
    # collect labels and per-label rules
    labels = sorted({r["outcome"] for r in rules})
    rules_by_label = {lab: [] for lab in labels}
    for r in rules:
        rules_by_label[r["outcome"]].append(r)
    return payload, labels, rules_by_label


def parse_rulecard_md(path: Path):
    """Return full text + list of lines that look like rulecard IF-lines."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if_lines = [ln for ln in lines if ln.lstrip().startswith("- IF")]
    return text, if_lines


def threshold_strings(val: float):
    """
    Generate a set of plausible string renderings of a numeric threshold
    to give the regex a fighting chance even if the LLM rounded slightly.
    """
    strs = set()
    # raw repr
    strs.add(str(val))
    # fixed decimals
    strs.add(f"{val:.4f}")
    strs.add(f"{val:.3f}")
    strs.add(f"{val:.2f}")
    # if it's basically an int, include that too
    if abs(val - round(val)) < 1e-6:
        strs.add(str(int(round(val))))
    # also drop trailing zeros / trailing dots versions
    clean = {s.rstrip("0").rstrip(".") for s in strs}
    return clean


def condition_covered(cond, if_lines):
    """
    Very simple sanity check:
    - JSON feature name appears as a word in some '- IF' line
    - AND one of the threshold string forms appears in the same line.
    """
    feat = cond["feature"]
    thr  = cond["threshold"]
    thr_strs = threshold_strings(thr)

    feat_pattern = re.compile(rf"\b{re.escape(feat)}\b")

    for line in if_lines:
        if not feat_pattern.search(line):
            continue
        if any(ts in line for ts in thr_strs):
            return True
    return False


def feature_name_alignment(feature: str, text: str):
    """
    Check that the canonical feature name appears in the rulecard text
    and (ideally) appears in parentheses somewhere, e.g. '... (aps)'.
    """
    # basic presence
    present = re.search(rf"\b{re.escape(feature)}\b", text) is not None

    # parenthetical or code formatting like (aps) or `aps`
    parenthetical = f"({feature})" in text or f"`{feature}`" in text

    return present, parenthetical


def make_synthetic_point(path_conditions, feature_names):
    """
    Build a synthetic feature vector that satisfies all inequalities
    in a JSON rule path. This is deliberately simple: it picks a
    value slightly inside each inequality and leaves others at 0.
    """
    x = {f: 0.0 for f in feature_names}

    for cond in path_conditions:
        f   = cond["feature"]
        op  = cond["op"]
        thr = float(cond["threshold"])
        # choose a small delta relative to magnitude
        delta = 0.1 * (abs(thr) if abs(thr) > 1e-3 else 1.0)

        if op == "<=":
            x[f] = thr - delta
        elif op == "<":
            x[f] = thr - delta
        elif op == ">=":
            x[f] = thr + delta
        elif op == ">":
            x[f] = thr + delta
        else:
            # unknown operator, just set to threshold
            x[f] = thr

    return x


# ---------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------

def validate_rulecards(stratum, rules_json, rulecards_dir, var_dict_path,
                       out_dir, model_bundle=None):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load core artefacts ----
    payload, labels, rules_by_label = load_ruleset(Path(rules_json))
    var_dict = json.loads(Path(var_dict_path).read_text(encoding="utf-8"))

    # optional model for synthetic checks
    clf = None
    le  = None
    feature_names_model = None
    if model_bundle is not None and joblib_load is not None:
        bundle = joblib_load(model_bundle)
        clf    = bundle["model"]
        le     = bundle["label_encoder"]
        meta   = bundle.get("meta", {})
        feature_names_model = meta.get("feature_names", None)

    # ---------------- 4.1 Rule coverage / consistency ----------------
    coverage_rows = []

    # ---------------- 4.2 Dictionary alignment -----------------------
    dict_rows = []

    # ---------------- 4.3 Synthetic profiles (optional) --------------
    synth_rows = []

    for label in labels:
        md_path = Path(rulecards_dir) / f"rulecard_{label}.md"
        if not md_path.exists():
            print(f"[WARN] Missing rulecard markdown for {label}: {md_path}")
            coverage_rows.append({
                "stratum": stratum,
                "phenotype_label": label,
                "n_rules_json": len(rules_by_label[label]),
                "n_rules_text": 0,
                "missing_features_flag": True,
                "mismatched_thresholds_flag": True,
                "missing_rulecard_file": True,
            })
            continue

        text, if_lines = parse_rulecard_md(md_path)
        n_json = len(rules_by_label[label])
        n_text = len(if_lines)

        missing_features = False
        mismatched_thresh = False

        # Check each condition in each JSON rule has some textual counterpart
        for rule in rules_by_label[label]:
            for cond in rule["path"]:
                if not condition_covered(cond, if_lines):
                    missing_features = True
                    mismatched_thresh = True
                    # no need to be super granular – any miss flags the phenotype
                    break
            if missing_features:
                break

        coverage_rows.append({
            "stratum": stratum,
            "phenotype_label": label,
            "n_rules_json": n_json,
            "n_rules_text": n_text,
            "missing_features_flag": bool(missing_features),
            "mismatched_thresholds_flag": bool(mismatched_thresh),
            "missing_rulecard_file": False,
        })

        # ----- Dictionary / parenthetical alignment per feature (4.2) -----
        # Features used in this phenotype’s rules:
        feat_set = set()
        for r in rules_by_label[label]:
            for cond in r["path"]:
                feat_set.add(cond["feature"])

        for feat in sorted(feat_set):
            in_dict = feat in var_dict
            present, parenthetical = feature_name_alignment(feat, text)

            dict_rows.append({
                "stratum": stratum,
                "phenotype_label": label,
                "feature": feat,
                "in_variable_dictionary": bool(in_dict),
                "mentions_feature_name": bool(present),
                "has_parenthetical_or_code": bool(parenthetical),
            })

        # ----- Optional synthetic-profile verification (4.3) -----
        if clf is not None and le is not None and feature_names_model is not None:
            for i, rule in enumerate(rules_by_label[label]):
                x_dict = make_synthetic_point(rule["path"], feature_names_model)
                X_df = pd.DataFrame([x_dict], columns=feature_names_model)
                pred_idx = clf.predict(X_df)[0]
                pred_label = le.inverse_transform([pred_idx])[0]

                synth_rows.append({
                    "stratum": stratum,
                    "phenotype_label": label,
                    "rule_index": i,
                    "support": rule.get("support", np.nan),
                    "purity": rule.get("purity", np.nan),
                    "predicted_label": str(pred_label),
                    "expected_label": label,
                    "match": str(pred_label) == label,
                })

    # ---- Write outputs ----
    cov_df = pd.DataFrame(coverage_rows)
    cov_path = out_dir / f"rulecard_validation_{stratum.lower()}.csv"
    cov_df.to_csv(cov_path, index=False)
    print(f"[VALIDATE] Rule coverage summary → {cov_path}")

    dict_df = pd.DataFrame(dict_rows)
    dict_path = out_dir / f"rulecard_feature_alignment_{stratum.lower()}.csv"
    dict_df.to_csv(dict_path, index=False)
    print(f"[VALIDATE] Feature/dictionary alignment → {dict_path}")

    if synth_rows:
        synth_df = pd.DataFrame(synth_rows)
        synth_path = out_dir / f"rulecard_synthetic_checks_{stratum.lower()}.csv"
        synth_df.to_csv(synth_path, index=False)
        print(f"[VALIDATE] Synthetic profile checks → {synth_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Validate LLM-generated rulecards against JSON rules and dictionary."
    )
    ap.add_argument("--stratum", required=True,
                    help="Stratum label, e.g. High_MM / Mid_MM / Low_MM")
    ap.add_argument("--rules-json", required=True,
                    help="Path to rule_ruleset.json for this stratum.")
    ap.add_argument("--rulecards-dir", required=True,
                    help="Directory containing canonical rulecard_*.md files.")
    ap.add_argument("--var-dict", required=True,
                    help="Path to variable_dictionary.json.")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write validation CSVs.")
    ap.add_argument("--model-bundle", default=None,
                    help="Optional joblib bundle with surrogate tree for synthetic checks.")

    args = ap.parse_args()

    validate_rulecards(
        stratum=args.stratum,
        rules_json=Path(args.rules_json),
        rulecards_dir=Path(args.rulecards_dir),
        var_dict_path=Path(args.var_dict),
        out_dir=Path(args.out_dir),
        model_bundle=args.model_bundle,
    )


if __name__ == "__main__":
    main()
