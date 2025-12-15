# src/normalize_rules.py
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

from src.rules_schema import (
    Predicate, Rule, RuleSet,
    compute_fingerprint, save_ruleset
)


def build_indices(nodes: List[dict], leaves: List[dict]):
    node_by_id = {n["id"]: n for n in nodes}
    leaf_by_id = {l["id"]: l for l in leaves}
    return node_by_id, leaf_by_id

def dfs_paths(node_id: int,
              node_by_id: Dict[int, dict],
              leaf_by_id: Dict[int, dict],
              path: List[Predicate],
              out_rules: List[Rule],
              class_names: List[str]):

    # Leaf?
    if node_id in leaf_by_id:
        leaf = leaf_by_id[node_id]
        probs = leaf.get("probs", [])
        purity = float(max(probs)) if probs else 1.0
        out_rules.append(
            Rule(
                path=list(path),
                outcome=leaf["pred_class"],
                support=int(leaf.get("samples", 0)),
                purity=purity
            )
        )
        return

    node = node_by_id[node_id]
    feat = node["feature"]
    thr  = float(node["threshold"])

    # Left branch: feature <= threshold
    path.append(Predicate(feature=feat, op="<=", threshold=thr))
    dfs_paths(node["left"], node_by_id, leaf_by_id, path, out_rules, class_names)
    path.pop()

    # Right branch: feature > threshold
    path.append(Predicate(feature=feat, op=">", threshold=thr))
    dfs_paths(node["right"], node_by_id, leaf_by_id, path, out_rules, class_names)
    path.pop()

def convert_tree_export_to_ruleset(tree_export: dict, stratum: str) -> RuleSet:
    feature_names = tree_export["feature_names"]
    class_names   = tree_export["class_names"]
    nodes         = tree_export["nodes"]
    leaves        = tree_export["leaves"]

    node_by_id, leaf_by_id = build_indices(nodes, leaves)

    # Assumption: root has id 0 in your export
    rules: List[Rule] = []
    dfs_paths(0, node_by_id, leaf_by_id, [], rules, class_names)

    # Sort rules for determinism: outcome, then by descending support
    rules.sort(key=lambda r: (r.outcome, -r.support))

    rs = RuleSet(
        stratum=stratum,
        K=len(class_names),
        class_names=class_names,
        feature_names=feature_names,
        rules=rules,
        model_fingerprint=compute_fingerprint(tree_export)
    )
    return rs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="path to tree_rules.json (nodes/leaves format)")
    ap.add_argument("--out", dest="outp", required=True, help="path to normalized ruleset json")
    ap.add_argument("--stratum", required=True, help="e.g., High_MM")
    args = ap.parse_args()

    tree_export = json.loads(Path(args.inp).read_text())
    ruleset = convert_tree_export_to_ruleset(tree_export, stratum=args.stratum)
    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    save_ruleset(ruleset, args.outp)

    print(f"Wrote canonical RuleSet → {args.outp}")
    print(f"Fingerprint: {ruleset.model_fingerprint}")

if __name__ == "__main__":
    main()
