# src/flowchart_fallback.py
import json, pathlib
from typing import Dict, Any

def ascii_flow_from_rules(rules_json_path: str) -> str:
    rules = json.loads(pathlib.Path(rules_json_path).read_text(encoding="utf-8"))
    lines = [f"Flow (fallback) — {rules['stratum']} (K={rules['K']})", ""]
    for r in rules["rules"]:
        path = " AND ".join([f"{p['feature']} {p['op']} {p['threshold']}" for p in r["path"]])
        lines.append(f"* ({path}) -> {r['outcome']}  [n={r['support']}, purity={r['purity']:.2f}]")
    return "\n".join(lines)

def rulecard_from_rules(rules_json_path: str, glossary_path: str) -> str:
    rules = json.loads(pathlib.Path(rules_json_path).read_text(encoding="utf-8"))
    glossary = json.loads(pathlib.Path(glossary_path).read_text(encoding="utf-8"))
    out = [f"# Rulecard (fallback) — {rules['stratum']} (K={rules['K']})", ""]
    by_outcome: Dict[str, list] = {c: [] for c in rules["class_names"]}
    for r in rules["rules"]:
        tokens = [f"{p['feature']} {p['op']} {p['threshold']}" for p in r["path"]]
        gloss = [f"- {p['feature']}: {glossary.get(p['feature'], '(no description)')}" for p in r["path"]]
        by_outcome[r["outcome"]].append((tokens, gloss, r["support"], r["purity"]))

    for outcome in rules["class_names"]:
        out.append(f"## {outcome}")
        for tokens, gloss, n, purity in by_outcome[outcome]:
            out.append(f"- ({' AND '.join(tokens)})")
            out.extend([f"  {g}" for g in gloss])
            out.append(f"  Support={n}, Purity={purity:.2f}")
        out.append("")
    return "\n".join(out).strip()
