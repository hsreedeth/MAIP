import json
from pathlib import Path

# Paths to your surrogate rule JSONs
RULE_FILES = [
    Path("reports/surrogate_high/tables/rule_ruleset.json"),
    Path("reports/surrogate_mid/tables/rule_ruleset.json"),
    Path("reports/surrogate_low/tables/rule_ruleset.json"),
]

VAR_DICT_PATH = Path("rag_corpus/variable_dictionary.json")

# ---- collect all features used by any surrogate ----
all_features = set()

for p in RULE_FILES:
    obj = json.loads(p.read_text(encoding="utf-8"))
    # safest: trust feature_names *and* scan rule paths
    if "feature_names" in obj:
        all_features.update(obj["feature_names"])
    if "rules" in obj:
        for r in obj["rules"]:
            for step in r.get("path", []):
                feat = step.get("feature")
                if feat:
                    all_features.add(feat)

print(f"Total unique features used by surrogates: {len(all_features)}")

# ---- load variable dictionary ----
var_dict = json.loads(VAR_DICT_PATH.read_text(encoding="utf-8"))

# ignore meta keys like "_meta"
vd_keys = {k for k in var_dict.keys() if not k.startswith("_")}

missing = sorted(all_features - vd_keys)
extra   = sorted(vd_keys - all_features)

print("\n=== Missing in variable_dictionary.json (must add) ===")
for m in missing:
    print("  -", m)

print("\n=== Present in variable_dictionary.json but unused (OK) ===")
for e in extra:
    print("  -", e)
