# src/rules_schema.py
import json, hashlib
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Predicate:
    feature: str
    op: str            # "<=" or ">"
    threshold: float

@dataclass
class Rule:
    path: List[Predicate]      # ordered predicates root→leaf
    outcome: str               # e.g., "High_MM_2"
    support: int               # n samples at the leaf
    purity: float              # max class prob at the leaf (0..1)

@dataclass
class RuleSet:
    stratum: str
    K: int
    class_names: List[str]
    feature_names: List[str]
    rules: List[Rule]          # sorted by (outcome, -support)
    model_fingerprint: str     # sha256 of the ORIGINAL tree export

def compute_fingerprint(tree_export: dict) -> str:
    """Stable hash of the raw tree export you’re converting (nodes+leaves+names)."""
    s = json.dumps(tree_export, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def save_ruleset(ruleset: RuleSet, path):
    with open(path, "w") as f:
        json.dump(asdict(ruleset), f, indent=2, sort_keys=True)

def load_ruleset(path) -> RuleSet:
    return json.loads(open(path).read())
