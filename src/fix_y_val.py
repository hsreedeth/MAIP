# scripts/fix_y_validation.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "01_processed"

base = PROC / "support_preprocessed_clean.csv"
if not base.exists():
    base = PROC / "support_preprocessed.csv"

df = pd.read_csv(base)
ycols = [c for c in ["death","hospdead","d.time","slos","hday","sfdm2","surv6m","prg6m","dnrday","totmcst"] if c in df.columns]
Y = df[["eid"] + ycols].copy()
Y.to_csv(PROC / "Y_validation.csv", index=False)
print("Wrote Y_validation.csv with eid.")
