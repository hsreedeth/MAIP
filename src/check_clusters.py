from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "01_processed"
CLUS = ROOT / "data" / "02_clusters"

labels = pd.read_csv(CLUS / "mmsp_clusters.csv")
print(labels.groupby(["stratum","label"]).size().rename("n").reset_index())
print("\nMin cluster size by stratum:")
print(labels.groupby("stratum")["label"].value_counts().groupby(level=0).min())

# flag very small clusters (< 2% of stratum or < 50 pts)
small = []
for s, g in labels.groupby("stratum"):
    n = len(g)
    counts = g["label"].value_counts()
    small.extend([(s, lab, cnt, round(cnt/n*100,2)) for lab, cnt in counts.items() if (cnt < 50 or cnt/n < 0.02)])
if small:
    print("\n[WARN] Tiny clusters:")
    for row in small: print(row)
