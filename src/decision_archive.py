import pandas as pd, json
from pathlib import Path

rows = []
for stratum, rep in [("High_MM","reports/snf_high"),
                     ("Low_MM","reports/snf_low"),
                     ("Mid_MM","reports/snf_mid")]:
    m = pd.read_csv(Path(rep)/"tables"/"snf_internal_metrics.csv")
    selK = int(m.loc[m["silhouette"].idxmax(), "K"]) if "selected_K" not in m else int(m["selected_K"].iloc[0])
    r = m.loc[m["K"]==selK].iloc[0].to_dict()
    r.update({"stratum":stratum, "method":"SNF-lite"})
    rows.append(r)

pd.DataFrame(rows).to_csv("reports/tables/model_selection_summary.csv", index=False)
print("Saved reports/tables/model_selection_summary.csv")
