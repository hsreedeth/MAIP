import json, argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_mmsp_metrics(cluster_dir: Path, stratum: str):
    p = cluster_dir / f"metrics_{stratum}.json"
    j = json.loads(p.read_text())
    # MMSP “final” metrics already computed at chosenK
    return {
        "method": "MMSP",
        "stratum": stratum,
        "K": int(j["bestK_rule"]),
        "stability_ARI": float(j.get("stability_ARI", np.nan)), # may be absent at final; fallback below
        "silhouette": float(j.get("silhouette", np.nan)),
        "calinski_harabasz": float(j.get("calinski_harabasz", np.nan)),
        "davies_bouldin": float(j.get("davies_bouldin", np.nan)),
        "source": str(p)
    }

def inject_mmsp_stability_from_grid(row, cluster_dir: Path):
    # Some runs only saved stability per-K in grid; fill it from there if missing
    p = cluster_dir / f"metrics_{row['stratum']}.json"
    grid = json.loads(p.read_text()).get("grid", [])
    g = pd.DataFrame(grid)
    if "stability_ARI" in g.columns:
        m = g[g["K"] == row["K"]]
        if not m.empty:
            row["stability_ARI"] = float(m["stability_ARI"].iloc[0])
    return row

def load_snf_metrics(reports_dir: Path, chosenK: int | None = None):
    t = reports_dir / "tables" / "snf_internal_metrics.csv"
    df = pd.read_csv(t)
    # If script saved the chosen K, use it; else pick the row with max silhouette (as in your code)
    if "selected_K" in df.columns:
        k = int(df["selected_K"].iloc[0])
    elif chosenK is not None:
        k = int(chosenK)
    else:
        k = int(df.sort_values(["silhouette","calinski_harabasz","davies_bouldin"],
                               ascending=[False,False,True]).iloc[0]["K"])
    row = df[df["K"] == k].iloc[0]
    return {
        "method": "SNF-lite",
        "stratum": None,  # filled by caller
        "K": int(row["K"]),
        "stability_ARI": float(row.get("stability_ARI", np.nan)),
        "silhouette": float(row["silhouette"]),
        "calinski_harabasz": float(row["calinski_harabasz"]),
        "davies_bouldin": float(row["davies_bouldin"]),
        "source": str(t)
    }

def decide(df):
    # Primary: higher stability_ARI
    # Tiebreaks: higher silhouette, higher CH, lower DB
    return df.sort_values(
        ["stability_ARI","silhouette","calinski_harabasz","davies_bouldin"],
        ascending=[False,False,False,True]
    ).iloc[0].to_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", required=True, choices=["Low_MM","Mid_MM","High_MM"])
    ap.add_argument("--mmsp-dir", default="data/02_clusters")
    ap.add_argument("--snf-reports-dir", required=True,
                    help="reports dir for this stratum, e.g. reports/snf_high")
    ap.add_argument("--out", default="reports/tables/model_selection.csv")
    args = ap.parse_args()

    mmsp = load_mmsp_metrics(Path(args.mmsp_dir), args.stratum)
    mmsp = inject_mmsp_stability_from_grid(mmsp, Path(args.mmsp_dir))
    snf  = load_snf_metrics(Path(args.snf_reports_dir))
    snf["stratum"] = args.stratum

    comp = pd.DataFrame([mmsp, snf])
    winner = decide(comp)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(outp, index=False)

    print("\n=== Model selection (", args.stratum, ") ===")
    print(comp.to_string(index=False))
    print("\nWinner:", winner["method"], "| K:", winner["K"])
    # also drop a tiny JSON you can read downstream
    Path(str(outp).replace(".csv",".json")).write_text(json.dumps({"stratum":args.stratum,"winner":winner}, indent=2))

if __name__ == "__main__":
    main()
