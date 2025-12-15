from pathlib import Path
import pandas as pd


def get_selected_k(reports_dir: Path) -> int:
    """
    Read the SNF internal metrics for a stratum and return the unique selected_K.
    This MUST match what snf_lite wrote to snf_internal_metrics.csv.
    """
    metrics_path = reports_dir / "tables" / "snf_internal_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Cannot find metrics file: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    if "selected_K" not in metrics.columns:
        raise ValueError(f"'selected_K' column not found in {metrics_path}")

    vals = metrics["selected_K"].dropna().unique()
    if len(vals) == 0:
        raise ValueError(f"No non-missing selected_K values in {metrics_path}")
    if len(vals) > 1:
        raise ValueError(f"Multiple selected_K values in {metrics_path}: {vals}")

    return int(vals[0])


def load_assign(dir_path: str, stratum: str) -> pd.DataFrame:
    """
    Load SNF assignments for a stratum and attach the correct K
    (read from snf_internal_metrics.csv).
    """
    reports_dir = Path(dir_path)
    K = get_selected_k(reports_dir)

    a = pd.read_csv(reports_dir / "tables" / "snf_assignments.csv")
    a = a.rename(columns={"cluster": "cluster_id"})
    a["stratum"] = stratum
    a["label"] = a["stratum"] + "_" + a["cluster_id"].astype(int).astype(str)
    a["K"] = K
    a["pca_components"] = K  # keep schema parity with MMSP (not used downstream)

    return a[["eid", "stratum", "cluster_id", "label", "K", "pca_components"]]


def main():
    out = Path("data/02_clusters")
    out.mkdir(parents=True, exist_ok=True)

    # paths must match how you ran snf_lite
    hi  = load_assign("reports/snf_high", "High_MM")
    lo  = load_assign("reports/snf_low",  "Low_MM")
    mid = load_assign("reports/snf_mid",  "Mid_MM")

    L = pd.concat([hi, lo, mid], ignore_index=True)
    L = L.sort_values(["stratum", "eid"])
    out_path = out / "mmsp_clusters.csv"
    L.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(L))


if __name__ == "__main__":
    main()
