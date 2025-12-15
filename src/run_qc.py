import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "01_processed"
DEFAULT_INPUT = PROCESSED_DIR / "support_preprocessed.csv"   # can override via CLI
QC_BASE_DIR = ROOT_DIR / "reports" / "qc"
QC_FIGURE_DIR = QC_BASE_DIR / "figures"
QC_VIOL_DIR = QC_BASE_DIR / "violations"

# -----------------------------
# QC CONFIGURATION
# -----------------------------
QC_CONFIG = {
    # Demographics
    "age":   {"plausible_range": [18, 100], "check_iqr_outliers": True},

    # Physiology
    "meanbp": {"plausible_range": [0, 250], "check_iqr_outliers": True},
    "wblc":   {"plausible_range": [0, 201], "check_iqr_outliers": True},
    "hrt":    {"plausible_range": [0, 301], "check_iqr_outliers": True},
    "resp":   {"plausible_range": [0, 100], "check_iqr_outliers": True},
    "temp":   {"plausible_range": [30, 45], "check_iqr_outliers": True},
    "pafi":   {"plausible_range": [0, 700], "check_iqr_outliers": True},
    "alb":    {"plausible_range": [0, 6],   "check_iqr_outliers": True},
    "bili":   {"plausible_range": [0, 65],  "check_iqr_outliers": True},
    "crea":   {"plausible_range": [0, 22],  "check_iqr_outliers": True},
    "sod":    {"plausible_range": [100, 182], "check_iqr_outliers": True},
    "ph":     {"plausible_range": [6.8, 7.8], "check_iqr_outliers": True},
    "bun":    {"plausible_range": [0, 301], "check_iqr_outliers": True},
    "urine":  {"plausible_range": [0, 15000], "check_iqr_outliers": True},

    # Scores
    "sps":    {"check_iqr_outliers": True},
    "aps":    {"check_iqr_outliers": True},
    "scoma":  {"check_iqr_outliers": True},
    "adlp_s": {"check_iqr_outliers": True},

    # Outcomes (often heavy-tailed)
    "totmcst": {"plausible_range": [0, None], "check_iqr_outliers": True},
    "slos":    {"plausible_range": [0, None], "check_iqr_outliers": True},
}

# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    QC_BASE_DIR.mkdir(parents=True, exist_ok=True)
    QC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    QC_VIOL_DIR.mkdir(parents=True, exist_ok=True)

def coerce_numeric(df: pd.DataFrame, cols):
    """Coerce configured columns to numeric (errors→NaN)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def plausible_violations(df: pd.DataFrame, col: str, min_val, max_val) -> pd.Series:
    v = pd.Series(False, index=df.index)
    if min_val is not None:
        v |= df[col] < float(min_val)
    if max_val is not None:
        v |= df[col] > float(max_val)
    # exclude NaNs from being counted as violations
    v &= df[col].notna()
    return v

def iqr_bounds(s: pd.Series):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

def summarize_counts(n_viol, n_outliers, n_total):
    return {
        "n_total": int(n_total),
        "n_plausible_violations": int(n_viol),
        "pct_plausible_violations": float(100 * n_viol / n_total) if n_total else 0.0,
        "n_iqr_outliers": int(n_outliers),
        "pct_iqr_outliers": float(100 * n_outliers / n_total) if n_total else 0.0,
    }

def plot_var(df: pd.DataFrame, col: str, min_val, max_val, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Box Plot of {col}")

    # shade plausible range if defined
    if min_val is not None:
        axes[0].axvline(min_val, linestyle="--")
        axes[1].axvline(min_val, linestyle="--")
    if max_val is not None:
        axes[0].axvline(max_val, linestyle="--")
        axes[1].axvline(max_val, linestyle="--")

    plt.suptitle(f"QC Plots for {col}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = out_dir / f"{col}_qc.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path

# -----------------------------
# Main QC functions
# -----------------------------
def run_qc(input_path: Path, visualize: bool = False, max_print: int = 5):
    ensure_dirs()

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    print(f"Loaded: {input_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

    eid_present = "eid" in df.columns
    if not eid_present:
        print("[WARN] 'eid' not found—violation exports will omit patient IDs.")

    # Coerce configured vars to numeric
    vars_to_check = [c for c in QC_CONFIG.keys() if c in df.columns]
    df = coerce_numeric(df, vars_to_check)

    summary_rows = []
    percol_violation_files = []

    for col, rules in QC_CONFIG.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        n_total = int(df[col].notna().sum())

        # Range violations
        min_val, max_val = rules.get("plausible_range", [None, None])
        vmask = plausible_violations(df, col, min_val, max_val) if (min_val is not None or max_val is not None) else pd.Series(False, index=df.index)

        # IQR outliers
        omask = pd.Series(False, index=df.index)
        if rules.get("check_iqr_outliers", False) and n_total >= 10:
            lb, ub = iqr_bounds(df[col].dropna())
            omask = (df[col] < lb) | (df[col] > ub)
            omask &= df[col].notna()

        # Summarize
        nV = int(vmask.sum())
        nO = int(omask.sum())
        summary = {"variable": col, **summarize_counts(nV, nO, n_total)}
        if min_val is not None or max_val is not None:
            summary.update({"plausible_min": min_val, "plausible_max": max_val})
        summary_rows.append(summary)

        # Export violations per-column
        if nV > 0:
            cols_to_export = (["eid"] if eid_present else []) + [col]
            viol_df = df.loc[vmask, cols_to_export].sort_values(by=col, ascending=False)
            out_csv = QC_VIOL_DIR / f"{col}_range_violations.csv"
            viol_df.to_csv(out_csv, index=False)
            percol_violation_files.append(str(out_csv))
            # Quick console peek
            print(f"[{col}] {nV} outside plausible range; top {min(max_print, nV)}:")
            print(viol_df.head(max_print).to_string(index=False))

        if rules.get("check_iqr_outliers", False) and nO > 0:
            print(f"[{col}] {nO} IQR outliers (not necessarily 'errors'); bounds are data-driven.")

        # Visuals
        if visualize:
            plot_var(df, col, min_val, max_val, QC_FIGURE_DIR)

    # Write summary CSV + JSON
    qc_summary_df = pd.DataFrame(summary_rows).sort_values("variable")
    qc_summary_csv = QC_BASE_DIR / "qc_summary.csv"
    qc_summary_df.to_csv(qc_summary_csv, index=False)

    qc_meta = {
        "input": str(input_path),
        "n_rows": int(len(df)),
        "violation_files": percol_violation_files,
        "summary_csv": str(qc_summary_csv),
    }
    with open(QC_BASE_DIR / "qc_meta.json", "w") as f:
        json.dump(qc_meta, f, indent=2)

    print("\nQC summary written to:", qc_summary_csv)
    if visualize:
        print("Figures saved to:", QC_FIGURE_DIR)
    if percol_violation_files:
        print("Per-variable violation CSVs:", len(percol_violation_files), "→", QC_VIOL_DIR)
    else:
        print("No range violations exported.")

    return 0

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run QC on SUPPORT-II preprocessed data.")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                    help="Path to CSV (e.g., support_preprocessed.csv or support_preprocessed_clean.csv).")
    ap.add_argument("--visualize", action="store_true", help="Save hist+box plots with plausible range markers.")
    ap.add_argument("--max-print", type=int, default=5, help="Max rows to print per violation category.")
    args = ap.parse_args()

    exit_code = run_qc(Path(args.input), visualize=args.visualize, max_print=args.max_print)
    raise SystemExit(exit_code)

if __name__ == "__main__":
    main()
