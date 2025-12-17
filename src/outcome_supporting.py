#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supporting outcome analyses for SNF-lite clusters (secondary outputs).

Produces:
  - Kaplan–Meier plots per stratum (+ optional 3-panel combined)
  - Overall logrank p-values per stratum
  - LOS and cost Kruskal–Wallis tests per stratum
  - Cluster profiles: P-view medians (z-median) + comorbidity prevalence

Optional:
  - Simple severity-adjusted Cox per stratum (APS/SPS/SCOMA only) as robustness check.

Designed to complement primary R Cox outputs:
  - reports/tables/cox_snflite_clusters_adjusted.csv
  - reports/tables/cox_snflite_clusters_zph.csv

Example:
  python -m src.outcome_supporting \
    --clusters-file data/02_clusters/snf_clusters_all.csv \
    --horizon-days 365 \
    --out-root reports

  # include APS-only Cox check
  python -m src.outcome_supporting \
    --clusters-file data/02_clusters/snf_clusters_all.csv \
    --horizon-days 365 \
    --run-cox-check \
    --out-root reports
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test


# -----------------------------
# IO + utilities
# -----------------------------

def _ensure_stratum(L: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'stratum' exists. If not provided, infer from label prefixes like 'High_MM_0'.
    """
    if "stratum" in L.columns:
        return L
    if "label" not in L.columns:
        raise ValueError("Clusters file must contain either 'stratum' or 'label'.")
    # label example: High_MM_0 -> stratum = High_MM
    parts = L["label"].astype(str).str.split("_", expand=True)
    if parts.shape[1] >= 2:
        L = L.copy()
        L["stratum"] = parts[0].astype(str) + "_" + parts[1].astype(str)
        return L
    raise ValueError("Could not infer 'stratum' from label; please include 'stratum' column.")


def load_all(proc_dir: Path, clusters_file: Path) -> pd.DataFrame:
    C = pd.read_csv(proc_dir / "C_view.csv")
    P = pd.read_csv(proc_dir / "P_view_scaled.csv")
    S = pd.read_csv(proc_dir / "S_view.csv")
    Y = pd.read_csv(proc_dir / "Y_validation.csv")
    L = pd.read_csv(clusters_file)

    for df, name in [(C, "C_view"), (P, "P_view_scaled"), (S, "S_view"), (Y, "Y_validation"), (L, "clusters")]:
        if "eid" not in df.columns:
            raise ValueError(f"'eid' missing in {name}")

    L = _ensure_stratum(L)

    needed_cols = {"eid", "stratum", "label"}
    if not needed_cols.issubset(L.columns):
        raise ValueError(f"Clusters file must include columns {sorted(needed_cols)}")

    df = (
        C.merge(P, on="eid", how="inner")
         .merge(S, on="eid", how="inner")
         .merge(Y, on="eid", how="inner")
         .merge(L[["eid", "stratum", "label"]], on="eid", how="inner")
    )

    return df


def apply_horizon(df: pd.DataFrame, horizon_days: int | None) -> pd.DataFrame:
    df = df.copy()
    if horizon_days is None:
        # keep original names for downstream simplicity
        df["t"] = df["d.time"]
        df["e"] = df["death"]
        df["horizon_days"] = np.nan
        return df

    H = float(horizon_days)
    df["t"] = np.minimum(df["d.time"].astype(float), H)
    df["e"] = np.where((df["d.time"].astype(float) <= H) & (df["death"] == 1), 1, 0)
    df["horizon_days"] = horizon_days
    return df


# -----------------------------
# Analyses
# -----------------------------

def km_logrank_per_stratum(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("stratum"):
        if not {"t", "e", "label"}.issubset(g.columns):
            continue

        # Overall multi-group logrank
        try:
            lr = multivariate_logrank_test(g["t"], g["label"], g["e"])
            p_overall = float(lr.p_value)
        except Exception:
            p_overall = np.nan

        rows.append({
            "stratum": s,
            "n": int(len(g)),
            "events": int(g["e"].sum()),
            "logrank_p_overall": p_overall,
            "horizon_days": g["horizon_days"].iloc[0],
        })

        # KM plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for lab, gg in g.groupby("label"):
                km = KaplanMeierFitter()
                km.fit(gg["t"], event_observed=gg["e"], label=str(lab))
                km.plot(ax=ax)
            ax.set_title(f"KM by cluster — {s}")
            ax.set_xlabel("time (days)")
            ax.set_ylabel("survival")
            fig.tight_layout()
            fig.savefig(fig_dir / f"km_{s}.png", dpi=180)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"KM plot failed for {s}: {e}")

    return pd.DataFrame(rows)


def km_all_strata_panel(fig_dir: Path, strata=("High_MM", "Mid_MM", "Low_MM")) -> None:
    """
    If per-stratum KM plots exist, stitch into a single 3-panel figure.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    imgs = []
    for s in strata:
        p = fig_dir / f"km_{s}.png"
        if not p.exists():
            return
        imgs.append(Image.open(p).convert("RGB"))

    # simple horizontal concat
    widths = [im.size[0] for im in imgs]
    heights = [im.size[1] for im in imgs]
    W, H = sum(widths), max(heights)

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    out = fig_dir / "km_all_strata.png"
    canvas.save(out)


def los_cost_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("stratum"):
        row = {"stratum": s, "horizon_days": g["horizon_days"].iloc[0]}
        for target in ["slos", "totmcst"]:
            if target not in g.columns:
                row[f"{target}_kw_p"] = np.nan
                continue
            groups = [x.dropna().values for _, x in g.groupby("label")[target]]
            if sum(len(x) > 0 for x in groups) < 2:
                row[f"{target}_kw_p"] = np.nan
            else:
                _, p = kruskal(*groups)
                row[f"{target}_kw_p"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # P-view medians (z-scale)
    P_cols = [c for c in [
        "age", "scoma", "avtisst", "sps", "aps",
        "meanbp", "wblc", "hrt", "resp", "temp",
        "pafi", "alb", "bili", "crea", "sod", "ph",
        "glucose", "bun", "urine"
    ] if c in df.columns]

    prof = (
        df.groupby(["stratum", "label"])[P_cols]
          .median(numeric_only=True)
          .reset_index()
    )

    # C-view prevalence (dzgroup_*) + a few basics if present
    C_bins = [c for c in df.columns if c.startswith("dzgroup_")]
    basics = [c for c in ["diabetes", "dementia", "ca"] if c in df.columns]
    C_cols = C_bins + basics

    prev = pd.DataFrame()
    if len(C_cols) > 0:
        prev = (
            df.groupby(["stratum", "label"])[C_cols]
              .mean(numeric_only=True)
              .reset_index()
        )

    return prof, prev


def cox_severity_check(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """
    Simple per-stratum Cox: cluster dummies + one severity covariate (APS/SPS/SCOMA).
    This is a robustness/sanity-check only; primary inference is in R.
    """
    covar_priority = ["aps", "sps", "scoma"]
    out_rows = []

    for s, g in df.groupby("stratum"):
        covar = next((c for c in covar_priority if c in g.columns), None)
        if covar is None:
            out_rows.append({"stratum": s, "status": "skip_no_severity"})
            continue
        if not {"t", "e", "label"}.issubset(g.columns):
            out_rows.append({"stratum": s, "status": "skip_missing_survcols"})
            continue

        dat = g[["t", "e", "label", covar]].copy()
        dummies = pd.get_dummies(dat["label"], prefix="cl", drop_first=True)
        X = pd.concat([dat[["t", "e", covar]], dummies], axis=1).dropna()

        if X["e"].sum() < 5:
            out_rows.append({"stratum": s, "status": "skip_too_few_events"})
            continue

        cph = CoxPHFitter()
        try:
            cph.fit(X, duration_col="t", event_col="e", robust=True)
            hr = cph.summary.reset_index().rename(columns={"index": "term"})
            hr.insert(0, "stratum", s)
            hr.insert(1, "severity_covariate", covar)
            hr.insert(2, "horizon_days", g["horizon_days"].iloc[0])
            hr.to_csv(tables_dir / f"cox_severity_check_{s}.csv", index=False)
            out_rows.append({"stratum": s, "status": "ok", "severity_covariate": covar})
        except Exception as e:
            out_rows.append({"stratum": s, "status": "error", "error": str(e)})

    return pd.DataFrame(out_rows)



# Main

def main():
    ap = argparse.ArgumentParser(description="Supporting outcome analyses for SNF-lite cluster phenotypes.")
    ap.add_argument("--clusters-file", required=True, help="CSV with eid, label, and stratum (or inferable from label).")
    ap.add_argument("--proc-dir", default="data/01_processed", help="Directory containing processed C/P/S/Y views.")
    ap.add_argument("--out-root", default="reports", help="Output root; writes to <out-root>/tables and <out-root>/figures.")
    ap.add_argument("--horizon-days", type=int, default=365, help="Follow-up truncation horizon in days. Use 0 for untruncated.")
    ap.add_argument("--run-cox-check", action="store_true", help="Run simple severity-adjusted Cox as robustness check.")
    args = ap.parse_args()

    proc_dir = Path(args.proc_dir)
    clusters_file = Path(args.clusters_file)
    out_root = Path(args.out_root)
    tables_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    horizon = None if args.horizon_days == 0 else int(args.horizon_days)

    df = load_all(proc_dir=proc_dir, clusters_file=clusters_file)
    df = apply_horizon(df, horizon_days=horizon)

    # Counts (useful for reporting)
    counts = df.groupby(["stratum", "label"]).size().rename("n").reset_index()
    counts.to_csv(tables_dir / "cluster_counts.csv", index=False)

    # KM + logrank
    km_sum = km_logrank_per_stratum(df, fig_dir=fig_dir)
    km_sum.to_csv(tables_dir / "km_logrank_summary.csv", index=False)
    km_all_strata_panel(fig_dir)

    # LOS/cost tests
    kw = los_cost_tests(df)
    kw.to_csv(tables_dir / "kw_los_cost.csv", index=False)

    # Cluster profiles
    prof_p, prev_c = cluster_profiles(df)
    prof_p.to_csv(tables_dir / "cluster_profiles_P_medians.csv", index=False)
    if not prev_c.empty:
        prev_c.to_csv(tables_dir / "cluster_profiles_C_prevalence.csv", index=False)

    # Optional Cox (robustness only)
    if args.run_cox_check:
        overview = cox_severity_check(df, tables_dir=tables_dir)
        overview.to_csv(tables_dir / "cox_severity_check_overview.csv", index=False)

    print("[OK] Supporting outcomes complete.")
    print(f"  Tables:  {tables_dir}")
    print(f"  Figures: {fig_dir}")


if __name__ == "__main__":
    main()
