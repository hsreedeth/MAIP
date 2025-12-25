#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Secondary/supporting outcome analyses for MAIP portfolio + manuscript.

Outputs (under --out-root):
  tables/
    cluster_counts.csv
    km_logrank_summary.csv
    kw_los_cost.csv
    cluster_profiles_P_medians.csv
    cluster_profiles_C_prevalence.csv
    (optional) cox_python_<stratum>.csv
    (optional) cox_python_overview.csv
  figures/
    km_High_MM.png
    km_Mid_MM.png
    km_Low_MM.png

Key features:
  - Uses a user-supplied clusters file so Python matches the clusters used in R.
  - Optional follow-up truncation horizon (default 365 days) for KM/logrank
    and optional Python Cox cross-check.
"""

import argparse
import json
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter


LABEL_RE = re.compile(r"^(High_MM|Mid_MM|Low_MM)_(\d+)$")


# 
# Cluster file validation / alignment
# 
def _ensure_stratum(L: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure clusters table has columns: eid, label, stratum.
    If stratum missing, derive from label prefix.
    Enforce stratum matches label prefix.
    """
    L = L.copy()

    if "label" not in L.columns:
        raise ValueError("Clusters file missing required column: 'label'")

    # validate label format
    labels = L["label"].astype(str)
    bad = labels.loc[~labels.str.match(LABEL_RE)].unique()
    if len(bad) > 0:
        raise ValueError(
            f"Found invalid cluster labels (expected High_MM_0 etc). Examples: {bad[:10]}"
        )

    # derive stratum if missing
    if "stratum" not in L.columns:
        L["stratum"] = labels.str.extract(r"^(High_MM|Mid_MM|Low_MM)_", expand=False)

    # enforce match between stratum and label prefix
    derived = labels.str.extract(r"^(High_MM|Mid_MM|Low_MM)_", expand=False)
    mismatch = (L["stratum"].astype(str) != derived.astype(str))
    if mismatch.any():
        ex = L.loc[mismatch, ["eid", "label", "stratum"]].head(10)
        raise ValueError(
            "Stratum does not match label prefix for some rows. Examples:\n"
            f"{ex.to_string(index=False)}"
        )

    return L


# 
# Data loading
# 
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

    # sanity checks for outcomes
    if not {"d.time", "death"}.issubset(df.columns):
        raise ValueError("Merged dataframe is missing required survival columns: d.time, death")

    return df


def add_horizon(df: pd.DataFrame, horizon_days: float | None) -> pd.DataFrame:
    """
    Create horizon-truncated time/event columns:
      t_h, death_h
    If horizon_days is None, use original d.time/death.
    """
    df = df.copy()

    if horizon_days is None:
        df["t_h"] = df["d.time"].astype(float)
        df["death_h"] = df["death"].astype(int)
        df["horizon_days"] = np.nan
        return df

    h = float(horizon_days)
    t = df["d.time"].astype(float)
    e = df["death"].astype(int)

    df["t_h"] = np.minimum(t, h)
    # event only if death occurred on/before horizon
    df["death_h"] = np.where((t <= h) & (e == 1), 1, 0).astype(int)
    df["horizon_days"] = h
    return df


# 
# Analyses
# 
def write_cluster_counts(df: pd.DataFrame, out_tables: Path) -> None:
    counts = df.groupby(["stratum", "label"]).size().rename("n").reset_index()
    counts.to_csv(out_tables / "cluster_counts.csv", index=False)


def km_logrank_per_stratum(df: pd.DataFrame, out_tables: Path, out_fig: Path) -> None:
    out = []

    for s, g in df.groupby("stratum"):
        # overall multi-group logrank
        try:
            lr = multivariate_logrank_test(g["t_h"], g["label"], g["death_h"])
            p_overall = float(lr.p_value)
        except Exception:
            p_overall = np.nan

        out.append({"stratum": s, "logrank_p_overall": p_overall, "horizon_days": g["horizon_days"].iloc[0]})

        # KM plots
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for lab, gg in g.groupby("label"):
                km = KaplanMeierFitter()
                km.fit(gg["t_h"], event_observed=gg["death_h"], label=lab)
                km.plot(ax=ax)
            h = g["horizon_days"].iloc[0]
            title_h = f" (≤{int(h)}d)" if not np.isnan(h) else ""
            ax.set_title(f"KM by cluster — {s}{title_h}")
            ax.set_xlabel("time (days)")
            ax.set_ylabel("survival")
            fig.tight_layout()
            fig.savefig(out_fig / f"km_{s}.png", dpi=160)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"KM plot failed for {s}: {e}")

    pd.DataFrame(out).to_csv(out_tables / "km_logrank_summary.csv", index=False)


def los_cost_tests(df: pd.DataFrame, out_tables: Path) -> None:
    out = []
    for s, g in df.groupby("stratum"):
        row = {"stratum": s}
        for target in ["slos", "totmcst"]:
            if target not in g.columns:
                row[f"{target}_kw_p"] = np.nan
                continue

            groups = [x.dropna().values for _, x in g.groupby("label")[target]]
            if sum(len(x) > 0 for x in groups) < 2:
                row[f"{target}_kw_p"] = np.nan
                continue

            stat, p = kruskal(*groups)
            row[f"{target}_kw_p"] = float(p)

        out.append(row)

    pd.DataFrame(out).to_csv(out_tables / "kw_los_cost.csv", index=False)


def cluster_profiles(df: pd.DataFrame, out_tables: Path) -> None:
    # P-view medians (P is already standardized -> interpret as median z)
    P_cols = [c for c in [
        "age", "scoma", "avtisst", "sps", "aps", "meanbp", "wblc", "hrt", "resp", "temp",
        "pafi", "alb", "bili", "crea", "sod", "ph", "glucose", "bun", "urine"
    ] if c in df.columns]

    if P_cols:
        prof = (
            df.groupby(["stratum", "label"])[P_cols]
              .median()
              .reset_index()
        )
        prof.to_csv(out_tables / "cluster_profiles_P_medians.csv", index=False)

    # C-view prevalence (binary diagnosis indicators etc.)
    C_bins = [c for c in df.columns if c.startswith("dzgroup_")]
    basic = ["diabetes", "dementia", "ca"]
    C_cols = [c for c in (C_bins + basic) if c in df.columns]

    if C_cols:
        prev = df.groupby(["stratum", "label"])[C_cols].mean().reset_index()
        prev.to_csv(out_tables / "cluster_profiles_C_prevalence.csv", index=False)


def cox_python_crosscheck(df: pd.DataFrame, out_tables: Path) -> None:
    """
    Optional: simple Cox per stratum for robustness (NOT your primary model).
    Uses one severity covariate + cluster dummies.
    Uses horizon-truncated t_h / death_h to match KM horizon.
    """
    covar_priority = ["aps", "sps", "scoma"]
    overview = []

    for s, g in df.groupby("stratum"):
        covar = next((c for c in covar_priority if c in g.columns), None)
        if covar is None:
            overview.append({"stratum": s, "ok": False, "note": "no severity covariate found"})
            continue

        dat = g[["t_h", "death_h", "label", covar]].copy()
        dat = dat.dropna()
        if dat["death_h"].sum() < 20:
            overview.append({"stratum": s, "ok": False, "note": "too few events"})
            continue

        dummies = pd.get_dummies(dat["label"], prefix="cl", drop_first=True)
        X = pd.concat([dat[["t_h", "death_h", covar]], dummies], axis=1)

        cph = CoxPHFitter()
        try:
            cph.fit(X, duration_col="t_h", event_col="death_h", robust=True)
            hr = cph.summary.reset_index().rename(columns={"index": "term"})
            hr.insert(0, "stratum", s)
            hr.to_csv(out_tables / f"cox_python_{s}.csv", index=False)
            overview.append({"stratum": s, "ok": True, "note": f"covar={covar}"})
        except Exception as e:
            overview.append({"stratum": s, "ok": False, "note": str(e)})

    pd.DataFrame(overview).to_csv(out_tables / "cox_python_overview.csv", index=False)


# 
# CLI
# 
def main():
    ap = argparse.ArgumentParser(description="Secondary outcome analyses (KM/logrank, LOS/cost, profiles) aligned to a provided cluster file.")
    ap.add_argument("--proc-dir", default="data/01_processed", help="Processed data directory containing C_view/P_view_scaled/S_view/Y_validation CSVs.")
    ap.add_argument("--clusters-file", required=True, help="Clusters CSV with columns eid,label,(optional)stratum. Must match R clusters.")
    ap.add_argument("--out-root", default="reports/outcomes_supporting", help="Output root directory (tables/ and figures/ will be created).")
    ap.add_argument("--horizon-days", type=float, default=365.0, help="Truncation horizon in days for KM/logrank (and optional Python Cox). Set to 0 to disable.")
    ap.add_argument("--do-python-cox", action="store_true", help="Also run a simple severity-adjusted Cox per stratum as a robustness check.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    proc_dir = (root / args.proc_dir).resolve()
    clusters_file = Path(args.clusters_file).expanduser().resolve()
    out_root = (root / args.out_root).resolve()

    out_tables = out_root / "tables"
    out_fig = out_root / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    print(f"[OUTCOMES] proc_dir={proc_dir}")
    print(f"[OUTCOMES] clusters_file={clusters_file}")
    print(f"[OUTCOMES] out_root={out_root}")

    df = load_all(proc_dir=proc_dir, clusters_file=clusters_file)

    horizon = None if args.horizon_days == 0 else float(args.horizon_days)
    df = add_horizon(df, horizon_days=horizon)

    write_cluster_counts(df, out_tables)
    km_logrank_per_stratum(df, out_tables, out_fig)
    los_cost_tests(df, out_tables)
    cluster_profiles(df, out_tables)

    if args.do_python_cox:
        cox_python_crosscheck(df, out_tables)

    print("[OUTCOMES] Done.")
    print(f"[OUTCOMES] Wrote tables → {out_tables}")
    print(f"[OUTCOMES] Wrote figures → {out_fig}")


if __name__ == "__main__":
    main()
