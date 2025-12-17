# src/external_validation.py
import warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "01_processed"
CLUS = ROOT / "data" / "02_clusters"
REPORTS = ROOT / "reports"; FIG = REPORTS / "figures"; TAB = REPORTS / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

def load_all():
    C = pd.read_csv(PROC / "C_view.csv")
    P = pd.read_csv(PROC / "P_view_scaled.csv")
    S = pd.read_csv(PROC / "S_view.csv")
    Y = pd.read_csv(PROC / "Y_validation.csv")
    L = pd.read_csv(CLUS / "mmsp_snf_clusters.csv")
    # enforce eid join
    for df in (C,P,S,Y,L):
        assert "eid" in df.columns, "eid missing"
    df = (((C.merge(P, on="eid"))
              .merge(S, on="eid"))
              .merge(Y, on="eid"))
    df = df.merge(L[["eid","stratum","label"]], on="eid", how="inner")
    return df

def km_logrank_per_stratum(df):
    out = []
    for s, g in df.groupby("stratum"):
        if not {"d.time","death"}.issubset(g.columns): 
            continue
        # overall multi-group logrank
        try:
            lr = multivariate_logrank_test(g["d.time"], g["label"], g["death"])
            p_overall = float(lr.p_value)
        except Exception:
            p_overall = np.nan
        out.append({"stratum": s, "logrank_p_overall": p_overall})
        # KM plots
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7,5))
            for lab, gg in g.groupby("label"):
                km = KaplanMeierFitter()
                km.fit(gg["d.time"], event_observed=gg["death"], label=lab)
                km.plot(ax=ax)
            ax.set_title(f"KM by cluster — {s}")
            ax.set_xlabel("time (days)"); ax.set_ylabel("survival")
            fig.tight_layout(); fig.savefig(FIG / f"km_{s}.png", dpi=160); plt.close(fig)
        except Exception as e:
            warnings.warn(f"KM plot failed for {s}: {e}")
    pd.DataFrame(out).to_csv(TAB / "km_logrank_summary.csv", index=False)
    return out

def cox_per_stratum(df):
    # Adjust for baseline severity (use 'aps' if present; else 'sps' or 'scoma')
    covar_priority = ["aps","sps","scoma"]
    out = []
    for s, g in df.groupby("stratum"):
        covar = next((c for c in covar_priority if c in g.columns), None)
        if covar is None or not {"d.time","death","label"}.issubset(g.columns): 
            continue
        dat = g[["d.time","death","label",covar]].copy()
        # One-hot the clusters (reference omitted by Cox automatically if we drop one col)
        dummies = pd.get_dummies(dat["label"], prefix="cl", drop_first=True)
        X = pd.concat([dat[["d.time","death",covar]], dummies], axis=1)
        # drop rows with NA
        X = X.dropna()
        if X["death"].sum() < 5:
            out.append({"stratum": s, "note": "too few events"})
            continue
        cph = CoxPHFitter()
        try:
            cph.fit(X, duration_col="d.time", event_col="death", robust=True)
            cph.print_summary()
            # save HR table
            hr = cph.summary.reset_index().rename(columns={"index":"term"})
            hr.insert(0,"stratum",s)
            hr.to_csv(TAB / f"cox_{s}.csv", index=False)
            out.append({"stratum": s, "ok": True})
        except Exception as e:
            out.append({"stratum": s, "error": str(e)})
    pd.DataFrame(out).to_csv(TAB / "cox_overview.csv", index=False)
    return out

def los_cost_tests(df):
    out = []
    for s, g in df.groupby("stratum"):
        row = {"stratum": s}
        for target in ["slos","totmcst"]:
            if target not in g.columns: 
                row[f"{target}_kw_p"] = np.nan; continue
            groups = [x.dropna().values for _, x in g.groupby("label")[target]]
            if sum(len(x)>0 for x in groups) < 2:
                row[f"{target}_kw_p"] = np.nan
            else:
                stat, p = kruskal(*groups)
                row[f"{target}_kw_p"] = float(p)
        out.append(row)
    pd.DataFrame(out).to_csv(TAB / "kw_los_cost.csv", index=False)
    return out

def cluster_profiles(df):
    # z-score summaries for P-view columns; prevalence for a few comorbidities
    P_cols = [c for c in ["age","scoma","avtisst","sps","aps","meanbp","wblc","hrt","resp","temp","pafi","alb","bili","crea","sod","ph","glucose","bun","urine"] if c in df.columns]
    prof = (df.groupby(["stratum","label"])[P_cols]
              .median()  # medians (P is already standardized, so these are z-meds)
              .reset_index())
    prof.to_csv(TAB / "cluster_profiles_P_medians.csv", index=False)
    # simple comorbidity prevalence
    C_bins = [c for c in df.columns if c.startswith("dzgroup_")]
    basic = ["diabetes","dementia","ca"]
    C_cols = [c for c in (C_bins + basic) if c in df.columns]
    if C_cols:
        prev = df.groupby(["stratum","label"])[C_cols].mean().reset_index()
        prev.to_csv(TAB / "cluster_profiles_C_prevalence.csv", index=False)

def main():
    df = load_all()
    # counts
    counts = df.groupby(["stratum","label"]).size().rename("n").reset_index()
    counts.to_csv(TAB / "cluster_counts.csv", index=False)
    # analyses
    km_logrank_per_stratum(df)
    cox_per_stratum(df)
    los_cost_tests(df)
    cluster_profiles(df)
    print("External validation done. See /reports/tables and /reports/figures.")

def run():
    main()

if __name__ == "__main__":
    main()
