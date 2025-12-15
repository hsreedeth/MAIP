import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# --- Config ---
ROOT_DIR = Path(__file__).resolve().parents[1]
PROC = ROOT_DIR / "data" / "01_processed"
SRC_FILE = PROC / "support_preprocessed.csv"
OUT_FILE = PROC / "support_preprocessed_clean.csv"
SCALER_PATH = PROC / "scaler_P.joblib"   # from your preprocessing step
IMPUTER_PATH = PROC / "imputer_X.joblib" # optional: if you want to re-impute

# P-view columns used for scaling (keep in sync with your preprocessing)
P_VIEW_BASICS = [
    "age","scoma","avtisst","sps","aps","meanbp","wblc","hrt","resp","temp",
    "pafi","alb","bili","crea","sod","ph","glucose","bun","urine"
]

# Columns to round softly (physiology); leave costs/time alone or use different precision
ROUND4 = set(P_VIEW_BASICS)

def cap_clip(series: pd.Series, low=None, high=None):
    if series is None:
        return None, 0
    before = series.copy()
    s = series.copy()
    if low is not None:
        s = s.clip(lower=low)
    if high is not None:
        s = s.clip(upper=high)
    n_changed = (s != before).sum()
    return s, int(n_changed)

def clean_file(update_views: bool = True, reimpute_alb: bool = False, refit_scaler: bool = False):
    if not SRC_FILE.exists():
        print(f"[ERROR] {SRC_FILE} not found. Run preprocessing first.")
        return

    df = pd.read_csv(SRC_FILE)
    print(f"Loaded: {SRC_FILE} ({df.shape[0]} rows)")

    # --- Quick pre-clean QC (safe even if some cols are missing) ---
    def _cnt(df, col, thr):
        return int((df[col] > thr).sum()) if col in df.columns else "n/a"

    print("[Pre-QC] counts above thresholds:",
        {"age>100": _cnt(df, "age", 100),
        "pafi>700": _cnt(df, "pafi", 700),
        "alb>6": _cnt(df, "alb", 6)})

    if "eid" not in df.columns:
        raise ValueError("Missing 'eid'. Preprocessing must add it before cleaning.")

    changes = {}

    # --- Cap obvious extremes (no row drops) ---
    if "age" in df.columns:
        df["age"], n = cap_clip(df["age"], high=100)
        changes["age_capped"] = n
    if "pafi" in df.columns:
        df["pafi"], n = cap_clip(df["pafi"], high=700)
        changes["pafi_capped"] = n

    # --- Flag impossible albumin, set to NaN (no row drops) ---
    if "alb" in df.columns:
        mask_bad_alb = df["alb"] > 6
        n_bad = int(mask_bad_alb.sum())
        if n_bad:
            df.loc[mask_bad_alb, "alb"] = np.nan
        changes["alb_set_nan_gt6"] = n_bad

    # --- Optional: re-impute alb using your saved imputer over X (safer than dropping) ---
    if reimpute_alb and IMPUTER_PATH.exists():
        # reconstruct X in the same way as imputer was trained: drop outcomes; exclude eid
        # We'll infer X as: all columns except known outcomes (if present)
        Y_COLS = ['death','hospdead','d.time','slos','hday','sfdm2','surv6m','prg6m','dnrday','totmcst']
        feat_cols = [c for c in df.columns if c not in Y_COLS and c != "eid"]
        # from joblib import load
        imp = joblib.load(IMPUTER_PATH)
        # Ensure column order matches fit-time; if you saved a column-order JSON, load it here.
        X = df[feat_cols].copy()
        X_imp = imp.transform(X)  # transforms all; consistent with fit-time schema
        X_imp = pd.DataFrame(X_imp, columns=feat_cols, index=df.index)
        # Bring back only alb (or all imputed X if you prefer)
        if "alb" in feat_cols:
            n_filled = int(df["alb"].isna().sum())
            df["alb"] = X_imp["alb"]
            changes["alb_reimputed"] = n_filled
    else:
        # Simple fallback: median-impute alb if any NaNs remain
        if "alb" in df.columns and df["alb"].isna().any():
            med = float(df["alb"].median(skipna=True))
            df["alb"] = df["alb"].fillna(med)
            changes["alb_median_imputed"] = int(df["alb"].isna().sum())

    # --- Rounding (only physiologic floats, not everything) ---
    for c in ROUND4:
        if c in df.columns and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(4)

    # --- Save cleaned base file (versioned) ---
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved cleaned file: {OUT_FILE}")
    print("Changes summary:", {k: v for k, v in changes.items() if v})

    # --- Keep views/scaling in sync (recommended) ---
    if update_views:
        # Rebuild P_view from cleaned file
        p_cols = ["eid"] + [c for c in P_VIEW_BASICS if c in df.columns]
        P_view = df[p_cols].copy()

        # Use saved scaler to transform (or refit if asked)
        if SCALER_PATH.exists() and not refit_scaler:
            scaler = joblib.load(SCALER_PATH)
            P_scaled = P_view.copy()
            fcols = [c for c in P_view.columns if c != "eid"]
            P_scaled.loc[:, fcols] = scaler.transform(P_view[fcols])
            P_scaled.to_csv(PROC / "P_view_scaled.csv", index=False)
            print("Updated P_view_scaled.csv using existing scaler.")
        else:
            # Refit scaler on cleaned data (no split yet; later do train-only)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(P_view.drop(columns=["eid"]))
            joblib.dump(scaler, SCALER_PATH)
            P_scaled = P_view.copy()
            P_scaled.loc[:, P_scaled.columns != "eid"] = scaler.transform(P_view.drop(columns=["eid"]))
            P_scaled.to_csv(PROC / "P_view_scaled.csv", index=False)
            print("Refit scaler on cleaned data and updated P_view_scaled.csv.")

        # You can also rebuild C_view/S_view here if you like, to stay fully consistent.

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update-views", action="store_true", help="Do not rebuild P_view_scaled after cleaning.")
    ap.add_argument("--reimpute-alb", action="store_true", help="Re-impute 'alb' via saved MICE model if available.")
    ap.add_argument("--refit-scaler", action="store_true", help="Refit scaler on cleaned data (otherwise transform with saved scaler).")
    args = ap.parse_args()
    clean_file(update_views=not args.no_update_views, reimpute_alb=args.reimpute_alb, refit_scaler=args.refit_scaler)
