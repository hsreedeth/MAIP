#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib
import json, math, sklearn, platform
from pathlib import Path
from joblib import dump, load
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hashlib import sha256


OUTCOME_COLS = ['death','hospdead','d.time','slos','hday','sfdm2','surv6m','prg6m','dnrday','totmcst']

def _read_view(path, id_col):
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"{path} missing id column '{id_col}'")
    return df

def load_X(cview, pview, sview, id_col="eid"):
    C = _read_view(cview, id_col)
    P = _read_view(pview, id_col)
    S = _read_view(sview, id_col)

    # inner-join to enforce same patients, and drop any outcome cols if present
    for df in (C, P, S):
        drop_cols = [c for c in OUTCOME_COLS if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    X = C.merge(P, on=id_col, how="inner").merge(S, on=id_col, how="inner")
    # ensure no duplicate columns aside from id
    X = X.loc[:, ~X.columns.duplicated()]
    # keep id separate
    eids = X[id_col].copy()
    X = X.drop(columns=[id_col])

    # fill any nullable dtypes / NA (should be minimal post-imputation)
    X = X.fillna(0)

    # drop zero-variance columns. there are none its been tested. but for reproducibility's sake.
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    X = X[keep]

    return eids.values, X

def load_y(assign_csv, stratum=None, id_col="eid"):
    A = pd.read_csv(assign_csv)
    # allow either 'cluster' or 'cluster_id', have to get this consistent.
    if "cluster" in A.columns:
        cl = A[["eid","cluster"]].copy()
        cl["cluster"] = cl["cluster"].astype(int)
    elif "cluster_id" in A.columns:
        cl = A[["eid","cluster_id"]].copy()
        # cluster_id might be 0..K-1 ints. if strings like "Low_MM_1", keep as it is.
        if pd.api.types.is_integer_dtype(cl["cluster_id"]):
            cl.rename(columns={"cluster_id":"cluster"}, inplace=True)
        else:
            cl.rename(columns={"cluster_id":"label"}, inplace=True)
    elif "label" in A.columns:
        cl = A[["eid","label"]].copy()
    else:
        raise ValueError(f"{assign_csv} must contain 'cluster' or 'cluster_id' or 'label'")

    # build label
    if "label" in cl.columns:
        cl["y"] = cl["label"].astype(str)
    else:
        if stratum is None:
            cl["y"] = cl["cluster"].astype(str)
        else:
            cl["y"] = cl["cluster"].astype(int).map(lambda k: f"{stratum}_{k}")

    return cl[[id_col, "y"]]

def align_xy(eids, X, y_df, id_col="eid"):
    y_df = y_df.set_index(id_col).reindex(eids).dropna()
    # reindex X / eids to aligned subset
    idx = y_df.index.values
    mask = pd.Index(eids).isin(idx)
    X_aligned = X.loc[mask].reset_index(drop=True)
    y = y_df["y"].values
    eids_aligned = np.array([eid for eid, m in zip(eids, mask) if m])
    return eids_aligned, X_aligned, y

def cv_scores(X, y, max_depth=4, min_samples_leaf=0.02,
              n_splits=5, seed=42, class_weight=None):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []

    for tr, te in skf.split(X, y_enc):
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,   # << use same weights
            random_state=seed
        )
        clf.fit(X.iloc[tr, :], y_enc[tr])
        yp = clf.predict(X.iloc[te, :])
        accs.append(accuracy_score(y_enc[te], yp))
        f1s.append(f1_score(y_enc[te], yp, average="macro"))

    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s)), le


def fit_full(X, y, max_depth, min_samples_leaf, criterion, class_weight, seed):
    # keep label encoding consistent with your CV (if you used one)
    if isinstance(y[0], str):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        le = None
        y_enc = y

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,  # int or float fraction ok
        criterion=criterion,                # "gini" or "entropy"
        class_weight=class_weight,          # opts are: None | "balanced" | dict
        random_state=seed,
    )

    clf.fit(X, y_enc)
    return clf, le

def export_tree_png(clf, feature_names, class_names, out_png):
    plt.figure(figsize=(24, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=8
    )
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def export_text_rules(clf, feature_names, class_names, out_txt):
    txt = export_text(clf, feature_names=list(feature_names))
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(txt)

def tree_to_rules_json(clf, feature_names, class_names, out_json, stratum=None):
    T = clf.tree_
    nodes = []
    leaves = []
    for nid in range(T.node_count):
        fidx = T.feature[nid]
        if fidx >= 0:
            nodes.append({
                "id": int(nid),
                "feature": feature_names[fidx],
                "threshold": float(np.round(T.threshold[nid], 4)),
                "left": int(T.children_left[nid]),
                "right": int(T.children_right[nid])
            })
        else:
            # leaf
            probs = (T.value[nid][0] / T.value[nid][0].sum()).tolist()
            pred_idx = int(np.argmax(T.value[nid][0]))
            leaves.append({
                "id": int(nid),
                "pred_class": class_names[pred_idx],
                "probs": [float(x) for x in probs],
                "samples": int(T.n_node_samples[nid])
            })
    obj = {
        "stratum": stratum,
        "feature_names": list(feature_names),
        "class_names": list(class_names),
        "nodes": nodes,
        "leaves": leaves
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(obj, f, indent=2)

def save_confmat(y_true, y_pred, labels, out_csv, out_png=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_csv, index=True)
    if out_png:
        plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha="right")
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
        plt.close()

def _extract_rules_for_llm(clf, feature_names, class_names):
    T = clf.tree_
    rules = []
    def walk(nid, path):
        f = T.feature[nid]
        if f >= 0:
            thr = float(np.round(T.threshold[nid], 4))
            feat = feature_names[f]
            left = T.children_left[nid]
            right = T.children_right[nid]
            walk(left,  path + [{"feature": feat, "op": "<=", "threshold": thr}])
            walk(right, path + [{"feature": feat, "op": ">",  "threshold": thr}])
        else:
            counts  = T.value[nid][0]
            pred    = class_names[int(np.argmax(counts))]
            support = int(T.n_node_samples[nid])
            purity  = float(np.max(counts) / np.sum(counts))
            rules.append({"path": path, "outcome": pred, "support": support, "purity": round(purity, 3)})
    walk(0, [])
    return rules


def main():
    ap = argparse.ArgumentParser(description="Train shallow surrogate trees for SNF phenotypes (per stratum).")
    ap.add_argument("--cview", required=True)
    ap.add_argument("--pview", required=True)
    ap.add_argument("--sview", required=True)
    ap.add_argument("--snf-assign", required=True, help="CSV with eid + cluster/cluster_id/label for THIS stratum")
    ap.add_argument("--stratum", required=True, choices=["Low_MM","Mid_MM","High_MM"])
    ap.add_argument("--id-col", default="eid")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-samples-leaf", type=float, default=0.02)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--criterion", choices=["gini", "entropy"], default="gini",
                help="Split criterion for the decision tree (default: gini).")
    ap.add_argument("--balanced", action="store_true",
                help="Use class_weight='balanced' for the decision tree.")
    ap.add_argument("--class-weight-json", type=str, default=None,
                help="Optional JSON dict for class weights, e.g. '{\"High_MM_0\":1,\"High_MM_1\":1,\"High_MM_2\":1.3}'.")
    args = ap.parse_args()

    # class weights.
    class_weight = None
    if args.balanced:
        class_weight = "balanced"
    elif args.class_weight_json:
        try:
            class_weight = json.loads(args.class_weight_json)
            assert isinstance(class_weight, dict)
        except Exception as e:
            raise ValueError(f"Invalid --class-weight-json. Expect a JSON object. Got: {args.class_weight_json}") from e


    out_dir = Path(args.out_dir); (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    eids, X = load_X(args.cview, args.pview, args.sview, id_col=args.id_col)
    y_df = load_y(args.snf_assign, stratum=args.stratum, id_col=args.id_col)
    eids_a, X_a, y = align_xy(eids, X, y_df, id_col=args.id_col)

    # CV fidelity
    acc_m, acc_sd, f1_m, f1_sd, _ = cv_scores(
        X_a, y,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_splits=args.cv,
        seed=args.seed,
        class_weight=class_weight
    )
    pd.DataFrame([{
        "stratum": args.stratum,
        "n": len(y),
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "cv": args.cv,
        "accuracy_mean": acc_m, "accuracy_sd": acc_sd,
        "macro_f1_mean": f1_m, "macro_f1_sd": f1_sd
    }]).to_csv(out_dir / "tables" / "surrogate_cv_metrics.csv", index=False)

    # Fit on full data, export artifacts
    clf, le = fit_full(X_a, y, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, criterion = args.criterion, class_weight = class_weight, seed=args.seed)
    # Persist model + encoder + metadata
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    class_names = le.inverse_transform(np.arange(len(le.classes_)))  # already used below
    feature_names = list(X_a.columns)

    meta = {
        "type": "decision_tree_surrogate",
        "stratum": args.stratum,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "params": {
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "criterion": args.criterion,
            "class_weight": class_weight,
            "cv": args.cv,
            "random_seed": args.seed,
        },
        "feature_names": feature_names,
        "class_names": list(class_names),
    }

    # single portable bundle
    dump({"model": clf, "label_encoder": le, "meta": meta},
        models_dir / "surrogate_tree.joblib")

    # (opt.) human-readable sidecar
    (models_dir / "surrogate_tree.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[Surrogate] Saved model bundle -> {models_dir/'surrogate_tree.joblib'}")

    class_names = le.inverse_transform(np.arange(len(le.classes_)))
    export_tree_png(clf, X_a.columns, class_names, out_dir / "figures" / "tree.png")
    export_text_rules(clf, X_a.columns, class_names, out_dir / "tables" / "tree_rules.txt")
    tree_to_rules_json(clf, list(X_a.columns), list(class_names), out_dir / "tables" / "tree_rules.json", stratum=args.stratum)

    # LLM-ready ruleset ( the one thats preferred by cli_rulecard).
    feature_names = list(X_a.columns)
    class_names   = list(class_names)  # ensure list
    rules_llm     = _extract_rules_for_llm(clf, feature_names, class_names)

    payload_core = {
        "stratum": args.stratum,
        "K": None,  # totslly optional filled by cli_rulecard if needed
        "class_names": class_names,
        "feature_names": feature_names,
        "rules": rules_llm,
    }
    fp = sha256(json.dumps(payload_core, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    payload = dict(payload_core, model_fingerprint=fp)

    (out_dir / "tables" / "rule_ruleset.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


    # Full-data in-sample predictions -> confusion matrix (for a quick sense of fit)
    y_pred = le.inverse_transform(clf.predict(X_a))
    save_confmat(y, y_pred, labels=list(class_names),
                 out_csv=out_dir / "tables" / "confusion_matrix.csv",
                 out_png=out_dir / "figures" / "confusion_matrix.png")

    # Feature importances (rough guide only for trees). approximations.
    pd.DataFrame({
        "feature": X_a.columns,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False).to_csv(out_dir / "tables" / "feature_importances.csv", index=False)

    print(f"[Surrogate] Done for {args.stratum}. "
          f"CV acc={acc_m:.3f}±{acc_sd:.3f}, macro-F1={f1_m:.3f}±{f1_sd:.3f}. "
          f"Artifacts → {out_dir}")
if __name__ == "__main__":
    main()

# CLI Commands (SNF):
# High multimorbidity surrogate
# python -m src.surrogate_tree \
#   --cview data/01_processed/C_view.csv \
#   --pview data/01_processed/P_view_scaled.csv \
#   --sview data/01_processed/S_view.csv \
#   --snf-assign reports/snf_high/tables/snf_assignments.csv \
#   --stratum High_MM \
#   --out-dir reports/surrogate_high

# # Mid multimorbidity surrogate
# python -m src.surrogate_tree \
#   --cview data/01_processed/C_view.csv \
#   --pview data/01_processed/P_view_scaled.csv \
#   --sview data/01_processed/S_view.csv \
#   --snf-assign reports/snf_mid/tables/snf_assignments.csv \
#   --stratum Mid_MM \
#   --out-dir reports/surrogate_mid

# # Low multimorbidity surrogate
# python -m src.surrogate_tree \
#   --cview data/01_processed/C_view.csv \
#   --pview data/01_processed/P_view_scaled.csv \
#   --sview data/01_processed/S_view.csv \
#   --snf-assign reports/snf_low/tables/snf_assignments.csv \
#   --stratum Low_MM \
#   --out-dir reports/surrogate_low

# MMSP

# High multimorbidity surrogate 
# python -m src.surrogate_tree \
#   --cview data/01_processed/C_view.csv \
#   --pview data/01_processed/P_view_scaled.csv \
#   --sview data/01_processed/S_view.csv \
#   --snf-assign data/02_clusters/mmsp_clusters.csv \
#   --stratum High_MM \
#   --out-dir reports/mmsp_Dtree/surrogate_high