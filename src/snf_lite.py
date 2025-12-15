#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNF-lite (Multi-View Fusion) for SUPPORT-II phenotyping.

Theory (brief):
- Build view-specific patient *affinity* matrices:
  * C-view (chronic) & S-view (socio-contextual): Gower similarity (mixed types).
  * P-view (physiology): RBF kernel on standardized continuous features.
- Apply KNN masking + row-normalization to each view (prevents any single view dominating).
- Iteratively fuse (T iters): for each view v,
    P_v <- alpha * Pk_v  +  (1 - alpha) * mean_{u != v} P_u
  (Pk_v is the fixed KNN-normalized matrix of view v).
- Final fused matrix is the average of P_v across views.
- Spectral embedding of fused graph + KMeans yields labels.
- K is chosen using eigengap (from a KNN-sparsified Laplacian) and internal metrics.

Designed for ~9k patients. Uses dense float32 matrices with KNN sparsification where helpful.
"""

import argparse
import os
import math
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.metrics import pairwise_distances
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.run_mmsp_phase1_pam import bootstrap_stability_from_D as bootstrap_stability_from_D
from src.run_mmsp_phase1_pam import pam_kmedoids_best_of_n as pam_kmedoids_best_of_n
from src.run_mmsp_phase1_pam import pick_k_by_rule as pick_k_by_rule


# -----------------------------
# I/O utilities
# -----------------------------
def load_view(path: str, id_col: str) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        # If id isn't present, fall back to index as ID
        df = df.reset_index().rename(columns={"index": id_col})
    ids = df[id_col].to_numpy()
    X = df.drop(columns=[id_col])
    return ids, X


def align_views(
    ids_list: List[np.ndarray], X_list: List[pd.DataFrame]
) -> Tuple[np.ndarray, List[pd.DataFrame]]:
    """Intersect IDs across views and align rows in the same order."""
    id_sets = [set(ids) for ids in ids_list]
    common = set.intersection(*id_sets)
    if len(common) == 0:
        raise ValueError("No common IDs across views.")

    common = np.array(sorted(list(common)))
    out_X = []
    for ids, X in zip(ids_list, X_list):
        order = pd.Series(np.arange(len(ids)), index=ids)
        X_aligned = X.iloc[order.loc[common].values].reset_index(drop=True)
        out_X.append(X_aligned)
    return common, out_X


# -----------------------------
# Similarity builders
# -----------------------------
def _split_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return numeric_df, categorical_df based on pandas dtypes."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return df[num_cols].copy(), df[cat_cols].copy()

def gower_affinity(df: pd.DataFrame,
                   use_asym_binary: bool = True,
                   weight_by_prevalence: bool = True) -> np.ndarray:
    """
    Gower-like similarity for mixed data with an asymmetric treatment for binary flags:
      - numeric: L1 on [0,1] after min-max scaling
      - multi-level categorical: 0 if equal, 1 if not (standard Gower)
      - binary {0,1}: Jaccard distance on presence (ignore 0-0 matches)
        with optional prevalence weighting ~ p*(1-p)
    Returns: dense (n x n) float32 similarity with diag=1.
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    X_num, X_cat = _split_types(df)

    # accumulate in float64 for numeric stability, cast at end
    dist = np.zeros((n, n), dtype=np.float64)
    total_w = 0.0

    # ----- numeric part -----
    if X_num.shape[1] > 0:
        Xn = X_num.to_numpy(dtype=np.float64)
        mins = np.nanmin(Xn, axis=0)
        maxs = np.nanmax(Xn, axis=0)
        ranges = (maxs - mins)
        ranges[ranges == 0.0] = 1.0
        Xn_scaled = (Xn - mins) / ranges  # each col in [0,1]
        D_num = pairwise_distances(Xn_scaled, metric="manhattan")  # in [0, #num_cols]
        # average per numeric feature
        w = 1.0  # average across numeric columns by dividing by count
        dist += (D_num / float(X_num.shape[1])) * w
        total_w += w

    # ----- categorical / binary part -----
    if X_cat.shape[1] > 0:
        for col in X_cat.columns:
            s = X_cat[col]

            # detect exact binary {0,1}
            is_binary = False
            try:
                vals = pd.unique(s.dropna()).astype(int)
                is_binary = set(vals.tolist()).issubset({0, 1})
            except Exception:
                is_binary = False

            if use_asym_binary and is_binary:
                v = s.fillna(0).astype(int).to_numpy()
                v_row = v[:, None]
                v_col = v[None, :]
                m11 = (v_row & v_col).astype(np.float64)
                m10 = (v_row & (1 - v_col)).astype(np.float64)
                m01 = ((1 - v_row) & v_col).astype(np.float64)
                denom = m11 + m10 + m01  # pairs with at least one '1'

                d = np.zeros_like(denom, dtype=np.float64)
                np.divide(m10 + m01, denom, out=d, where=(denom > 0))

                if weight_by_prevalence:
                    p = float(v.mean())
                    w_col = p * (1.0 - p) + 1e-8
                else:
                    w_col = 1.0

                dist += w_col * d
                total_w += w_col

            else:
                # multi-level categorical: standard Gower distance
                codes = s.astype("category").cat.codes.to_numpy()
                eq = (codes[:, None] == codes[None, :]).astype(np.float64)
                d = 1.0 - eq  # 0 if equal, 1 if different
                dist += d
                total_w += 1.0

    # ----- finalize -----
    if total_w == 0.0:
        raise ValueError("No features available to compute similarity.")

    dist /= total_w
    sim = 1.0 - dist
    np.fill_diagonal(sim, 1.0)
    sim = np.clip(sim, 0.0, 1.0).astype(np.float32)
    return sim

def rbf_affinity(X: pd.DataFrame, k_local: int = 7) -> np.ndarray:
    Xz = StandardScaler().fit_transform(X.to_numpy(dtype=np.float64))
    D = pairwise_distances(Xz, metric="euclidean")
    # per-node scale = k-th NN distance
    sortD = np.sort(D, axis=1)
    sigma = sortD[:, min(k_local, sortD.shape[1]-1)]
    sigma[sigma <= 1e-12] = np.median(sigma[sigma > 0])  # guard
    A = np.exp(-(D**2) / (np.outer(sigma, sigma) + 1e-12))
    np.fill_diagonal(A, 0.0)  # keep diag 0 here; fusion can add tiny loops if needed
    return A.astype(np.float32)


def knn_mask(A, k, include_self=True):
    n = A.shape[0]
    out = np.zeros_like(A, dtype=np.float32)
    for i in range(n):
        row = A[i].copy()
        row[i] = -np.inf  # drop self for selection
        idx = np.argpartition(row, -k)[-k:]
        out[i, idx] = A[i, idx]
        if include_self:
            out[i, i] = A[i, i]
    return out


def row_normalize(A: np.ndarray, diag_to: Optional[float] = None) -> np.ndarray:
    """
    Row-stochastic normalization; optional set diagonal value afterward (e.g., None or 0.0).
    """
    A = A.astype(np.float32, copy=True)
    if diag_to is not None:
        np.fill_diagonal(A, diag_to)
    rs = A.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    A /= rs
    return A


def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def snf_fuse(affinities, k, iters, alpha):
    m = len(affinities)
    Pk = []
    for A in affinities:
        A_knn = knn_mask(A, k=k, include_self=True)
        A_knn = row_normalize(A_knn, diag_to=0.0)  # per-view, stochastic
        Pk.append(A_knn.astype(np.float32))

    P = [row_normalize(A, diag_to=0.0).astype(np.float32) for A in affinities]

    for _ in range(iters):
        P_new = []
        P_sum = None
        for v in range(m):
            if P_sum is None:
                P_sum = np.zeros_like(P[v]);  [P_sum.__iadd__(p) for p in P]
            mean_others = (P_sum - P[v]) / float(m - 1)
            Pv = alpha * Pk[v] + (1.0 - alpha) * mean_others
            Pv = row_normalize(Pv, diag_to=0.0)
            P_new.append(Pv.astype(np.float32))
        P = P_new

    fused = np.mean(P, axis=0).astype(np.float32)
    fused = symmetrize(fused)
    # np.fill_diagonal(fused, 1.0)  # small self-loops
    return fused


# -----------------------------
# K selection & clustering
# -----------------------------
def eigengap_k(
    fused: np.ndarray,
    kmin: int,
    kmax: int,
    knn_for_laplacian: int,
    fig_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Optional[int], np.ndarray]:
    """
    Suggest K via eigengap on a *sparsified* normalized Laplacian
    (keeps computation tractable and aligns with graph-clustering practice).
    Returns (K_suggested or None, eigenvalues array sorted ascending).
    """
    # Build symmetric KNN graph from fused, then normalized Laplacian
    A = knn_mask(fused, k=knn_for_laplacian, include_self=True)
    A = symmetrize(A)
    # Normalize rows & symmetrize again
    A = row_normalize(A, diag_to=0.0)
    A = symmetrize(A)
    # Degree
    d = np.asarray(A.sum(axis=1)).ravel()
    d[d == 0.0] = 1.0
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(d))
    # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    A_sp = sparse.csr_matrix(A)
    L = sparse.identity(A.shape[0], dtype=np.float64) - D_inv_sqrt @ A_sp @ D_inv_sqrt

    nev = min(max(kmax + 3, 5), A.shape[0] - 1)
    try:
        vals, _ = eigsh(L, k=nev, which="SM", tol=1e-3)
        vals = np.sort(vals)
    except Exception:
        # Fallback: return no suggestion
        vals = np.array([])
        return None, vals

    # Eigengap between lambda_i and lambda_{i+1} (i starting at 0)
    # Consider gaps within [kmin-1, kmax-1]
    K_suggest = None
    if len(vals) >= (kmax + 1):
        gaps = vals[1:] - vals[:-1]
        lo = max(kmin - 1, 0)
        hi = min(kmax - 1, len(gaps) - 1)
        if hi >= lo:
            i_best = lo + int(np.argmax(gaps[lo:hi + 1]))
            K_suggest = i_best + 1  # because gap between i and i+1 corresponds to K=i+1

    if fig_path is not None and vals.size > 0:
        plt.figure()
        plt.plot(np.arange(len(vals)), vals, marker="o", linewidth=1)
        plt.xlabel("Eigenvalue index (ascending)")
        plt.ylabel("Eigenvalue (normalized Laplacian)")
        plt.title("SNF fused graph: eigenspectrum (for eigengap heuristic)")
        plt.grid(True, alpha=0.3)
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()

    return K_suggest, vals

def cluster_and_score_with_pam(fused, ids, kmin, kmax, seed=42):
    # One embedding dimension that’s safely large:
    emb = spectral_embedding(adjacency=fused, n_components=max(kmax, 8),
                             random_state=seed, eigen_solver="arpack", drop_first=True)
    results, labels_map = [], {}
    for K in range(kmin, kmax + 1):
        Z = emb[:, :K]   # use first K eigenvectors
        D = pairwise_distances(Z, metric="euclidean")
        # stability (bootstrap on D, like phase-1)
        stab = bootstrap_stability_from_D(D, k=K, seed=seed)
        # pick best of N PAM initializations
        lab, _ = pam_kmedoids_best_of_n(D, K, n_init=20, seed=seed)
        sil = silhouette_score(Z, lab)
        ch  = calinski_harabasz_score(Z, lab)
        db  = davies_bouldin_score(Z, lab)
        results.append({"K":K, "stability_ARI":stab,
                        "silhouette":sil, "calinski_harabasz":ch, "davies_bouldin":db})
        labels_map[K] = lab

    # your rule:
    chosenK = pick_k_by_rule(results, stability_tol=0.20)
    return chosenK, pd.DataFrame(results), labels_map[chosenK]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="SNF-lite (Multi-View Fusion) clustering")
    ap.add_argument("--cview", required=True, help="Path to C_view.csv")
    ap.add_argument("--pview", required=True, help="Path to P_view.csv")
    ap.add_argument("--sview", required=True, help="Path to S_view.csv")
    ap.add_argument("--id-col", default="eid", help="Common ID column name (default: id)")

    ap.add_argument("--knn", type=int, default=25, help="K neighbors for KNN graph (default: 25)")
    ap.add_argument("--iters", type=int, default=10, help="SNF iterations (default: 10)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Weight on view-local KNN graph (default: 0.5)")

    ap.add_argument("--kmin", type=int, default=2, help="Min clusters (default: 2)")
    ap.add_argument("--kmax", type=int, default=8, help="Max clusters (default: 8)")
    ap.add_argument("--lap_knn", type=int, default=25, help="KNN for Laplacian eigengap (default: 25)")

    ap.add_argument("--out-models", default="models", help="Output dir for models/artifacts")
    ap.add_argument("--out-reports", default="reports", help="Output dir for tables/figures")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    # ap.add_argument("--id-col", default="eid", help="Common ID column (default: eid)")
    ap.add_argument("--stratum-file", help="CSV with eid + stratum column")
    ap.add_argument("--stratum", choices=["Low_MM","Mid_MM","High_MM"], help="Which stratum to run")
    ap.add_argument("--emit-K", type=int, nargs="*", default=[], dest="emit_K",
                help="Optional list of Ks to force and export alongside the selected K")

    # ...

    args = ap.parse_args()

    np.random.seed(args.seed)

    # 1) Load & align
    ids_c, Xc = load_view(args.cview, args.id_col)
    ids_p, Xp = load_view(args.pview, args.id_col)
    ids_s, Xs = load_view(args.sview, args.id_col)
    ids, (Xc, Xp, Xs) = align_views([ids_c, ids_p, ids_s], [Xc, Xp, Xs])


    def drop_low_signal(df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for c in df.columns:
            s = df[c]
            if s.nunique(dropna=True) <= 1:     # constant
                continue
            if set(s.dropna().unique()) <= {0,1}:
                p = float(s.mean())
                if p < 0.005 or p > 0.995:      # ultra-rare/common dummy
                    continue
            keep.append(c)
        return df[keep].copy()

    Xc = drop_low_signal(Xc)
    Xs = drop_low_signal(Xs)


    if args.stratum_file and args.stratum:
        s = pd.read_csv(args.stratum_file)
        s = s[s["stratum"]==args.stratum]
        keep = np.intersect1d(ids, s[args.id_col].values)
        sel = np.isin(ids, keep)
        ids, Xc, Xp, Xs = ids[sel], Xc.iloc[sel], Xp.iloc[sel], Xs.iloc[sel]

    bad_cols = set(Xc.columns) & {"death","hospdead","d.time","slos","hday","sfdm2","surv6m","prg6m","dnrday","totmcst"}
    assert not bad_cols, f"Outcome(s) found in C/S view: {bad_cols}"

    # 2) Build affinities per view
    A_c = gower_affinity(Xc)
    A_s = gower_affinity(Xs)
    A_p = rbf_affinity(Xp, k_local=7)  # or any k_local you prefer


    # 3) SNF-lite fusion
    fused = snf_fuse([A_c, A_p, A_s], k=args.knn, iters=args.iters, alpha=args.alpha)

    # 4) K suggestion via eigengap
    fig_dir = Path(args.out_reports) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    eigengap_fig = fig_dir / "snf_eigengap.png"
    K_suggest, eigvals = eigengap_k(
        fused, args.kmin, args.kmax, knn_for_laplacian=args.lap_knn, fig_path=str(eigengap_fig)
    )

    # 5) Cluster across K range & score
    K_star, metrics_df, labels = cluster_and_score_with_pam(
        fused=fused, ids=ids, kmin=args.kmin, kmax=args.kmax, seed=args.seed
    )

    # Optional: if eigengap suggestion is "close enough", prefer it (within 5% of best Silhouette)
    if K_suggest is not None and (args.kmin <= K_suggest <= args.kmax):
        sil_best = metrics_df.loc[metrics_df["K"] == K_star, "silhouette"].values[0]
        sil_sug = metrics_df.loc[metrics_df["K"] == K_suggest, "silhouette"].values[0]
        if sil_sug >= 0.95 * sil_best:
            K_star = K_suggest
            # recompute labels for K_suggest to be explicit
            emb = spectral_embedding(adjacency=fused, n_components=K_star, random_state=args.seed, eigen_solver="arpack", drop_first=True)
            km = KMeans(n_clusters=K_star, n_init=20, random_state=args.seed)
            labels = km.fit_predict(emb)

    # 6) Save outputs
    models_dir = Path(args.out_models)
    reports_dir = Path(args.out_reports)
    (reports_dir / "tables").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    forced = sorted(set(args.emit_K or []))
    all_rows = []
    for K in forced:
        emb = spectral_embedding(adjacency=fused, n_components=K, random_state=args.seed, eigen_solver="arpack")
        # PAM on embedding for consistency
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(emb, metric="euclidean")
        lab, _ = pam_kmedoids_best_of_n(D, K, n_init=20, seed=args.seed)
        pd.DataFrame({args.id_col: ids, "K": K, "cluster_id": lab}).to_csv(
            Path(args.out_models)/f"snf_clusters_K{K}.csv", index=False
        )

    # fused matrix
    np.save(models_dir / "snf_fused.npy", fused.astype(np.float32))

    # assignments
    assign_df = pd.DataFrame({args.id_col: ids, "cluster": labels.astype(int)})
    assign_df.to_csv(reports_dir / "tables" / "snf_assignments.csv", index=False)
    assign_df.rename(columns={ "cluster":"cluster_id" }, inplace=True)
    assign_df["label"] = [f"{args.stratum}_{c}" if "stratum" in vars(args) and args.stratum
                        else f"SNF_{c}" for c in assign_df["cluster_id"]]
    assign_df.to_csv(Path(args.out_models) / "mmsp_snf_clusters.csv", index=False)

    # metrics
    # add eigengap suggestion for transparency
    metrics_df["eigengap_suggested"] = K_suggest if K_suggest is not None else np.nan
    metrics_df["selected_K"] = K_star
    metrics_df.to_csv(reports_dir / "tables" / "snf_internal_metrics.csv", index=False)

    print(f"[SNF-lite] Finished. Selected K = {K_star}.")
    print(f"Assignments -> {reports_dir / 'tables' / 'snf_assignments.csv'}")
    print(f"Metrics     -> {reports_dir / 'tables' / 'snf_internal_metrics.csv'}")
    print(f"Fused mat   -> {models_dir / 'snf_fused.npy'}")
    print(f"Eigengap fig-> {eigengap_fig}")


if __name__ == "__main__":
    main()
