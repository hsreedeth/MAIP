#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNF-lite sensitivity: stability-ARI vs hyperparameters (one-page figure).

Vary one hyperparameter at a time (knn, iters, alpha, lap_knn), hold others fixed.
Compute fused graph -> spectral embedding -> PAM (k-medoids) stability on K (default: 3).
Outputs CSV of results + a single 2x2 figure.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.manifold import spectral_embedding

# reuse your existing implementations
from src.snf_lite import (
    load_view, align_views, gower_affinity, rbf_affinity,
    knn_mask, row_normalize, symmetrize, snf_fuse, eigengap_k
)
from src.run_mmsp_phase1_pam import (
    bootstrap_stability_from_D, pam_kmedoids_best_of_n
)

def stability_for_params(Xc, Xp, Xs, K, knn, iters, alpha, lap_knn, seed=42):
    # build per-view affinities
    A_c = gower_affinity(Xc)
    A_s = gower_affinity(Xs)
    A_p = rbf_affinity(Xp)

    # fuse
    fused = snf_fuse([A_c, A_p, A_s], k=knn, iters=iters, alpha=alpha)

    # spectral embedding on fused graph
    emb = spectral_embedding(
        adjacency=fused, n_components=max(K, 8),
        random_state=seed, eigen_solver="arpack", drop_first=True
    )
    Z = emb[:, :K]
    D = pairwise_distances(Z, metric="euclidean")

    # bootstrap stability (same routine as MMSP)
    stab = bootstrap_stability_from_D(D, k=K, seed=seed)

    return float(stab)

def main():
    ap = argparse.ArgumentParser(description="SNF-lite sensitivity sweep (stability-ARI vs hyperparams)")
    ap.add_argument("--cview", required=True)
    ap.add_argument("--pview", required=True)
    ap.add_argument("--sview", required=True)
    ap.add_argument("--id-col", default="eid")
    ap.add_argument("--stratum-file", help="CSV with eid + stratum")
    ap.add_argument("--stratum", choices=["Low_MM","Mid_MM","High_MM"])

    # baseline (use what you ran last)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--lap-knn", type=int, default=15)

    # sweep grids (feel free to tweak)
    ap.add_argument("--grid-knn", type=int, nargs="+", default=[5,10,15,20,25,30])
    ap.add_argument("--grid-iters", type=int, nargs="+", default=[5,10,15,20,30])
    ap.add_argument("--grid-alpha", type=float, nargs="+", default=[0.3,0.5,0.7,0.9])
    ap.add_argument("--grid-lap-knn", type=int, nargs="+", default=[10,15,20,25,30])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-reports", default="reports/sensitivity")
    args = ap.parse_args()

    np.random.seed(args.seed)
    Path(args.out_reposts if hasattr(args, "out_reposts") else args.out_reports).mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.out_reports) / "figures"
    tab_dir = Path(args.out_reports) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # load + align
    ids_c, Xc = load_view(args.cview, args.id_col)
    ids_p, Xp = load_view(args.pview, args.id_col)
    ids_s, Xs = load_view(args.sview, args.id_col)
    ids, (Xc, Xp, Xs) = align_views([ids_c, ids_p, ids_s], [Xc, Xp, Xs])

    # optional: restrict to a stratum
    if args.stratum_file and args.stratum:
        s = pd.read_csv(args.stratum_file)
        keep = set(s.loc[s["stratum"] == args.stratum, args.id_col].values.tolist())
        mask = [i in keep for i in ids]
        Xc = Xc.loc[mask].reset_index(drop=True)
        Xp = Xp.loc[mask].reset_index(drop=True)
        Xs = Xs.loc[mask].reset_index(drop=True)

    # baseline
    base = dict(K=args.K, knn=args.knn, iters=args.iters, alpha=args.alpha, lap_knn=args.lap_knn)

    rows = []

    # sweep knn
    for val in args.grid_knn:
        stab = stability_for_params(Xc, Xp, Xs, K=base["K"], knn=val, iters=base["iters"],
                                    alpha=base["alpha"], lap_knn=base["lap_knn"], seed=args.seed)
        rows.append({"param":"knn","value":val,"stability_ARI":stab})

    # sweep iters
    for val in args.grid_iters:
        stab = stability_for_params(Xc, Xp, Xs, K=base["K"], knn=base["knn"], iters=val,
                                    alpha=base["alpha"], lap_knn=base["lap_knn"], seed=args.seed)
        rows.append({"param":"iters","value":val,"stability_ARI":stab})

    # sweep alpha
    for val in args.grid_alpha:
        stab = stability_for_params(Xc, Xp, Xs, K=base["K"], knn=base["knn"], iters=base["iters"],
                                    alpha=val, lap_knn=base["lap_knn"], seed=args.seed)
        rows.append({"param":"alpha","value":val,"stability_ARI":stab})

    # sweep lap_knn (it only affects eigengap normally, but include to stunt on these critiques)
    for val in args.grid_lap_knn:
        stab = stability_for_params(Xc, Xp, Xs, K=base["K"], knn=base["knn"], iters=base["iters"],
                                    alpha=base["alpha"], lap_knn=val, seed=args.seed)
        rows.append({"param":"lap_knn","value":val,"stability_ARI":stab})

    df = pd.DataFrame(rows)
    # save the raw grid
    suffix = args.stratum if args.stratum else "ALL"
    df.to_csv(tab_dir / f"snf_sensitivity_{suffix}.csv", index=False)

    # # one-page plot: 2x2 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    # panels = [("knn", axes[0,0]), ("iters", axes[0,1]), ("alpha", axes[1,0]), ("lap_knn", axes[1,1])]

    # for name, ax in panels:
    #     sub = df[df["param"] == name].sort_values("value")
    #     ax.plot(sub["value"], sub["stability_ARI"], marker="o")
    #     ax.set_title(f"Stability vs {name}")
    #     ax.set_xlabel(name)
    #     ax.set_ylabel("Bootstrap ARI")
    #     ax.grid(True, alpha=0.3)

    #     # baseline marker
    #     base_x = base[name]
    #     if name in sub["value"].values:
    #         base_y = float(sub.loc[sub["value"] == base_x, "stability_ARI"].iloc[0])
    #         ax.axvline(base_x, linestyle="--", alpha=0.5)
    #         ax.scatter([base_x], [base_y])

    # title_stratum = f" ({args.stratum})" if args.stratum else ""
    # fig.suptitle(f"SNF-lite Stability Sensitivity{title_stratum}  |  K={base['K']}  (vary one hyperparam at a time)")
    # out_png = fig_dir / f"snf_sensitivity_{suffix}.png"
    # fig.savefig(out_png, dpi=200)
    # plt.close(fig)

    # print(f"[OK] Saved grid -> {tab_dir / f'snf_sensitivity_{suffix}.csv'}")
    # print(f"[OK] Saved figure -> {out_png}")

        # one-page plot: 2x2 subplots, with shared y-axis
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.ravel()

    panels = [("knn", axes[0]), ("iters", axes[1]), ("alpha", axes[2]), ("lap_knn", axes[3])]

    # common y-axis range (tweak if needed)
    YMIN, YMAX = 0.40, 0.50
    global_med = df["stability_ARI"].median()

    y_lo = float(df["stability_ARI"].min())
    y_hi = float(df["stability_ARI"].max())
    pad = max(0.005, 0.02 * (y_hi - y_lo))  # 2% or at least 0.005
    y_min, y_max = max(0.0, y_lo - pad), min(1.0, y_hi + pad)

    for name, ax in panels:
        sub = df[df["param"] == name].sort_values("value")
        x = sub["value"].values
        y = sub["stability_ARI"].values

        ax.plot(x, y, marker="o", linestyle="-")
        ax.set_ylim(y_min, y_max)
        ax.axhspan(global_med-0.005, global_med+0.005, alpha=0.08, color="gray")  # median band

        ax.set_title(f"Stability vs {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Bootstrap ARI")
        ax.set_ylim(YMIN, YMAX)
        ax.grid(True, alpha=0.3)

        # baseline marker
        base_x = base[name]
        if base_x in sub["value"].values:
            base_y = float(sub.loc[sub["value"] == base_x, "stability_ARI"].iloc[0])
            ax.axvline(base_x, linestyle="--", alpha=0.5)
            ax.scatter([base_x], [base_y], color="red", zorder=5)

    title_stratum = f" ({args.stratum})" if args.stratum else ""
    fig.suptitle(f"SNF-lite Stability Sensitivity{title_stratum}  |  K={base['K']}  (vary one hyperparam at a time)")
    out_png = fig_dir / f"snf_sensitivity_{suffix}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
