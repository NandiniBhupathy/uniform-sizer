"""Diagnostic plots (saved to results/plots)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402


def explore(df, labels, outdir=config.PLOTS_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, len(df))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(df["height_cm"], df["size_index"] + jitter, s=5, alpha=0.4)
    ax[0].set_yticks(range(len(labels)), labels)
    ax[0].set(xlabel="height (cm)", ylabel="fitted size", title="Size vs height")
    band = df[(df["height_cm"] >= 125) & (df["height_cm"] < 130)]
    ax[1].scatter(band["weight_kg"], band["size_index"] + jitter[:len(band)], s=8, alpha=0.5)
    ax[1].set_yticks(range(len(labels)), labels)
    ax[1].set(xlabel="weight (kg)", title="Size vs weight, height 125-130 cm\n(upward trend = weight matters)")
    fig.tight_layout()
    fig.savefig(outdir / "explore_size_vs_height_weight.png", dpi=130)
    plt.close(fig)


def results_plots(fold_df, by_grade, by_band, model, fset, alpha, outdir=config.PLOTS_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    sel = fold_df[(fold_df["model"] == model) & (fold_df["feature_set"] == fset)]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(sel["school"], sel["coverage"], label="coverage")
    ax[0].plot(sel["school"], sel["auto_rate"], "ro-", label="auto rate")
    ax[0].axhline(1 - alpha, color="k", ls="--", lw=1)
    ax[0].set(ylim=(0, 1.05), title=f"Held-out school ({model}, {fset})")
    ax[0].legend()

    g = by_grade[(by_grade["model"] == model) & (by_grade["feature_set"] == fset)]
    g = g.groupby("grade")[["coverage", "auto_rate"]].mean()
    ax[1].bar(g.index.astype(str), g["coverage"], label="coverage")
    ax[1].plot(g.index.astype(str), g["auto_rate"], "ro-", label="auto rate")
    ax[1].axhline(1 - alpha, color="k", ls="--", lw=1)
    ax[1].set(ylim=(0, 1.05), xlabel="grade", title="By grade (mean over schools)")

    b = by_band[(by_band["model"] == model) & (by_band["feature_set"] == fset)]
    b = b.groupby("build_band")[["coverage", "flag_rate", "mae_point"]].mean().reindex(["lean", "typical", "heavy"])
    ax[2].bar(b.index, b["flag_rate"], color="tab:orange", label="flag rate")
    ax[2].plot(b.index, b["mae_point"], "ko-", label="point MAE (sizes)")
    ax[2].set(title="By weight-for-height band")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(outdir / "results_selected_config.png", dpi=130)
    plt.close(fig)
