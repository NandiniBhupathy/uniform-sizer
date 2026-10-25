"""Metrics, build bands, and the 'simplest model within noise of the best' selection rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config


def add_build_band(train: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Weight-for-height band: 'lean' / 'typical' / 'heavy' (+-1 SD of residual from a train-fit line)."""
    lin = LinearRegression().fit(train[["height_cm"]], train["weight_kg"])
    sd = float(np.std(train["weight_kg"] - lin.predict(train[["height_cm"]])))
    resid = df["weight_kg"] - lin.predict(df[["height_cm"]])
    return pd.Series(np.where(resid < -sd, "lean", np.where(resid > sd, "heavy", "typical")),
                     index=df.index, name="build_band")


def metrics(test: pd.DataFrame, res: pd.DataFrame) -> dict:
    """test must align with res (same index). All sizes are indices."""
    y = test["size_index"].to_numpy(float)
    covered = (res["lo"].to_numpy() <= y) & (y <= res["hi"].to_numpy())
    auto = (res["decision"] == "AUTO").to_numpy()
    err = np.abs(res["size_index"].astype(float).to_numpy() - y)
    n_auto = int(auto.sum())
    return {
        "n": int(len(test)),
        "coverage": float(covered.mean()),
        "auto_rate": float(auto.mean()),
        "flag_rate": float(1 - auto.mean()),
        "exact_among_auto": float((err[auto] == 0).mean()) if n_auto else np.nan,
        "within1_among_auto": float((err[auto] <= 1).mean()) if n_auto else np.nan,
        "mean_sizes_in_range": float(res["n_in_range"].mean()),
        "mae_point": float(np.mean(np.abs(res["yhat"].to_numpy() - y))),
    }


def grouped_metrics(test: pd.DataFrame, res: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    rows = []
    for key in sorted(by.unique()):
        m = (by == key).to_numpy()
        d = metrics(test[m], res[m])
        d[by.name] = key
        rows.append(d)
    return pd.DataFrame(rows)


def summarise_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across held-out schools, per (model, feature_set)."""
    cols = ["coverage", "auto_rate", "exact_among_auto", "within1_among_auto", "flag_rate"]
    g = fold_df.groupby(["model", "feature_set"])[cols]
    mean, std = g.mean().add_suffix("_mean"), g.std(ddof=0).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def select_config(summary: pd.DataFrame, alpha: float = config.ALPHA, cov_slack: float = 0.02):
    """Pick the simplest (model, feature_set) whose auto-rate is within one school-to-school
    SD of the best auto-rate, among configs whose mean coverage is near the target."""
    s = summary.copy()
    s["complexity"] = s["model"].map(config.MODEL_RANK) + s["feature_set"].map(config.FEATURE_RANK)
    ok = s[s["coverage_mean"] >= (1 - alpha) - cov_slack]
    if ok.empty:
        ok = s[s["coverage_mean"] == s["coverage_mean"].max()]
    best = ok.loc[ok["auto_rate_mean"].idxmax()]
    # "within noise of the best" on BOTH auto-rate and exact-match-among-AUTO, using the
    # school-to-school SD, so a cheap model can't win by being conservative and wrong more often.
    cand = ok[(ok["auto_rate_mean"] >= best["auto_rate_mean"] - best["auto_rate_std"])
              & (ok["exact_among_auto_mean"] >= best["exact_among_auto_mean"] - best["exact_among_auto_std"])]
    choice = cand.sort_values(["complexity", "auto_rate_mean"], ascending=[True, False]).iloc[0]
    return str(choice["model"]), str(choice["feature_set"]), best
