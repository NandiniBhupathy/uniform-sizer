"""Inference: apply a saved bundle to new measurements -> AUTO size or FLAG (hand-fit)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .conformal import decide_many
from .load_clean import add_derived, valid_input_mask


def load_bundle(path):
    return joblib.load(path)


def predict_frame(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """df needs: grade, height_cm, weight_kg (age_years optional). Invalid rows are always FLAG."""
    df = df.copy().reset_index(drop=True)
    for c in ["grade", "height_cm", "weight_kg"]:
        if c not in df.columns:
            raise ValueError(f"input is missing column '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    labels = bundle["size_labels"]
    valid = valid_input_mask(df)
    d = add_derived(df)

    out = df.copy()
    out["decision"], out["size"], out["candidate_sizes"], out["flag_reason"] = "FLAG", "", "", ""
    out.loc[~valid, "flag_reason"] = "missing or impossible height/weight/grade"

    if valid.any():
        sub = d[valid]
        yhat, sigma = bundle["predictor"].predict(sub)
        res = decide_many(sub, yhat, sigma, bundle["conformal"], len(labels))
        for i, r in zip(sub.index, res.itertuples()):
            out.at[i, "yhat_size_index"] = r.yhat
            out.at[i, "sigma"] = r.sigma
            if r.decision == "AUTO":
                out.at[i, "decision"], out.at[i, "size"] = "AUTO", labels[r.size_index]
            else:
                lo, hi = r.lo, r.hi
                cands = [labels[s] for s in range(len(labels)) if lo <= s <= hi]
                out.at[i, "candidate_sizes"] = "|".join(cands)
                out.at[i, "flag_reason"] = ("not enough calibration data" if not np.isfinite(r.lo)
                                            else "several sizes plausible")
    return out


def mark_audit_sample(pred: pd.DataFrame, frac: float = 0.10, seed: int = 0) -> pd.DataFrame:
    """Randomly mark ~frac of AUTO students for a hand-fit check."""
    pred = pred.copy()
    pred["audit_hand_fit"] = False
    auto_idx = pred.index[pred["decision"] == "AUTO"]
    if len(auto_idx):
        rng = np.random.default_rng(seed)
        n = max(1, int(round(frac * len(auto_idx))))
        pred.loc[rng.choice(auto_idx, size=n, replace=False), "audit_hand_fit"] = True
    return pred


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Predict AUTO size / FLAG for new students.")
    ap.add_argument("--model", default=str(config.MODELS_DIR / "size_model.joblib"))
    ap.add_argument("--input", required=True, help="CSV with grade,height_cm,weight_kg[,age_years,student_id]")
    ap.add_argument("--output", default=str(config.RESULTS_DIR / "predictions.csv"))
    ap.add_argument("--audit-frac", type=float, default=0.10)
    a = ap.parse_args()
    pred = predict_frame(pd.read_csv(a.input), load_bundle(a.model))
    pred = mark_audit_sample(pred, a.audit_frac)
    pred.to_csv(a.output, index=False)
    n = len(pred)
    print(f"{n} students: AUTO={int((pred.decision == 'AUTO').sum())}, FLAG={int((pred.decision == 'FLAG').sum())}, "
          f"audit sample={int(pred.audit_hand_fit.sum())}  -> {a.output}")


if __name__ == "__main__":
    main()
