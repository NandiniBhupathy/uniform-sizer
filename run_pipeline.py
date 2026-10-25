"""End-to-end: clean -> leave-one-school-out evaluation of all configs -> pick final -> train & save.

Usage:
    python run_pipeline.py                       # uses data/students.csv, alpha=0.10
    python run_pipeline.py --data my.csv --alpha 0.1
"""
import argparse
import json

import joblib
import pandas as pd

from src import config
from src.conformal import GradeConformal, decide_many
from src.evaluate import (add_build_band, grouped_metrics, metrics, select_config,
                          summarise_folds)
from src.load_clean import load_and_clean
from src.models import SizePredictor
from src.plots import explore, results_plots
from src.splits import leave_one_school_out, split_train_cal


def run_fold(train, cal, test, model, fset, alpha, n_sizes):
    pred = SizePredictor(model, fset).fit(train)
    conf = GradeConformal(alpha=alpha).calibrate(cal, *pred.predict(cal))
    yhat, sigma = pred.predict(test)
    res = decide_many(test, yhat, sigma, conf, n_sizes)
    return pred, conf, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_PATH))
    ap.add_argument("--alpha", type=float, default=config.ALPHA)
    a = ap.parse_args()

    for d in (config.RESULTS_DIR, config.PLOTS_DIR, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. clean ------------------------------------------------------------
    df, labels, report = load_and_clean(a.data)
    (config.RESULTS_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    explore(df, labels)
    n_sizes = len(labels)
    print(f"\n{len(df)} clean students, {df.school_id.nunique()} schools, sizes: {labels}")

    # 2-7. leave-one-school-out over all model x feature-set configs ---------
    configs = [(m, f) for m in config.MODEL_NAMES for f in config.FEATURE_SETS
               if not (m == "lookup" and f != "height")]
    fold_rows, grade_rows, band_rows = [], [], []
    for school, train, cal, test in leave_one_school_out(df):
        print(f"\n== held-out school {school}: train={len(train)} cal={len(cal)} test={len(test)}")
        bands = add_build_band(train, test)
        for model, fset in configs:
            _, _, res = run_fold(train, cal, test, model, fset, a.alpha, n_sizes)
            tag = dict(school=school, model=model, feature_set=fset)
            fold_rows.append({**tag, **metrics(test, res)})
            grade_rows += [{**tag, **r} for r in grouped_metrics(test, res, test["grade"].rename("grade")).to_dict("records")]
            band_rows += [{**tag, **r} for r in grouped_metrics(test, res, bands).to_dict("records")]
    fold_df, by_grade, by_band = map(pd.DataFrame, (fold_rows, grade_rows, band_rows))
    fold_df.to_csv(config.RESULTS_DIR / "metrics_per_school.csv", index=False)
    by_grade.to_csv(config.RESULTS_DIR / "metrics_by_grade.csv", index=False)
    by_band.to_csv(config.RESULTS_DIR / "metrics_by_build_band.csv", index=False)

    summary = summarise_folds(fold_df)
    summary.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Mean over held-out schools (alpha=%.2f) ===" % a.alpha)
    print(summary.round(3).to_string(index=False))

    # 8. choose final config -------------------------------------------------
    model, fset, best = select_config(summary, a.alpha)
    print(f"\nBest auto-rate config: {best['model']} / {best['feature_set']}")
    print(f"Selected (simplest within school-to-school noise): {model} / {fset}")
    results_plots(fold_df, by_grade, by_band, model, fset, a.alpha)

    # alpha sweep for the selected config
    sweep = []
    for al in config.ALPHA_SWEEP:
        for school, train, cal, test in leave_one_school_out(df):
            _, _, res = run_fold(train, cal, test, model, fset, al, n_sizes)
            sweep.append({"alpha": al, "target_coverage": 1 - al, "school": school, **metrics(test, res)})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(config.RESULTS_DIR / "alpha_sweep_per_school.csv", index=False)
    sw = sweep_df.groupby(["alpha", "target_coverage"])[["coverage", "auto_rate", "exact_among_auto",
                                                         "within1_among_auto"]].mean().reset_index()
    sw.to_csv(config.RESULTS_DIR / "alpha_sweep_summary.csv", index=False)
    print("\n=== Alpha sweep (selected config, mean over schools) ===")
    print(sw.round(3).to_string(index=False))

    # final model: all 3 schools, calibrate on held-out sections ------------
    train, cal = split_train_cal(df)
    pred = SizePredictor(model, fset).fit(train)
    yhat_c, sigma_c = pred.predict(cal)
    conf = GradeConformal(alpha=a.alpha).calibrate(cal, yhat_c, sigma_c)
    cal_store = cal[["student_id", "grade", "size_index"]].copy()
    cal_store["yhat"], cal_store["sigma"] = yhat_c, sigma_c
    bundle = dict(predictor=pred, conformal=conf, size_labels=labels, alpha=a.alpha,
                  model_name=model, feature_set=fset, cal_data=cal_store,
                  orig_labels=report["orig_labels"], size_mapping=report["size_mapping"])
    joblib.dump(bundle, config.MODELS_DIR / "size_model.joblib")
    (config.MODELS_DIR / "final_config.json").write_text(json.dumps({
        "model": model, "feature_set": fset, "alpha": a.alpha, "size_labels": labels,
        "grade_to_group": {int(k): int(v) for k, v in conf.grade_to_group.items()},
        "q_by_group": conf.q_by_group, "n_train": len(train), "n_cal": len(cal)}, indent=2))
    print(f"\nSaved models/size_model.joblib  (q by grade group: {conf.q_by_group})")
    print("Results in results/, plots in results/plots/")


if __name__ == "__main__":
    main()
