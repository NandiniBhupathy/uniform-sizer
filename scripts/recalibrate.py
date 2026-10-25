"""Yearly recalibration: add newly labelled students (final fitted sizes) to the calibration set
and recompute q. The point/error models are NOT retrained (use run_pipeline.py for that).

Usage: python scripts/recalibrate.py --new data/new_year.csv
       (same columns as students.csv: grade, height_cm, weight_kg, fitted_size, ...)
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.conformal import GradeConformal  # noqa: E402
from src.load_clean import add_derived, valid_input_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(config.MODELS_DIR / "size_model.joblib"))
ap.add_argument("--new", required=True)
ap.add_argument("--out", default=None, help="default: overwrite --model")
a = ap.parse_args()

b = joblib.load(a.model)
new = pd.read_csv(a.new, dtype={"fitted_size": str, "student_id": str})
for c in ["grade", "height_cm", "weight_kg"]:
    new[c] = pd.to_numeric(new[c], errors="coerce")
new["fitted_size"] = new["fitted_size"].str.strip()
new = new[new["fitted_size"].isin(b["orig_labels"]) & valid_input_mask(new)].copy()
orig_idx = new["fitted_size"].map({l: i for i, l in enumerate(b["orig_labels"])}).astype(int)
new["size_index"] = np.array(b["size_mapping"])[orig_idx.to_numpy()]
new = add_derived(new)
yhat, sigma = b["predictor"].predict(new)
add = new[["student_id", "grade", "size_index"]].copy()
add["yhat"], add["sigma"] = yhat, sigma

cal = pd.concat([b["cal_data"], add], ignore_index=True)
old_q = dict(b["conformal"].q_by_group)
b["conformal"] = GradeConformal(alpha=b["alpha"]).calibrate(cal, cal["yhat"].to_numpy(), cal["sigma"].to_numpy())
b["cal_data"] = cal
joblib.dump(b, a.out or a.model)
print(f"Added {len(add)} students (calibration set now {len(cal)}).")
print("q before:", {k: round(v, 3) for k, v in old_q.items()})
print("q after: ", {k: round(v, 3) for k, v in b['conformal'].q_by_group.items()})
