"""Check the hand-fit audit of AUTO students. If accuracy is below target -> fall back to manual.

Input CSV: output of predict.py with an extra column 'final_fitted_size' filled in for the rows where
audit_hand_fit is True. Usage: python scripts/audit_check.py --input results/predictions_audited.csv --target 0.85
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--target", type=float, default=0.85, help="min exact-match rate among audited AUTO students")
a = ap.parse_args()

d = pd.read_csv(a.input, dtype={"size": str, "final_fitted_size": str})
d = d[(d["decision"] == "AUTO") & d["audit_hand_fit"] & d["final_fitted_size"].notna()]
if d.empty:
    raise SystemExit("No audited AUTO rows with final_fitted_size filled in.")
acc = (d["size"] == d["final_fitted_size"]).mean()
print(f"Audited AUTO students: {len(d)}   exact-match: {acc:.1%}   target: {a.target:.0%}")
if acc < a.target:
    print("BELOW TARGET -> switch to manual fitting until you recalibrate / retrain.")
else:
    print("OK -> keep using AUTO.")
