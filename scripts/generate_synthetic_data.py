"""Generate SYNTHETIC student data (NOT real). Replace data/students.csv with real data later.

Simulates: 3 schools x 5 grades x 4 sections x ~50 girls = ~3,000 rows.
The 'fitted size' is generated from height and weight with heteroscedastic noise
(atypical builds -> noisier fits) plus small school/section effects, then mapped
to the labels in data/size_chart.csv. A few dirty rows are injected so the cleaning
step has something to do.

Usage:  python scripts/generate_synthetic_data.py [--seed 7] [--out data/students.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


def generate(seed: int = 7, per_section: int = 50, year: int = 2025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chart = pd.read_csv(config.SIZE_CHART_PATH)
    labels = chart.sort_values("size_index")["size_label"].tolist()
    n_sizes = len(labels)

    school_shift = {"A": 0.00, "B": 0.15, "C": -0.10}   # schools differ a bit (tailor / cohort)
    rows = []
    sid = 0
    for school, shift in school_shift.items():
        for grade in range(1, 6):
            for sec in "ABCD":
                section_shift = rng.normal(0, 0.05)
                n = int(rng.integers(per_section - 4, per_section + 5))
                age = grade + 5 + rng.uniform(0, 1, n)                 # 6-11 years
                height = 115 + 5.5 * (age - 6) + rng.normal(0, 5.5, n)
                w_exp = 0.6 * height - 49
                weight = w_exp * np.exp(rng.normal(0, 0.12, n))        # skewed build variation
                dev = weight - w_exp
                sd = 0.18 + 0.04 * np.abs(dev)                          # atypical builds -> noisier
                latent = 0.13 * (height - 105) + 0.10 * dev + shift + section_shift + rng.normal(0, sd)
                idx = np.clip(np.round(latent), 0, n_sizes - 1).astype(int)
                for i in range(n):
                    sid += 1
                    rows.append({
                        "student_id": f"S{sid:05d}",
                        "school_id": school,
                        "grade": grade,
                        "section": f"{grade}{sec}",
                        "age_years": round(float(age[i]), 1),
                        "height_cm": round(float(height[i]) * 2) / 2,
                        "weight_kg": round(float(weight[i]) * 2) / 2,
                        "fitted_size": labels[idx[i]],
                        "year": year,
                    })
    df = pd.DataFrame(rows)

    # ---- inject dirt (so load_clean.py is exercised) ----
    miss = rng.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[miss, "weight_kg"] = np.nan                                   # missing weights = empty
    bad = rng.choice(df.index, size=4, replace=False)
    df.loc[bad[:2], "height_cm"] = [12.8, 1285.0]                       # impossible heights (typos)
    df.loc[bad[2:], "weight_kg"] = [0.0, 260.0]                         # impossible weights
    dups = df.sample(5, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True)                        # duplicate rows
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(config.DATA_PATH))
    args = ap.parse_args()
    df = generate(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {args.out}")
    print(df["fitted_size"].value_counts().sort_index().to_string())
