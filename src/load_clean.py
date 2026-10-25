"""Load students.csv, validate, clean, and map sizes to ordered indices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_size_chart(path=config.SIZE_CHART_PATH):
    """Return (ordered labels as str list, {label: index})."""
    chart = pd.read_csv(path, dtype={"size_label": str}).sort_values("size_index")
    labels = chart["size_label"].tolist()
    return labels, {lab: i for i, lab in enumerate(labels)}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """bmi, age_years (fallback grade+5.5) and section_key. Works on raw or clean frames."""
    df = df.copy()
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
    if "age_years" not in df.columns:
        df["age_years"] = np.nan
    df["age_years"] = df["age_years"].fillna(df["grade"] + 5.5)
    if {"school_id", "section"} <= set(df.columns):
        df["section_key"] = df["school_id"].astype(str) + "_" + df["section"].astype(str)
    return df


def valid_input_mask(df: pd.DataFrame) -> pd.Series:
    """True where height/weight/grade are present and physically plausible."""
    h, w = df["height_cm"], df["weight_kg"]
    ok = h.between(*config.HEIGHT_RANGE) & w.between(*config.WEIGHT_RANGE)
    return ok & df["grade"].notna()


def merge_rare_sizes(counts: np.ndarray, labels: list[str], min_n: int):
    """Merge sizes with < min_n students into the neighbouring size (smaller neighbour first).

    Returns (orig_index -> new_index array, new_labels).
    """
    groups = [[i] for i in range(len(labels))]
    sizes = [int(c) for c in counts]
    while len(groups) > 1:
        small = [g for g in range(len(groups)) if sizes[g] < min_n]
        if not small:
            break
        g = min(small, key=lambda k: sizes[k])
        nbrs = [k for k in (g - 1, g + 1) if 0 <= k < len(groups)]
        k = min(nbrs, key=lambda j: sizes[j])
        lo, hi = min(g, k), max(g, k)
        groups[lo] = groups[lo] + groups[hi]
        sizes[lo] += sizes[hi]
        del groups[hi], sizes[hi]
    mapping = np.zeros(len(labels), dtype=int)
    new_labels = []
    for gi, members in enumerate(groups):
        for m in members:
            mapping[m] = gi
        if len(members) == 1:
            new_labels.append(labels[members[0]])
        elif members[0] == 0:
            new_labels.append(f"<={labels[members[-1]]}")
        elif members[-1] == len(labels) - 1:
            new_labels.append(f">={labels[members[0]]}")
        else:
            new_labels.append(f"{labels[members[0]]}-{labels[members[-1]]}")
    return mapping, new_labels


def load_and_clean(path=config.DATA_PATH, size_chart_path=config.SIZE_CHART_PATH,
                   merge_rare=True, verbose=True):
    """Return (clean_df, size_labels, report).

    clean_df has: size_index (int, after rare-size merging), bmi, age_years, section_key, ...
    Rows with missing/impossible height or weight, unknown sizes, or duplicates are dropped
    (and counted in `report`) so every feature set is compared on the same students.
    """
    report: dict = {}
    raw = pd.read_csv(path, dtype={"student_id": str, "school_id": str,
                                   "section": str, "fitted_size": str})
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"students.csv is missing required columns: {missing_cols}")
    report["rows_raw"] = len(raw)

    df = raw.drop_duplicates().copy()
    df = df.drop_duplicates(subset=["student_id", "year"], keep="first")
    report["dropped_duplicates"] = report["rows_raw"] - len(df)

    for c in ["grade", "height_cm", "weight_kg", "year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fitted_size"] = df["fitted_size"].astype(str).str.strip()

    labels, label_to_idx = load_size_chart(size_chart_path)
    known = df["fitted_size"].isin(label_to_idx)
    report["dropped_unknown_size"] = int((~known).sum())
    df = df[known].copy()

    ok = valid_input_mask(df)
    report["dropped_missing_or_impossible_inputs"] = int((~ok).sum())
    df = df[ok].copy()

    df["size_index_orig"] = df["fitted_size"].map(label_to_idx).astype(int)
    counts = np.bincount(df["size_index_orig"], minlength=len(labels))
    report["size_counts_before_merge"] = dict(zip(labels, counts.tolist()))
    if merge_rare:
        mapping, new_labels = merge_rare_sizes(counts, labels, config.MIN_SIZE_N)
    else:
        mapping, new_labels = np.arange(len(labels)), labels
    df["size_index"] = mapping[df["size_index_orig"].to_numpy()]
    report["size_labels_used"] = new_labels
    report["orig_labels"] = labels
    report["size_mapping"] = mapping.tolist()   # original index -> modelling index
    report["rows_clean"] = len(df)

    df = add_derived(df).reset_index(drop=True)
    if verbose:
        skip = {"size_counts_before_merge", "size_mapping", "orig_labels"}
        print("[load_clean]", {k: v for k, v in report.items() if k not in skip})
    return df, new_labels, report
