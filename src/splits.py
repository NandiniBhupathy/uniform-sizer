"""Leave-one-school-out splits with section-level train/calibration separation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def split_train_cal(df: pd.DataFrame, cal_fraction: float = config.CAL_FRACTION,
                    seed: int = config.RANDOM_STATE):
    """Assign WHOLE sections to train or calibration, spread evenly across grades.

    Returns (train_df, cal_df). Calibration rows whose student_id also appears in train
    (e.g. same girl in two years) are dropped from calibration to avoid leakage.
    """
    rng = np.random.default_rng(seed)
    # a section_key could in principle span grades; use its modal grade
    sections = (df.groupby("section_key")["grade"].agg(lambda s: s.mode().iloc[0]).reset_index())
    cal_keys: set[str] = set()
    for _, g in sections.groupby("grade"):
        keys = g["section_key"].to_numpy().copy()
        rng.shuffle(keys)
        n_cal = int(round(cal_fraction * len(keys)))
        if len(keys) >= 2:
            n_cal = min(max(n_cal, 1), len(keys) - 1)
        cal_keys.update(keys[:n_cal])
    is_cal = df["section_key"].isin(cal_keys)
    train, cal = df[~is_cal].copy(), df[is_cal].copy()
    cal = cal[~cal["student_id"].isin(set(train["student_id"]))]
    return train, cal


def leave_one_school_out(df: pd.DataFrame, seed: int = config.RANDOM_STATE):
    """Yield (held_out_school, train_df, cal_df, test_df) for each school."""
    for school in sorted(df["school_id"].unique()):
        test = df[df["school_id"] == school].copy()
        rest = df[df["school_id"] != school]
        train, cal = split_train_cal(rest, seed=seed)
        assert_no_leakage(train, cal, test)
        yield school, train, cal, test


def assert_no_leakage(train, cal, test):
    """Train / calibration / test must not share a school (test), a section or a student."""
    assert set(test["school_id"]).isdisjoint(train["school_id"]), "test school appears in train"
    assert set(test["school_id"]).isdisjoint(cal["school_id"]), "test school appears in cal"
    assert set(train["section_key"]).isdisjoint(cal["section_key"]), "section shared train/cal"
    for a, b, name in [(train, cal, "train/cal"), (train, test, "train/test"), (cal, test, "cal/test")]:
        assert set(a["student_id"]).isdisjoint(b["student_id"]), f"student shared {name}"
