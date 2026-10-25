"""Lightweight tests. Run: python tests/test_core.py   (or pytest if installed)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conformal import conformal_quantile, decide, merge_grade_groups  # noqa: E402
from src.load_clean import load_and_clean, merge_rare_sizes  # noqa: E402
from src.splits import leave_one_school_out  # noqa: E402


def test_quantile():
    s = np.arange(1, 10)  # n=9, alpha=.1 -> k=ceil(10*.9)=9 -> 9th smallest
    assert conformal_quantile(s, 0.1) == 9
    assert conformal_quantile(np.arange(1, 5), 0.1) == float("inf")  # k=5 > n=4


def test_decide():
    assert decide(3.0, 0.2, 1.0, 7)["decision"] == "AUTO"            # range 2.8-3.2 -> only size 3
    assert decide(3.0, 0.2, 1.0, 7)["size_index"] == 3
    assert decide(3.0, 1.0, 1.0, 7)["decision"] == "FLAG"            # 2-4 -> three sizes
    r = decide(3.5, 0.1, 1.0, 7)                                      # 3.4-3.6 -> no whole size
    assert r["decision"] == "AUTO" and r["n_in_range"] == 0
    assert decide(3.0, 0.2, float("inf"), 7)["decision"] == "FLAG"


def test_merges():
    assert merge_grade_groups({1: 50, 2: 10, 3: 50}, 30) == {1: 0, 2: 0, 3: 1}
    m, labels = merge_rare_sizes(np.array([5, 100, 100, 100, 4]), ["a", "b", "c", "d", "e"], 30)
    assert labels == ["<=b", "c", ">=d"] and list(m) == [0, 0, 1, 2, 2]


def test_no_leakage():
    df, _, _ = load_and_clean(verbose=False)
    for school, train, cal, test in leave_one_school_out(df):   # asserts inside
        assert school not in set(train.school_id) | set(cal.school_id)
        assert set(train.grade) == set(cal.grade) == {1, 2, 3, 4, 5}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
