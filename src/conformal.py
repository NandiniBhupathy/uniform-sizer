"""Normalised split-conformal prediction with grade-wise calibration, plus the AUTO/FLAG rule."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """k-th smallest score with k = ceil((n+1)(1-alpha)); +inf if k > n (too few samples)."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n or n == 0 else float(scores[k - 1])


def merge_grade_groups(counts: dict[int, int], min_n: int) -> dict[int, int]:
    """Map each grade -> group id; merge grades with < min_n calibration students into a neighbour."""
    grades = sorted(counts)
    groups = [[g] for g in grades]
    sizes = [counts[g] for g in grades]
    while len(groups) > 1 and min(sizes) < min_n:
        i = int(np.argmin(sizes))
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        j = min(nbrs, key=lambda k: sizes[k])
        lo, hi = min(i, j), max(i, j)
        groups[lo] += groups[hi]
        sizes[lo] += sizes[hi]
        del groups[hi], sizes[hi]
    return {g: gi for gi, members in enumerate(groups) for g in members}


@dataclass
class GradeConformal:
    alpha: float = config.ALPHA
    min_group_n: int = config.MIN_GROUP_N
    grade_to_group: dict = field(default_factory=dict)
    q_by_group: dict = field(default_factory=dict)

    def calibrate(self, cal: pd.DataFrame, yhat, sigma):
        scores = np.abs(cal["size_index"].to_numpy(float) - yhat) / sigma
        grades = cal["grade"].astype(int).to_numpy()
        counts = pd.Series(grades).value_counts().to_dict()
        self.grade_to_group = merge_grade_groups({int(k): int(v) for k, v in counts.items()},
                                                 self.min_group_n)
        groups = np.array([self.grade_to_group[g] for g in grades])
        self.q_by_group = {int(gr): conformal_quantile(scores[groups == gr], self.alpha)
                           for gr in np.unique(groups)}
        return self

    def q_for(self, grade) -> float:
        """q for a grade; unseen grade -> nearest calibrated grade; unusable -> inf (FLAG)."""
        if grade is None or (isinstance(grade, float) and np.isnan(grade)):
            return float("inf")
        known = np.array(sorted(self.grade_to_group))
        g = int(known[np.argmin(np.abs(known - int(grade)))])
        return self.q_by_group[self.grade_to_group[g]]


def decide(yhat, sigma, q, n_sizes: int):
    """Apply the AUTO/FLAG rule for ONE student. Returns dict(decision, size_index, lo, hi, n_sizes_in_range).

    range = yhat +/- q*sigma; whole sizes inside the range:
      exactly 1 -> AUTO that size;  >=2 -> FLAG;  0 (range between sizes) -> AUTO nearest size.
    q = inf (not enough calibration data) -> FLAG.
    """
    if not np.isfinite(q):
        return dict(decision="FLAG", size_index=None, lo=-np.inf, hi=np.inf, n_in_range=n_sizes)
    lo, hi = yhat - q * sigma, yhat + q * sigma
    inside = [s for s in range(n_sizes) if lo <= s <= hi]
    if len(inside) == 1:
        return dict(decision="AUTO", size_index=inside[0], lo=lo, hi=hi, n_in_range=1)
    if len(inside) == 0:
        nearest = int(np.clip(round(yhat), 0, n_sizes - 1))
        return dict(decision="AUTO", size_index=nearest, lo=lo, hi=hi, n_in_range=0)
    return dict(decision="FLAG", size_index=None, lo=lo, hi=hi, n_in_range=len(inside))


def decide_many(df: pd.DataFrame, yhat, sigma, conf: GradeConformal, n_sizes: int) -> pd.DataFrame:
    out = []
    for g, yh, s in zip(df["grade"].to_numpy(), yhat, sigma):
        out.append(decide(yh, s, conf.q_for(g), n_sizes))
    res = pd.DataFrame(out, index=df.index)
    res["yhat"], res["sigma"] = yhat, sigma
    return res
