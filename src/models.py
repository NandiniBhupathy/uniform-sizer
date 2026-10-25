"""Point models (size index) and the error model (sigma-hat)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from . import config


class LookupModel:
    """Baseline: median size index per height band (5 cm bands). Height-only."""

    def __init__(self, band_cm: float = 5.0):
        self.band_cm = band_cm

    def _band(self, h):
        return np.floor(np.asarray(h, dtype=float) / self.band_cm).astype(int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        bands = self._band(X[:, 0])
        s = pd.Series(np.asarray(y, dtype=float)).groupby(bands).median()
        self.table_ = s.to_dict()
        self.fallback_ = float(np.median(y))
        # linear fallback for bands with no training data (extrapolation)
        self.lin_ = LinearRegression().fit(X[:, :1], y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        bands = self._band(X[:, 0])
        lin = self.lin_.predict(X[:, :1])
        return np.array([self.table_.get(b, l) for b, l in zip(bands, lin)], dtype=float)

    def get_params(self, deep=True):
        return {"band_cm": self.band_cm}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


def make_point_model(name: str):
    if name == "lookup":
        return LookupModel()
    if name == "linear":
        return LinearRegression()
    if name == "gbm":
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                         subsample=0.8, random_state=config.RANDOM_STATE)
    raise ValueError(f"unknown model {name}")


def make_sigma_model():
    """Small model that predicts |residual| from the same inputs."""
    return GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=config.RANDOM_STATE)


def feature_matrix(df: pd.DataFrame, feature_set: str) -> np.ndarray:
    return df[config.FEATURE_SETS[feature_set]].to_numpy(dtype=float)


class SizePredictor:
    """Point model + error model for a given feature set.

    The error model is trained on OUT-OF-FOLD residuals (grouped by section) so it sees
    realistic, not over-fitted, errors.
    """

    def __init__(self, model_name: str, feature_set: str):
        if model_name == "lookup" and feature_set != "height":
            raise ValueError("lookup baseline only supports the 'height' feature set")
        self.model_name, self.feature_set = model_name, feature_set

    def fit(self, train: pd.DataFrame):
        X, y = feature_matrix(train, self.feature_set), train["size_index"].to_numpy(float)
        groups = train["section_key"].to_numpy()
        self.point_ = make_point_model(self.model_name).fit(X, y)

        n_splits = min(config.N_OOF_FOLDS, len(np.unique(groups)))
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = make_point_model(self.model_name).fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
        self.sigma_ = make_sigma_model().fit(X, np.abs(y - oof))
        return self

    def predict(self, df: pd.DataFrame):
        """Return (y_hat, sigma_hat) in size-index units."""
        X = feature_matrix(df, self.feature_set)
        yhat = self.point_.predict(X)
        sigma = np.maximum(self.sigma_.predict(X), config.SIGMA_FLOOR)
        return yhat, sigma
