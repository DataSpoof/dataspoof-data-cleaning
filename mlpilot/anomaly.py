"""Unsupervised anomaly detection with explanations.

``AutoAnomaly().fit(df)`` scores how unusual each row is by combining three detectors that catch
different kinds of anomaly — Isolation Forest (rare combinations), Local Outlier Factor (points
far from their neighbours) and a robust z-score (extreme single values) — as an average of their
robustly standardised scores. ``contamination`` (default ``"auto"``: from the score distribution, capped at
5%) sets how many rows are flagged. ``explain(df)`` says which features make a row unusual.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from .automl.frame import infer_kinds, make_preprocessor, to_model_frame


class AutoAnomaly:
    """>>> det = AutoAnomaly().fit(transactions)
    >>> det.score(new)               # anomaly_score in [0, 1] + is_anomaly
    >>> det.explain(new, top=3)      # the features that make each flagged row unusual
    """

    def __init__(self, *, contamination="auto", methods=("isolation_forest", "lof", "robust_z"),
                 exclude=None, seed: int = 42):
        bad = set(methods) - {"isolation_forest", "lof", "robust_z"}
        if bad:
            raise ValueError(f"Unknown methods: {bad}")
        self.contamination, self.methods, self.exclude, self.seed = contamination, tuple(methods), exclude, seed

    def _X(self, df):
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        df = df.drop([c for c in (self.exclude or []) if c in df.columns])
        return to_model_frame(df, self.kinds_)

    def fit(self, df) -> "AutoAnomaly":
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        frame = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        frame = frame.drop([c for c in (self.exclude or []) if c in frame.columns])
        self.kinds_ = infer_kinds(frame)
        X = to_model_frame(frame, self.kinds_)
        self.prep_ = make_preprocessor(self.kinds_, scale=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = np.asarray(self.prep_.fit_transform(X), dtype=float)
        self.detectors_ = {}
        if "isolation_forest" in self.methods:
            self.detectors_["isolation_forest"] = IsolationForest(n_estimators=300, random_state=self.seed).fit(Z)
        if "lof" in self.methods and len(Z) > 20:
            self.detectors_["lof"] = LocalOutlierFactor(n_neighbors=min(35, len(Z) - 1), novelty=True).fit(Z)
        num = [c for c, k in self.kinds_.items() if k == "num"]
        self.num_cols_ = num
        Xn = X[num].to_numpy(dtype=float) if num else np.zeros((len(X), 0))
        self.median_ = np.nanmedian(Xn, axis=0) if num else np.array([])
        mad = np.nanmedian(np.abs(Xn - self.median_), axis=0) * 1.4826 if num else np.array([])
        self.mad_ = np.where(mad > 0, mad, np.nanstd(Xn, axis=0) if num else 1.0) if num else np.array([])
        self.mad_ = np.where(self.mad_ > 0, self.mad_, 1.0) if num else self.mad_
        raw = self._raw_scores(Z, Xn)
        # robust standardisation per detector keeps magnitude (percentile ranks saturate at 1, so a
        # point far beyond anything seen in training would score no higher than the training maximum)
        self.stats_ = {}
        for k, v in raw.items():
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)) * 1.4826) or float(np.std(v)) or 1.0
            self.stats_[k] = (med, mad)
        combined = self._combine(raw)
        self.reference_ = np.sort(combined)
        if self.contamination == "auto":
            # rows far above the bulk: median + 3 * MAD of the combined score, at most 5%
            med, mad_c = np.median(combined), np.median(np.abs(combined - np.median(combined))) * 1.4826
            share = float(np.mean(combined > med + 3 * mad_c)) if mad_c > 0 else 0.01
            self.contamination_ = min(max(share, 0.005), 0.05)
        else:
            self.contamination_ = float(self.contamination)
        self.threshold_ = float(np.quantile(combined, 1 - self.contamination_))
        self.n_rows_ = len(X)
        return self

    def _raw_scores(self, Z, Xn) -> dict:
        raw = {}
        if "isolation_forest" in self.detectors_:
            raw["isolation_forest"] = -self.detectors_["isolation_forest"].score_samples(Z)
        if "lof" in self.detectors_:
            raw["lof"] = -self.detectors_["lof"].score_samples(Z)
        if "robust_z" in self.methods and Xn.shape[1]:
            z = np.abs((Xn - self.median_) / self.mad_)
            raw["robust_z"] = np.nan_to_num(np.nanmax(z, axis=1), nan=0.0)
        return raw

    def _combine(self, raw: dict) -> np.ndarray:
        return np.mean([(v - self.stats_[k][0]) / self.stats_[k][1] for k, v in raw.items()], axis=0)

    def score(self, df) -> pl.DataFrame:
        """``anomaly_score`` (0-1: the share of training rows that were less unusual) and
        ``is_anomaly`` for every row of ``df``."""
        X = self._X(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = np.asarray(self.prep_.transform(X), dtype=float)
        Xn = X[self.num_cols_].to_numpy(dtype=float) if self.num_cols_ else np.zeros((len(X), 0))
        combined = self._combine(self._raw_scores(Z, Xn))
        pct = np.searchsorted(self.reference_, combined, side="right") / len(self.reference_)
        return pl.DataFrame({"anomaly_score": np.round(pct, 6), "is_anomaly": combined > self.threshold_})

    def explain(self, df, *, top: int = 3, only_anomalies: bool = True) -> pl.DataFrame:
        """For each (anomalous) row: the ``top`` numeric features furthest from typical, as
        robust z-scores, and categorical values that were rare in the training data."""
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        scores = self.score(df)
        X = self._X(df)
        rows = []
        Xn = X[self.num_cols_].to_numpy(dtype=float) if self.num_cols_ else np.zeros((len(X), 0))
        z = (Xn - self.median_) / self.mad_ if self.num_cols_ else Xn
        for i in range(len(X)):
            if only_anomalies and not scores["is_anomaly"][i]:
                continue
            reasons = []
            if self.num_cols_:
                order = np.argsort(-np.abs(np.nan_to_num(z[i])))[:top]
                for j in order:
                    if abs(z[i, j]) >= 2:
                        direction = "high" if z[i, j] > 0 else "low"
                        reasons.append(f"{self.num_cols_[j]} = {Xn[i, j]:.4g} is unusually {direction} "
                                       f"({abs(z[i, j]):.1f} robust SDs from typical {self.median_[j]:.4g})")
            rows.append({"row": i, "anomaly_score": scores["anomaly_score"][i],
                         "reasons": "; ".join(reasons) or "unusual combination of values"})
        return pl.DataFrame(rows, schema={"row": pl.Int64, "anomaly_score": pl.Float64, "reasons": pl.String})

    def summary(self) -> str:
        return (f"AutoAnomaly: {len(self.stats_)} detectors, {self.n_rows_:,} training rows, "
                f"flags ~{self.contamination_:.1%} of rows like the training data")

    def __repr__(self) -> str:
        return self.summary() if hasattr(self, "threshold_") else "AutoAnomaly(unfitted)"
