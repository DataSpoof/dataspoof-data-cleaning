"""AutoML for unlabelled data: choose a clustering method and number of clusters."""

from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np
import polars as pl

from .._detect import is_num
from ..core import Pipeline
from ._persist import load_artifact, save_artifact
from .frame import infer_kinds, make_preprocessor, to_model_frame

METHODS = ("kmeans", "gaussian_mixture")


def _model(method: str, k: int, seed: int):
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=10, random_state=seed)
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, random_state=seed)


class AutoCluster:
    """Pick the best clustering method and number of clusters by silhouette score.

    Tries each method in ``methods`` (k-means, Gaussian mixture) for every ``k`` in
    ``k_range`` on a sample, scores with the silhouette coefficient (also reporting
    Calinski–Harabasz and Davies–Bouldin), then refits the winner on all rows. Inputs are
    cleaned, dates expanded and categories one-hot encoded, and everything is standardised
    so no single column dominates the distances.
    """

    def __init__(self, *, k_range=(2, 8), methods=METHODS, preprocess="auto", sample: int = 5000,
                 seed: int = 42, verbose: bool = True):
        unknown = set(methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown method(s) {sorted(unknown)}; choose from {METHODS}")
        if k_range[0] < 2 or k_range[1] < k_range[0]:
            raise ValueError("k_range must be (low, high) with 2 <= low <= high")
        self.k_range, self.methods, self.preprocess = tuple(k_range), tuple(methods), preprocess
        self.sample, self.seed, self.verbose = sample, seed, verbose
        self.model_ = None

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _build_preprocess(self):
        if self.preprocess is None or isinstance(self.preprocess, Pipeline):
            return self.preprocess
        from ..cleaners import EncodeCategorical, default_cleaning_pipeline
        from ..features import DateTimeFeatures
        return Pipeline([*default_cleaning_pipeline().steps, DateTimeFeatures(),
                         EncodeCategorical(method="onehot", max_unique=15)], name="cluster-preprocess")

    def _matrix(self, df):
        X = df
        if self.preprocess_ is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                X = self.preprocess_.transform(df)
        return to_model_frame(X, self.feature_kinds_)

    def fit(self, df) -> "AutoCluster":
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
        from sklearn.pipeline import Pipeline as SkPipeline

        t0 = time.perf_counter()
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        self.preprocess_ = self._build_preprocess()
        prepared = df
        if self.preprocess_ is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = self.preprocess_.fit_transform(df)
        self.feature_kinds_ = infer_kinds(prepared)
        X = to_model_frame(prepared, self.feature_kinds_)
        transformer = make_preprocessor(self.feature_kinds_, scale=True)
        Z = transformer.fit_transform(X)
        rng = np.random.default_rng(self.seed)
        rows = rng.choice(len(Z), size=min(len(Z), self.sample), replace=False) if len(Z) > self.sample else np.arange(len(Z))
        Zs = Z[rows]
        self._log(f"AutoCluster: {len(X):,} rows x {Z.shape[1]} features | k {self.k_range[0]}-{self.k_range[1]} | {', '.join(self.methods)}")
        self.trials_ = []
        for method in self.methods:
            for k in range(self.k_range[0], min(self.k_range[1], len(Zs) - 1) + 1):
                start = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        labels = _model(method, k, self.seed).fit_predict(Zs)
                    if len(set(labels)) < 2:
                        raise ValueError("produced a single cluster")
                    trial = {"method": method, "k": k, "silhouette": float(silhouette_score(Zs, labels)),
                             "calinski_harabasz": float(calinski_harabasz_score(Zs, labels)),
                             "davies_bouldin": float(davies_bouldin_score(Zs, labels)), "error": None}
                except Exception as exc:
                    trial = {"method": method, "k": k, "silhouette": None, "calinski_harabasz": None,
                             "davies_bouldin": None, "error": str(exc)[:200]}
                trial["seconds"] = round(time.perf_counter() - start, 3)
                self.trials_.append(trial)
                if trial["error"] is None:
                    self._log(f"  {method:<17} k={k:<3} silhouette = {trial['silhouette']:.4f}")
        ok = [t for t in self.trials_ if t["error"] is None]
        if not ok:
            raise RuntimeError("No clustering succeeded: " + "; ".join(t["error"] for t in self.trials_))
        self.best_ = max(ok, key=lambda t: (t["silhouette"], -t["k"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = SkPipeline([("prep", make_preprocessor(self.feature_kinds_, scale=True)),
                                      ("model", _model(self.best_["method"], self.best_["k"], self.seed))]).fit(X)
        self.n_rows_, self.n_features_ = len(X), Z.shape[1]
        self.labels_ = self.predict(df)                 # one label per input row
        self.fit_seconds_ = round(time.perf_counter() - t0, 2)
        self._log(f"Best: {self.best_['method']} with {self.best_['k']} clusters "
                  f"(silhouette {self.best_['silhouette']:.4f}) in {self.fit_seconds_}s")
        return self

    def predict(self, df) -> pl.Series:
        """Cluster label (0..k-1) for each row."""
        if self.model_ is None:
            raise RuntimeError("AutoCluster is not fitted. Call fit() first or AutoCluster.load().")
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        return pl.Series("cluster", self.model_.predict(self._matrix(df)).astype(int))

    def cluster_profiles(self, df) -> pl.DataFrame:
        """Size, share and the mean of every numeric column per cluster."""
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        labelled = df.with_columns(self.predict(df))
        numeric = [c for c, t in df.schema.items() if is_num(t)]
        return (labelled.group_by("cluster")
                .agg([pl.len().alias("size"), *[pl.col(c).mean().alias(f"{c}_mean") for c in numeric]])
                .with_columns((pl.col("size") / labelled.height).alias("share"))
                .sort("cluster"))

    @property
    def leaderboard_(self) -> pl.DataFrame:
        return pl.DataFrame(self.trials_, infer_schema_length=None).sort(
            "silhouette", descending=True, nulls_last=True)

    def summary(self) -> str:
        b = self.best_
        return (f"AutoCluster: {self.n_rows_:,} rows x {self.n_features_} features | "
                f"{len(self.trials_)} trials in {self.fit_seconds_}s\n"
                f"  Best: {b['method']} with {b['k']} clusters | silhouette {b['silhouette']:.4f} | "
                f"Calinski-Harabasz {b['calinski_harabasz']:.1f} | Davies-Bouldin {b['davies_bouldin']:.3f}")

    def __repr__(self) -> str:
        return self.summary() if self.model_ is not None else "AutoCluster(unfitted)"

    def save(self, path):
        if self.model_ is None:
            raise RuntimeError("AutoCluster is not fitted")
        manifest = {"feature_kinds": self.feature_kinds_, "best": self.best_, "trials": self.trials_,
                    "n_rows": self.n_rows_, "n_features": self.n_features_, "fit_seconds": self.fit_seconds_}
        return save_artifact(path, "clustering", self.model_, manifest, self.preprocess_)

    @classmethod
    def load(cls, path) -> "AutoCluster":
        manifest, model, preprocess = load_artifact(path, "clustering")
        obj = cls(verbose=False)
        obj.model_, obj.preprocess_ = model, preprocess
        obj.feature_kinds_, obj.best_, obj.trials_ = manifest["feature_kinds"], manifest["best"], manifest["trials"]
        obj.n_rows_, obj.n_features_, obj.fit_seconds_ = manifest["n_rows"], manifest["n_features"], manifest["fit_seconds"]
        return obj
