"""AutoML for supervised problems: classification and regression, including time series."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import time
import warnings

import numpy as np
import polars as pl

from ..core import AutomatedCleaningWarning, Pipeline
from ..features.leakage import LeakageError, detect_leakage
from ._persist import load_artifact, save_artifact
from .frame import infer_kinds, make_preprocessor, to_model_frame
from .models import model_specs, sample_params
from .tasks import TaskInfo, detect_task

DEFAULT_METRICS = {"binary": "roc_auc", "multiclass": "f1_macro",
                   "regression": "neg_root_mean_squared_error"}
_FAMILY_WEIGHTS = (0.5, 0.3, 0.2)


def _as_polars(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


_SHORT_NAMES = {"neg_root_mean_squared_error": "rmse", "neg_mean_squared_error": "mse",
                "neg_mean_absolute_error": "mae", "neg_median_absolute_error": "median_ae",
                "neg_mean_absolute_percentage_error": "mape", "neg_log_loss": "log_loss"}


def _metric_label(metric: str) -> str:
    return _SHORT_NAMES.get(metric, metric[4:] if metric.startswith("neg_") else metric)


def _shown(metric: str, score: float) -> float:
    """Scores are 'higher is better' internally; show error metrics as positive numbers."""
    return -score if metric.startswith("neg_") else score


class AutoML:
    """Automatic model selection, tuning and ensembling for one target column.

    ``AutoML(target).fit(df)``:

    1. detects the task (binary / multiclass / regression; time series when rows are ordered
       by a date column) and picks a metric (ROC-AUC, macro-F1, RMSE) unless ``metric`` is set;
    2. checks for target leakage and drops leaky columns (``on_leakage``);
    3. cleans and engineers features with a target-aware pipeline (``preprocess="auto"``:
       ``default_cleaning_pipeline`` + ``default_feature_pipeline``), or yours, or none;
    4. cross-validates a baseline and every candidate model with sensible defaults, then
       spends the rest of ``time_budget`` on random hyperparameter search around the best
       model families (stratified K-fold; time-ordered splits for time series);
    5. tries a soft-voting ensemble of the best three model families;
    6. refits the winner on all training rows.

    Then ``predict``, ``predict_proba``, ``evaluate``, ``feature_importance``, ``save``/``load``.
    """

    def __init__(self, target: str, *, task: str = "auto", time_col="auto", metric: str | None = None,
                 time_budget: float = 60.0, max_trials: int | None = None, cv: int = 5,
                 preprocess="auto", include=None, exclude=None, ensemble: bool = True,
                 on_leakage: str = "drop", search_sample: int = 50_000, seed: int = 42,
                 verbose: bool = True):
        if task not in ("auto", "binary", "multiclass", "regression", "classification"):
            raise ValueError("task must be auto, binary, multiclass, regression or classification")
        if on_leakage not in ("drop", "warn", "raise", "ignore"):
            raise ValueError("on_leakage must be drop, warn, raise or ignore")
        if cv < 2:
            raise ValueError("cv must be at least 2")
        self.target, self.task, self.time_col, self.metric = target, task, time_col, metric
        self.time_budget, self.max_trials, self.cv = float(time_budget), max_trials, cv
        self.preprocess, self.include, self.exclude = preprocess, include, exclude
        self.ensemble, self.on_leakage = ensemble, on_leakage
        self.search_sample, self.seed, self.verbose = search_sample, seed, verbose
        self.model_ = None

    # ── helpers ─────────────────────────────────────────────────────────────
    def _log(self, message: str):
        if self.verbose:
            print(message, flush=True)

    @property
    def is_classification(self) -> bool:
        return self.task_.task in ("binary", "multiclass")

    def _build_preprocess(self, info: TaskInfo):
        if self.preprocess is None:
            return None
        if isinstance(self.preprocess, Pipeline):
            if self.preprocess.target is None:
                self.preprocess.target = self.target
            return self.preprocess
        if self.preprocess != "auto":
            raise TypeError("preprocess must be 'auto', None or a Pipeline")
        from ..cleaners import default_cleaning_pipeline
        from ..features import default_feature_pipeline

        # Leakage is checked by AutoML itself; target encoding uses random folds, which would
        # let future rows inform past ones in a time series.
        features = default_feature_pipeline(self.target, drop_leakage=False,
                                            target_encode=False if info.time_series else None)
        return Pipeline([*default_cleaning_pipeline().steps, *features.steps],
                        name="automl-preprocess", target=self.target)

    def _splitter(self, y, info: TaskInfo):
        from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
        if info.time_series:
            return TimeSeriesSplit(n_splits=self.cv), f"{self.cv}-fold time-ordered CV"
        if info.task in ("binary", "multiclass"):
            counts = np.bincount(y)
            smallest = int(counts[counts > 0].min())
            if smallest < 2:
                rare = self.classes_[int(np.argmin(np.where(counts > 0, counts, np.inf)))]
                raise ValueError(f"Class {rare!r} has only 1 example; each class needs at least 2.")
            folds = min(self.cv, smallest)
            return (StratifiedKFold(folds, shuffle=True, random_state=self.seed),
                    f"{folds}-fold stratified CV")
        return KFold(self.cv, shuffle=True, random_state=self.seed), f"{self.cv}-fold CV"

    def _estimator(self, spec, params):
        from sklearn.pipeline import Pipeline as SkPipeline
        model = spec.build({**spec.defaults, **params}, self.seed, self.is_classification)
        return SkPipeline([("prep", make_preprocessor(self.feature_kinds_, spec.scale)), ("model", model)])

    def _ensemble_estimator(self, members):
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        estimators = [(spec.name, self._estimator(spec, params)) for spec, params in members]
        if self.is_classification:
            return VotingClassifier(estimators, voting="soft")
        return VotingRegressor(estimators)

    def _evaluate(self, name, estimator, params, X, y, splitter):
        from sklearn.model_selection import cross_validate
        start = time.perf_counter()
        trial = {"model": name, "params": params, "score": float("nan"), "std": float("nan"),
                 "seconds": 0.0, "error": None}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cross_validate(estimator, X, y, cv=splitter, scoring=self.metric_,
                                     error_score="raise", n_jobs=1)
            trial["score"] = float(np.mean(res["test_score"]))
            trial["std"] = float(np.std(res["test_score"]))
        except Exception as exc:          # a failing model shouldn't stop the search
            trial["error"] = f"{type(exc).__name__}: {exc}"[:300]
        trial["seconds"] = round(time.perf_counter() - start, 3)
        ok = trial["error"] is None
        shown = f"{_shown(self.metric_, trial['score']):.4f} ± {trial['std']:.4f}" if ok else "failed"
        self._log(f"  {len(self.trials_) + 1:>3}. {name:<24} cv {_metric_label(self.metric_)} = {shown}"
                  f"  ({trial['seconds']:.1f}s)")
        self.trials_.append(trial)
        return trial

    # ── search ──────────────────────────────────────────────────────────────
    def _search(self, X, y, splitter):
        specs = [s for s in model_specs(self.task_.task)
                 if (s.baseline or self.include is None or s.name in self.include)
                 and (self.exclude is None or s.name not in self.exclude)
                 and (s.max_rows is None or len(X) <= s.max_rows)]
        by_name = {s.name: s for s in specs}
        start = time.perf_counter()
        search_end = start + self.time_budget * (0.75 if self.ensemble else 1.0)
        rng = np.random.default_rng(self.seed)

        def model_trials():
            return [t for t in self.trials_ if t["model"] in by_name and not by_name[t["model"]].baseline]

        def room(estimate=0.0):
            if self.max_trials is not None and len(model_trials()) >= self.max_trials:
                return False
            return time.perf_counter() + estimate <= search_end

        baseline = next(s for s in specs if s.baseline)
        self._evaluate(baseline.name, self._estimator(baseline, {}), {}, X, y, splitter)
        for spec in specs:                        # every model with default settings
            if spec.baseline:
                continue
            if model_trials() and not room():
                break
            self._evaluate(spec.name, self._estimator(spec, {}), {}, X, y, splitter)

        while room():                             # random search around the best families
            best = {}
            for t in model_trials():
                if t["error"] is None and t["score"] > best.get(t["model"], (-np.inf,))[0]:
                    best[t["model"]] = (t["score"], t["seconds"])
            ranked = sorted(best, key=lambda m: -best[m][0])[:3]
            fitting = [m for m in ranked if room(best[m][1])]
            if not fitting:
                break
            weights = np.array(_FAMILY_WEIGHTS[: len(fitting)])
            family = fitting[int(rng.choice(len(fitting), p=weights / weights.sum()))]
            spec = by_name[family]
            params = sample_params(spec.space, rng)
            self._evaluate(family, self._estimator(spec, params), params, X, y, splitter)

        if self.ensemble:
            best_per_family = {}
            for t in model_trials():
                if t["error"] is None and t["score"] > best_per_family.get(t["model"], {"score": -np.inf})["score"]:
                    best_per_family[t["model"]] = t
            members = sorted(best_per_family.values(), key=lambda t: -t["score"])[:3]
            if len(members) >= 2:
                pairs = [(by_name[t["model"]], t["params"]) for t in members]
                params = {"members": [{"model": t["model"], "params": t["params"]} for t in members]}
                self._evaluate("ensemble", self._ensemble_estimator(pairs), params, X, y, splitter)
        return by_name

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, df) -> "AutoML":
        t0 = time.perf_counter()
        df = _as_polars(df)
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' is not in the data")
        info = detect_task(df, self.target, time_col=self.time_col if self.time_col else None)
        if self.task != "auto":
            task = self.task
            if task == "classification":
                task = info.task if info.task in ("binary", "multiclass") else "multiclass"
            info = dataclasses.replace(info, task=task, reason=info.reason + f"; task set to {task}")
        self.task_ = info
        df = df.filter(pl.col(self.target).is_not_null())
        if info.time_series:
            df = df.sort(info.time_col)
        self._log(f"AutoML: {info.task} | {info.reason}")

        # 1. leakage
        self.leakage_, self.dropped_leaky_ = None, []
        if self.on_leakage != "ignore":
            report = detect_leakage(df, self.target, seed=self.seed)
            self.leakage_ = report
            leaky = report.leaky_columns
            if leaky and self.on_leakage == "raise":
                raise LeakageError(report)
            if leaky and self.on_leakage == "warn":
                warnings.warn(report.summary(), AutomatedCleaningWarning, stacklevel=2)
            if leaky and self.on_leakage == "drop":
                self.dropped_leaky_ = leaky
                df = df.drop(leaky)
                self._log(f"  dropped leaky column(s): {leaky}")

        # 2. preprocessing
        self.preprocess_ = self._build_preprocess(info)
        if self.preprocess_ is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                df = self.preprocess_.fit_transform(df)
        X_pl, y_raw = df.drop(self.target), df[self.target]
        self.feature_kinds_ = infer_kinds(X_pl)
        X = to_model_frame(X_pl, self.feature_kinds_)

        # 3. labels and metric
        if info.task in ("binary", "multiclass"):
            labels = y_raw.to_list()
            self.classes_ = sorted(set(labels), key=lambda v: (str(type(v)), v))
            index = {v: i for i, v in enumerate(self.classes_)}
            y = np.array([index[v] for v in labels])
            self.task_ = dataclasses.replace(self.task_, n_classes=len(self.classes_))
        else:
            self.classes_ = None
            y = y_raw.cast(pl.Float64).to_numpy()
        self.metric_ = self.metric or DEFAULT_METRICS[info.task]
        from sklearn.metrics import get_scorer
        get_scorer(self.metric_)                  # fail early on a bad metric name

        # 4. search on a sample if the data is large; refit on everything
        Xs, ys = X, y
        if len(X) > self.search_sample:
            if info.time_series:
                Xs, ys = X.iloc[-self.search_sample:], y[-self.search_sample:]
            else:
                from sklearn.model_selection import train_test_split
                Xs, _, ys, _ = train_test_split(X, y, train_size=self.search_sample, random_state=self.seed,
                                                stratify=y if self.is_classification else None)
        splitter, self.cv_description_ = self._splitter(ys, info)
        self.trials_ = []
        self._log(f"  {len(X):,} rows x {X.shape[1]} features | metric {_metric_label(self.metric_)} | "
                  f"{self.cv_description_} | budget {self.time_budget:.0f}s")
        by_name = self._search(Xs, ys, splitter)

        ok = [t for t in self.trials_ if t["error"] is None]
        if not ok:
            raise RuntimeError("Every model failed: " + "; ".join(t["error"] for t in self.trials_))
        best = max(ok, key=lambda t: (t["score"], t["model"] != "ensemble"))
        self.best_ = best
        if best["model"] == "ensemble":
            pairs = [(by_name[m["model"]], m["params"]) for m in best["params"]["members"]]
            estimator = self._ensemble_estimator(pairs)
        else:
            estimator = self._estimator(by_name[best["model"]], best["params"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = estimator.fit(X, y)
        sample = np.random.default_rng(self.seed).choice(len(X), size=min(len(X), 2000), replace=False)
        self._importance_sample = (X.iloc[np.sort(sample)], y[np.sort(sample)])
        self.n_rows_, self.n_features_ = len(X), X.shape[1]
        self.fit_seconds_ = round(time.perf_counter() - t0, 2)
        self._log(f"Best: {best['model']} | cv {_metric_label(self.metric_)} = "
                  f"{_shown(self.metric_, best['score']):.4f} | {len(self.trials_)} trials in {self.fit_seconds_}s")
        return self

    # ── outputs ─────────────────────────────────────────────────────────────
    def _check_fitted(self):
        if self.model_ is None:
            raise RuntimeError("AutoML is not fitted. Call fit() first or AutoML.load().")

    def _features(self, df):
        self._check_fitted()
        df = _as_polars(df)
        X = df.drop([c for c in [self.target, *self.dropped_leaky_] if c in df.columns])
        if self.preprocess_ is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                X = self.preprocess_.transform(X)
        return to_model_frame(X, self.feature_kinds_)

    def predict(self, df) -> pl.Series:
        """Predicted labels (classification) or values (regression)."""
        X = self._features(df)                    # raises clearly if not fitted
        pred = self.model_.predict(X)
        name = f"{self.target}_pred"
        if self.classes_ is not None:
            return pl.Series(name, [self.classes_[int(i)] for i in pred])
        return pl.Series(name, np.asarray(pred, dtype=float))

    def predict_proba(self, df) -> pl.DataFrame:
        """Class probabilities, one ``proba_<class>`` column per class."""
        self._check_fitted()
        if self.classes_ is None:
            raise ValueError("predict_proba is only available for classification")
        proba = self.model_.predict_proba(self._features(df))
        return pl.DataFrame({f"proba_{c}": proba[:, i] for i, c in enumerate(self.classes_)})

    def _encode_labels(self, series: pl.Series) -> np.ndarray:
        index = {v: i for i, v in enumerate(self.classes_)}
        unknown = set(series.to_list()) - set(index)
        if unknown:
            raise ValueError(f"Labels not seen in training: {sorted(map(str, unknown))[:5]}")
        return np.array([index[v] for v in series.to_list()])

    def evaluate(self, df) -> dict:
        """Metrics on labelled data (e.g. a held-out test set)."""
        from sklearn import metrics as m
        df = _as_polars(df).filter(pl.col(self.target).is_not_null())
        X = self._features(df)
        if self.classes_ is not None:
            y = self._encode_labels(df[self.target])
            pred = self.model_.predict(X)
            proba = self.model_.predict_proba(X)
            out = {"accuracy": m.accuracy_score(y, pred), "balanced_accuracy": m.balanced_accuracy_score(y, pred),
                   "f1_macro": m.f1_score(y, pred, average="macro")}
            labels = list(range(len(self.classes_)))
            if len(self.classes_) == 2:
                out["roc_auc"] = m.roc_auc_score(y, proba[:, 1]) if len(set(y)) == 2 else None
            elif len(set(y)) == len(self.classes_):
                out["roc_auc_ovr"] = m.roc_auc_score(y, proba, multi_class="ovr", labels=labels)
            out["log_loss"] = m.log_loss(y, proba, labels=labels)
        else:
            y = df[self.target].cast(pl.Float64).to_numpy()
            pred = self.model_.predict(X)
            out = {"rmse": float(np.sqrt(m.mean_squared_error(y, pred))), "mae": m.mean_absolute_error(y, pred),
                   "r2": m.r2_score(y, pred)}
            if not np.any(y == 0):
                out["mape"] = m.mean_absolute_percentage_error(y, pred)
        return {k: (round(float(v), 6) if v is not None else None) for k, v in out.items()}

    def feature_importance(self, df=None, n_repeats: int = 5) -> pl.DataFrame:
        """Permutation importance: how much the score drops when each feature is shuffled.

        Uses ``df`` (labelled) if given, else a sample of the training rows."""
        from sklearn.inspection import permutation_importance
        self._check_fitted()
        if df is not None:
            df = _as_polars(df).filter(pl.col(self.target).is_not_null())
            X = self._features(df)
            y = (self._encode_labels(df[self.target]) if self.classes_ is not None
                 else df[self.target].cast(pl.Float64).to_numpy())
        elif getattr(self, "_importance_sample", None) is not None:
            X, y = self._importance_sample
        else:
            raise ValueError("Pass labelled data: training rows aren't stored in a loaded model.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = permutation_importance(self.model_, X, y, scoring=self.metric_, n_repeats=n_repeats,
                                         random_state=self.seed, n_jobs=1)
        return (pl.DataFrame({"feature": list(X.columns), "importance": res.importances_mean,
                              "std": res.importances_std})
                .sort("importance", descending=True))

    # ── evaluation & explainability (see automatedcleaning.evaluation) ─────
    def report(self, df, **kw):
        """Detailed held-out metrics: an ``EvaluationReport``."""
        from ..evaluation import evaluate_model
        return evaluate_model(self, df, **kw)

    def explain(self, df, **kw):
        """SHAP values for rows of ``df``: an ``Explanation``."""
        from ..evaluation import explain
        return explain(self, df, **kw)

    def error_analysis(self, df, **kw):
        """Worst predictions and failing data segments: an ``ErrorReport``."""
        from ..evaluation import error_analysis
        return error_analysis(self, df, **kw)

    def fairness(self, df, sensitive, **kw):
        """Bias checks across groups of the ``sensitive`` column(s): a ``FairnessReport``."""
        from ..evaluation import fairness_report
        return fairness_report(self, df, sensitive, **kw)

    def model_card(self, test_df=None, **kw):
        """An auto-generated ``ModelCard`` (``.save("card.html")`` / ``.md`` / ``.json``)."""
        from ..evaluation import model_card
        return model_card(self, test_df, **kw)

    @property
    def leaderboard_(self) -> pl.DataFrame:
        """Every trial, best first (``score`` is higher-is-better; error metrics are negative)."""
        def num(v):                               # failed trials: null, not NaN (NaN sorts first)
            return None if v is None or v != v else v
        rows = [{"model": t["model"], "score": num(t["score"]), "std": num(t["std"]),
                 _metric_label(self.metric_): num(_shown(self.metric_, t["score"])),
                 "seconds": t["seconds"], "params": json.dumps(t["params"], default=str),
                 "error": t["error"]}
                for t in self.trials_]
        frame = pl.DataFrame(rows, infer_schema_length=None)
        return frame.sort("score", descending=True, nulls_last=True)

    def summary(self) -> str:
        self._check_fitted()
        label = _metric_label(self.metric_)
        lines = [f"AutoML: {self.task_.task} '{self.target}' | {self.task_.reason}",
                 f"  {self.n_rows_:,} rows x {self.n_features_} features | metric {label} | "
                 f"{self.cv_description_} | {len(self.trials_)} trials in {self.fit_seconds_}s",
                 f"  Best: {self.best_['model']}  cv {label} = {_shown(self.metric_, self.best_['score']):.4f}"
                 f" ± {self.best_['std']:.4f}",
                 "  Best per model:"]
        best = {}
        for t in self.trials_:
            if t["error"] is None and t["score"] > best.get(t["model"], {"score": -np.inf})["score"]:
                best[t["model"]] = t
        for i, t in enumerate(sorted(best.values(), key=lambda t: -t["score"]), 1):
            lines.append(f"    {i:>2}. {t['model']:<24} {_shown(self.metric_, t['score']):.4f} ± {t['std']:.4f}")
        if self.dropped_leaky_:
            lines.append(f"  Dropped leaky columns: {self.dropped_leaky_}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary() if self.model_ is not None else f"AutoML(target={self.target!r}, unfitted)"

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path):
        """Save to a directory (manifest.json + model.joblib + preprocess.json)."""
        self._check_fitted()
        manifest = {
            "target": self.target, "task": dataclasses.asdict(self.task_), "metric": self.metric_,
            "classes": self.classes_, "feature_kinds": self.feature_kinds_,
            "dropped_leaky": self.dropped_leaky_, "best": self.best_, "trials": self.trials_,
            "cv": self.cv_description_, "n_rows": self.n_rows_, "n_features": self.n_features_,
            "fit_seconds": self.fit_seconds_,
            "leakage": self.leakage_.to_dict() if self.leakage_ is not None else None,
        }
        return save_artifact(path, "supervised", self.model_, manifest, self.preprocess_)

    @classmethod
    def load(cls, path) -> "AutoML":
        """Load a model saved with :meth:`save` (verifies the model checksum first)."""
        manifest, model, preprocess = load_artifact(path, "supervised")
        obj = cls(manifest["target"], verbose=False)
        obj.model_, obj.preprocess_ = model, preprocess
        obj.task_ = TaskInfo(**manifest["task"])
        obj.metric_, obj.classes_ = manifest["metric"], manifest["classes"]
        obj.feature_kinds_, obj.dropped_leaky_ = manifest["feature_kinds"], manifest["dropped_leaky"]
        obj.best_, obj.trials_, obj.cv_description_ = manifest["best"], manifest["trials"], manifest["cv"]
        obj.n_rows_, obj.n_features_ = manifest["n_rows"], manifest["n_features"]
        obj.fit_seconds_, obj.leakage_ = manifest["fit_seconds"], None
        obj._importance_sample = None
        return obj
