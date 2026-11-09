"""Candidate models and their hyperparameter search spaces."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from typing import Callable

import numpy as np

# Search-space entries: ("log", lo, hi) | ("uniform", lo, hi) | ("int", lo, hi) | ("choice", [...])


@dataclasses.dataclass
class ModelSpec:
    name: str
    build: Callable            # build(params: dict, seed: int, classify: bool) -> estimator
    space: dict
    defaults: dict
    scale: bool = False        # needs standardised inputs
    max_rows: int | None = None
    baseline: bool = False


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    params = {}
    for key, spec in space.items():
        kind = spec[0]
        if kind == "log":
            params[key] = float(math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2]))))
        elif kind == "uniform":
            params[key] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "int":
            params[key] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "choice":
            params[key] = spec[1][int(rng.integers(len(spec[1])))]
        else:
            raise ValueError(f"Unknown search-space kind {kind!r}")
    return params


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# ── builders ─────────────────────────────────────────────────────────────────

def _dummy(p, seed, classify):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    return DummyClassifier(strategy="prior") if classify else DummyRegressor(strategy="mean")


def _linear(p, seed, classify):
    from sklearn.linear_model import LogisticRegression, Ridge
    if classify:
        return LogisticRegression(C=p.get("C", 1.0), class_weight=p.get("class_weight"),
                                  max_iter=3000, random_state=seed)
    return Ridge(alpha=p.get("alpha", 1.0), random_state=seed)


def _forest(kind):
    def build(p, seed, classify):
        from sklearn import ensemble
        cls = {("rf", True): ensemble.RandomForestClassifier, ("rf", False): ensemble.RandomForestRegressor,
               ("et", True): ensemble.ExtraTreesClassifier, ("et", False): ensemble.ExtraTreesRegressor}[(kind, classify)]
        extra = {"class_weight": p.get("class_weight")} if classify else {}
        return cls(n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth"),
                   min_samples_leaf=p.get("min_samples_leaf", 1),
                   max_features=p.get("max_features", "sqrt" if classify else 1.0),
                   n_jobs=-1, random_state=seed, **extra)
    return build


def _hgb(p, seed, classify):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    cls = HistGradientBoostingClassifier if classify else HistGradientBoostingRegressor
    extra = {"class_weight": p.get("class_weight")} if classify else {}
    return cls(learning_rate=p.get("learning_rate", 0.1), max_iter=p.get("max_iter", 200),
               max_leaf_nodes=p.get("max_leaf_nodes", 31), min_samples_leaf=p.get("min_samples_leaf", 20),
               l2_regularization=p.get("l2_regularization", 0.0), random_state=seed, **extra)


def _knn(p, seed, classify):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    cls = KNeighborsClassifier if classify else KNeighborsRegressor
    return cls(n_neighbors=p.get("n_neighbors", 15), weights=p.get("weights", "distance"))


def _lightgbm(p, seed, classify):
    import lightgbm as lgb
    cls = lgb.LGBMClassifier if classify else lgb.LGBMRegressor
    return cls(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
               num_leaves=p.get("num_leaves", 31), min_child_samples=p.get("min_child_samples", 20),
               subsample=p.get("subsample", 1.0), subsample_freq=1,
               colsample_bytree=p.get("colsample_bytree", 1.0), reg_lambda=p.get("reg_lambda", 0.0),
               random_state=seed, verbose=-1, n_jobs=-1)


def _xgboost(p, seed, classify):
    import xgboost as xgb
    cls = xgb.XGBClassifier if classify else xgb.XGBRegressor
    return cls(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
               max_depth=p.get("max_depth", 6), subsample=p.get("subsample", 1.0),
               colsample_bytree=p.get("colsample_bytree", 1.0),
               min_child_weight=p.get("min_child_weight", 1.0), random_state=seed, n_jobs=-1,
               tree_method="hist", verbosity=0)


_FOREST_SPACE = {"n_estimators": ("int", 100, 400), "max_depth": ("choice", [None, 6, 10, 16, 24]),
                 "min_samples_leaf": ("int", 1, 10), "max_features": ("choice", ["sqrt", "log2", 0.5, 1.0])}


def model_specs(task: str) -> list:
    """Candidate models for ``task`` (binary | multiclass | regression), cheapest first."""
    classify = task in ("binary", "multiclass")
    weight = {"class_weight": ("choice", [None, "balanced"])} if classify else {}
    specs = [
        ModelSpec("baseline", _dummy, {}, {}, baseline=True),
        ModelSpec("logistic_regression" if classify else "ridge", _linear,
                  {"C": ("log", 1e-3, 1e2), **weight} if classify else {"alpha": ("log", 1e-3, 1e3)},
                  {}, scale=True),
        ModelSpec("hist_gradient_boosting", _hgb,
                  {"learning_rate": ("log", 0.02, 0.3), "max_iter": ("int", 100, 500),
                   "max_leaf_nodes": ("int", 15, 63), "min_samples_leaf": ("int", 5, 50),
                   "l2_regularization": ("log", 1e-6, 1.0), **weight}, {}),
        ModelSpec("random_forest", _forest("rf"), {**_FOREST_SPACE, **weight}, {}),
        ModelSpec("extra_trees", _forest("et"), {**_FOREST_SPACE, **weight}, {}),
        ModelSpec("knn", _knn, {"n_neighbors": ("int", 3, 50), "weights": ("choice", ["uniform", "distance"])},
                  {}, scale=True, max_rows=20000),
    ]
    boost_space = {"n_estimators": ("int", 100, 600), "learning_rate": ("log", 0.02, 0.3),
                   "subsample": ("uniform", 0.6, 1.0), "colsample_bytree": ("uniform", 0.6, 1.0)}
    if _has("lightgbm"):
        specs.append(ModelSpec("lightgbm", _lightgbm,
                               {**boost_space, "num_leaves": ("int", 15, 127),
                                "min_child_samples": ("int", 5, 60), "reg_lambda": ("log", 1e-6, 10.0)}, {}))
    if _has("xgboost"):
        specs.append(ModelSpec("xgboost", _xgboost,
                               {**boost_space, "max_depth": ("int", 3, 10),
                                "min_child_weight": ("log", 0.5, 10.0)}, {}))
    return specs


def available_models(task: str = "binary") -> list:
    """Names of the models AutoML can try for ``task`` in this environment."""
    return [s.name for s in model_specs(task)]
