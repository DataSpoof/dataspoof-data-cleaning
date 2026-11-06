"""A uniform view of a fitted model, so every evaluation tool works the same way."""

from __future__ import annotations

import numpy as np
import polars as pl


def as_polars(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


class ModelAdapter:
    """Wraps a fitted ``AutoML`` model or a scikit-learn estimator.

    Internally classes are integer codes ``0..k-1`` aligned with ``classes`` (the original
    labels); ``X(df)`` returns the model-input frame (pandas) for a Polars/pandas frame.
    """

    def __init__(self, *, name, target, task, classes, to_X, predict_raw, proba, automl=None):
        self.name, self.target, self.task, self.classes = name, target, task, classes
        self._to_X, self._predict_raw, self._proba, self.automl = to_X, predict_raw, proba, automl

    @classmethod
    def of(cls, model, target: str | None = None) -> "ModelAdapter":
        if isinstance(model, cls):
            return model
        from ..automl import AutoML
        if isinstance(model, AutoML):
            model._check_fitted()
            classify = model.classes_ is not None
            known = getattr(model.preprocess_, "input_schema_", None) if model.preprocess_ is not None \
                else model.feature_kinds_

            def to_X(df):
                # Evaluation data often carries extra columns (e.g. a sensitive attribute that was
                # deliberately left out of training); drop them quietly instead of warning.
                if known:
                    df = df.select([c for c in df.columns if c in known or c == model.target])
                return model._features(df)

            return cls(
                name=model.best_["model"], target=model.target, task=model.task_.task,
                classes=list(model.classes_) if classify else None, to_X=to_X,
                predict_raw=(lambda X: np.asarray(model.model_.predict(X)).astype(int)) if classify
                else (lambda X: np.asarray(model.model_.predict(X), dtype=float)),
                proba=model.model_.predict_proba if classify else None, automl=model)
        if hasattr(model, "predict"):
            if target is None:
                raise ValueError("Pass target=... when evaluating a scikit-learn model.")
            classify = hasattr(model, "predict_proba") and hasattr(model, "classes_")
            classes = list(model.classes_) if classify else None
            index = {c: i for i, c in enumerate(classes or [])}

            def to_X(df):
                df = as_polars(df)
                return df.drop([target] if target in df.columns else []).to_pandas()

            def predict_raw(X):
                pred = model.predict(X)
                return np.array([index[p] for p in pred]) if classify else np.asarray(pred, dtype=float)

            task = ("binary" if len(classes) == 2 else "multiclass") if classify else "regression"
            return cls(name=type(model).__name__, target=target, task=task, classes=classes,
                       to_X=to_X, predict_raw=predict_raw,
                       proba=model.predict_proba if classify else None)
        raise TypeError(f"Can't evaluate a {type(model).__name__}; pass a fitted AutoML model "
                        f"or a scikit-learn estimator with target=...")

    # -- views ---------------------------------------------------------------
    @property
    def is_classification(self) -> bool:
        return self.classes is not None

    def X(self, df):
        return self._to_X(as_polars(df))

    def predict(self, X) -> np.ndarray:
        """Class codes (classification) or values (regression) for a model-input frame."""
        return self._predict_raw(X)

    def proba(self, X) -> np.ndarray:
        if self._proba is None:
            raise ValueError("This model has no probabilities (it isn't a classifier).")
        return np.asarray(self._proba(X), dtype=float)

    def y(self, df) -> np.ndarray:
        """True target as class codes (classification) or floats (regression)."""
        values = as_polars(df)[self.target]
        if not self.is_classification:
            return values.cast(pl.Float64).to_numpy()
        index = {c: i for i, c in enumerate(self.classes)}
        unknown = set(values.to_list()) - set(index)
        if unknown:
            raise ValueError(f"Labels not seen in training: {sorted(map(str, unknown))[:5]}")
        return np.array([index[v] for v in values.to_list()])

    def positive_index(self, pos_label=None) -> int:
        """Index of the positive class for a binary model (default: the second sorted class)."""
        if self.task != "binary":
            raise ValueError("A positive class only applies to binary classification.")
        if pos_label is None:
            return 1
        for i, c in enumerate(self.classes):
            if c == pos_label or str(c) == str(pos_label):
                return i
        raise ValueError(f"pos_label {pos_label!r} is not one of the classes {self.classes}")

    def class_index(self, label) -> int:
        for i, c in enumerate(self.classes or []):
            if c == label or str(c) == str(label):
                return i
        raise ValueError(f"{label!r} is not one of the classes {self.classes}")

    def output_name(self, pos: int | None = None) -> str:
        if not self.is_classification:
            return f"predicted {self.target}"
        if pos is not None:
            return f"P({self.target} = {self.classes[pos]})"
        return f"P({self.target} = class)"

    def labelled(self, df) -> pl.DataFrame:
        """Rows with a known target (unlabelled rows can't be evaluated)."""
        df = as_polars(df)
        if self.target not in df.columns:
            raise ValueError(f"Evaluation data needs the target column '{self.target}'.")
        return df.filter(pl.col(self.target).is_not_null())
