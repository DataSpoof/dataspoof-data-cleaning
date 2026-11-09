"""The ``Step`` contract shared by every module (cleaning, features, AutoML, ...)."""

from __future__ import annotations

import inspect
from typing import Any

import polars as pl

from . import registry


class NotFittedError(RuntimeError):
    """Raised when ``transform`` is called before ``fit``."""


class Step:
    """Base class for every pipeline step.

    Subclass contract
    -----------------
    * ``name``: unique registry key used in configs (``{"step": name}``).
    * ``__init__`` takes keyword-only, JSON-serialisable parameters and forwards them to
      ``super().__init__(fit_only=fit_only, **params)``; they are kept in ``self.params``.
    * ``_fit(df)`` learns everything the step needs from *training* data and returns it as a
      JSON-serialisable dict, stored in ``self.state_``. Stateless steps don't override it.
    * ``_transform(df)`` applies ``self.state_`` and must never learn from ``df``. That is
      what makes cleaning new data reproduce training-time behaviour exactly.
    * ``fit_only = True`` marks steps that only run while fitting (e.g. dropping duplicate
      or outlier rows); ``Pipeline.transform`` skips them.

    Targets
    -------
    In a ``Pipeline(target=...)`` the target column is hidden from ordinary steps, so they
    can't impute, clip, encode or drop it. Steps that need it (target encoding, feature
    selection, leakage checks) set ``uses_target = True`` and receive the full frame; during
    ``transform`` on new data the target may be absent and they must cope. Steps that only
    remove rows implement ``row_mask`` so the pipeline can drop the same rows from the target.
    """

    name = ""
    fit_only = False
    uses_target = False

    def __init__(self, fit_only: bool | None = None, **params: Any):
        self.params = params
        if fit_only is not None:
            self.fit_only = bool(fit_only)
        self.state_: dict | None = None

    # -- public API ---------------------------------------------------------
    @property
    def label(self) -> str:
        """Name shown in reports."""
        return self.name

    def row_mask(self, df: pl.DataFrame):
        """For steps that only remove rows: a boolean Series of rows to keep. Others: ``None``."""
        return None

    def _target(self, required: bool = True):
        """The target column: the ``target`` param, else the pipeline's, else the one fitted on."""
        target = (self.params.get("target") or getattr(self, "_pipeline_target", None)
                  or (self.state_ or {}).get("target"))
        if required and not target:
            raise ValueError(f"Step '{self.label}' needs a target column: pass target=... or "
                             f"build the Pipeline with target=...")
        return target

    @property
    def is_fitted(self) -> bool:
        return self.state_ is not None

    def fit(self, df: pl.DataFrame) -> "Step":
        self.state_ = self._fit(df) or {}
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.state_ is None:
            raise NotFittedError(f"Step '{self.label}' must be fitted before transform().")
        return self._transform(df)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)

    # -- to implement -------------------------------------------------------
    def _fit(self, df: pl.DataFrame) -> dict:
        return {}

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError(f"{type(self).__name__} must implement _transform().")

    # -- serialisation ------------------------------------------------------
    def to_dict(self, include_state: bool = True) -> dict:
        d: dict[str, Any] = {"step": self.name, "params": dict(self.params)}
        if self.fit_only != type(self).fit_only:
            d["fit_only"] = self.fit_only
        if include_state:
            d["state"] = self.state_
        return d

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


class FunctionStep(Step):
    """Wrap a stateless ``df -> df`` function (e.g. ``ac.normalize_whitespace``) as a step.

    Function steps learn nothing in ``fit``; they re-run their logic on every batch.
    """

    name = "function"

    def __init__(self, function, fit_only: bool | None = None, **kwargs: Any):
        fn = function if callable(function) else registry.resolve_function(function)
        if fn is None:
            raise ValueError(f"No function named '{function}' is registered as a step.")
        fname = function if isinstance(function, str) else fn.__name__
        try:
            inspect.signature(fn).bind(None, **kwargs)
        except TypeError as exc:
            raise TypeError(f"Invalid parameters for step '{fname}': {exc}") from None
        super().__init__(fit_only=fit_only, **kwargs)
        self.function = fname
        self._fn = fn

    @property
    def label(self) -> str:
        return self.function

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        out = self._fn(df, **self.params)
        if isinstance(out, tuple):   # e.g. check_cardinality returns (df, info)
            out = next((x for x in out if isinstance(x, pl.DataFrame)), None)
        if not isinstance(out, pl.DataFrame):
            raise TypeError(f"Step '{self.function}' returned {type(out).__name__}; "
                            f"a polars DataFrame was expected.")
        return out

    def to_dict(self, include_state: bool = True) -> dict:
        d = super().to_dict(include_state)
        # A stateful step with the same name would shadow this function on reload.
        shadowed = registry.get_step_class(self.function) is not None
        d["step"] = f"fn:{self.function}" if shadowed else self.function
        return d

    def __repr__(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.params.items())
        return f"FunctionStep({self.function!r}{args})"
