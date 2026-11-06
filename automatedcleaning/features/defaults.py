"""The default, target-aware feature-engineering pipeline."""

from __future__ import annotations

from ..core import Pipeline

# (step name, default params, enabled by default)
DEFAULT_FEATURE_STEPS = [
    # Lags first: they need the raw time column, which datetime_features replaces with its parts.
    ("lag_features", {}, False),            # enabled automatically when time_col is given
    ("datetime_features", {}, True),
    ("drop_leakage", {}, True),
    ("text_embeddings", {}, True),
    ("target_encode", {"min_unique": 16}, True),
    ("encode_categorical", {"method": "onehot", "max_unique": 15}, True),
    ("interactions", {}, False),
    ("group_aggregations", {}, False),
    ("select_features", {}, True),
]


def default_feature_pipeline(target: str, *, time_col: str | None = None,
                             name: str = "feature-engineering", on_schema_mismatch: str = "warn",
                             verbose: bool = False, audit_log=None, **overrides) -> Pipeline:
    """The recommended feature pipeline for a supervised problem.

    Steps: (time-series lags) → date/time features → drop leaky columns → embed free text → target-encode
    high-cardinality categories (≥ 16 values, out-of-fold) → one-hot encode the rest →
    select features by importance. Off by default: ``interactions``, ``group_aggregations``.

    With ``time_col`` (time series), target lags/rolling means are added and target encoding is
    turned off, because random folds would let future rows inform past ones.

    Override any step by name: ``False`` removes it, ``True`` enables an optional one, a dict
    enables it with those parameters, e.g. ``interactions={"max_features": 4}``.
    """
    known = {n for n, _, _ in DEFAULT_FEATURE_STEPS}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown feature step(s) {sorted(unknown)}; choose from {sorted(known)}")
    defaults = {n: (dict(p), on) for n, p, on in DEFAULT_FEATURE_STEPS}
    if time_col:
        defaults["lag_features"] = ({"time_col": time_col}, True)
        defaults["target_encode"] = (defaults["target_encode"][0], False)
    steps = []
    for step_name, _, _ in DEFAULT_FEATURE_STEPS:
        params, enabled = defaults[step_name]
        override = overrides.get(step_name)
        if override is False or (override is None and not enabled):
            continue
        if isinstance(override, dict):
            params = {**params, **override}
        steps.append({"step": step_name, "params": params})
    return Pipeline(steps, name=name, target=target, on_schema_mismatch=on_schema_mismatch,
                    verbose=verbose, audit_log=audit_log)
