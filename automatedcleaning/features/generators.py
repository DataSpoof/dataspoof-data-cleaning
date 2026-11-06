"""Feature generators: interactions, group aggregations, and time-series lags/rolling windows."""

from __future__ import annotations

from itertools import combinations

import polars as pl

from .. import _rustcore
from .._detect import floats, is_num, is_text
from ..core import Step, register_step

_ROW = "__fe_row__"
AGGREGATIONS = ("mean", "std", "min", "max", "median", "sum")


def _numeric_features(df, exclude=(), columns=None) -> list:
    return [c for c, t in df.schema.items()
            if c not in exclude and (columns is None or c in columns)
            and is_num(t) and df[c].drop_nulls().n_unique() > 2]


def _rank(df, cols, target=None) -> list:
    """Order columns by |correlation with a numeric target|, else by variance."""
    if target and target in df.columns and is_num(df.schema[target]):
        y = floats(df[target])
        def score(c):
            r = _rustcore.correlation_matrix([floats(df[c]), y])[0][1]
            return abs(r) if r == r else 0.0
    else:
        def score(c):
            v = df[c].cast(pl.Float64).var()
            return v if v is not None and v == v else 0.0
    return sorted(cols, key=lambda c: (-score(c), c))


@register_step
class Interactions(Step):
    """Pairwise products and ratios of the most informative numeric columns.

    Picks the ``max_features`` numeric columns most correlated with a numeric target (or with
    the highest variance when there's no numeric target) and adds ``a_x_b`` (product) and
    ``a_div_b`` (ratio; null when dividing by zero) for every pair.
    """

    name = "interactions"
    uses_target = True
    OPERATIONS = ("multiply", "ratio", "add", "subtract")

    def __init__(self, *, columns=None, max_features=6, operations=("multiply", "ratio"),
                 target=None, fit_only=None):
        unknown = set(operations) - set(self.OPERATIONS)
        if unknown:
            raise ValueError(f"Unknown operation(s) {sorted(unknown)}; choose from {self.OPERATIONS}")
        super().__init__(fit_only=fit_only, columns=columns, max_features=max_features,
                         operations=list(operations), target=target)

    def _fit(self, df):
        target = self._target(required=False)
        cols = _numeric_features(df, exclude={target}, columns=self.params["columns"])
        top = set(_rank(df, cols, target)[: self.params["max_features"]])
        top = [c for c in df.columns if c in top]          # name pairs in column order: a_x_b
        return {"target": target, "features": top, "pairs": [list(p) for p in combinations(top, 2)]}

    def _transform(self, df):
        ops = self.params["operations"]
        exprs = []
        for a, b in self.state_["pairs"]:
            if a not in df.columns or b not in df.columns:
                continue
            x, y = pl.col(a).cast(pl.Float64), pl.col(b).cast(pl.Float64)
            if "multiply" in ops:
                exprs.append((x * y).alias(f"{a}_x_{b}"))
            if "ratio" in ops:
                exprs.append(pl.when(y != 0).then(x / y).otherwise(None).alias(f"{a}_div_{b}"))
            if "add" in ops:
                exprs.append((x + y).alias(f"{a}_plus_{b}"))
            if "subtract" in ops:
                exprs.append((x - y).alias(f"{a}_minus_{b}"))
        return df.with_columns(exprs) if exprs else df


@register_step
class GroupAggregations(Step):
    """Add per-group statistics of numeric columns, e.g. the average fare of a passenger's class.

    For each grouping column (auto: up to ``max_keys`` text columns with 2–50 distinct values)
    the learned table adds ``<key>_count`` and ``<col>_<agg>_by_<key>`` columns. The target is
    never aggregated (that would leak it). Unseen groups get nulls.
    """

    name = "group_aggregations"
    uses_target = True

    def __init__(self, *, group_by=None, columns=None, aggregations=("mean", "std"), max_groups=1000,
                 max_keys=3, max_columns=10, target=None, fit_only=None):
        unknown = set(aggregations) - set(AGGREGATIONS)
        if unknown:
            raise ValueError(f"Unknown aggregation(s) {sorted(unknown)}; choose from {AGGREGATIONS}")
        super().__init__(fit_only=fit_only, group_by=group_by, columns=columns,
                         aggregations=list(aggregations), max_groups=max_groups, max_keys=max_keys,
                         max_columns=max_columns, target=target)

    def _fit(self, df):
        p = self.params
        target = self._target(required=False)
        keys = p["group_by"]
        if keys is None:
            keys = sorted((c for c, t in df.schema.items()
                           if c != target and is_text(t) and 2 <= df[c].n_unique() <= 50),
                          key=lambda c: (df[c].n_unique(), c))[: p["max_keys"]]
        keys = [k for k in keys if k in df.columns and k != target]
        cols = _rank(df, _numeric_features(df, exclude={target, *keys}, columns=p["columns"]))
        cols = cols[: p["max_columns"]]
        tables = {}
        for key in keys:
            if df[key].n_unique() > p["max_groups"]:
                continue
            aggs = [pl.len().alias(f"{key}_count")] + [
                getattr(pl.col(c), agg)().cast(pl.Float64).alias(f"{c}_{agg}_by_{key}")
                for c in cols for agg in p["aggregations"]]
            table = df.group_by(key).agg(aggs).sort(key, nulls_last=True)
            tables[key] = {"rows": table.to_dicts()}
        return {"target": target, "tables": tables}

    def _transform(self, df):
        for key, info in self.state_["tables"].items():
            if key not in df.columns or not info["rows"]:
                continue
            table = pl.DataFrame(info["rows"], infer_schema_length=None)
            table = table.with_columns(pl.col(key).cast(df.schema[key], strict=False))
            new_cols = [c for c in table.columns if c != key and c not in df.columns]
            df = (df.with_row_index(_ROW)
                  .join(table.select([key, *new_cols]), on=key, how="left")
                  .sort(_ROW).drop(_ROW))
        return df


@register_step
class LagFeatures(Step):
    """Time-series lags and rolling means, computed in time order (optionally per entity).

    Adds ``<col>_lag<k>`` for each lag and ``<col>_roll<w>`` (mean of the previous ``w`` values,
    excluding the current one, so it never peeks at the present). Lags of the target are
    included by default; they're computed from earlier rows, so they're legitimate. Row order
    is preserved. To predict new periods, pass recent history (with known targets) together
    with the new rows; if the target column is absent, its lag columns are null.
    """

    name = "lag_features"
    uses_target = True

    def __init__(self, *, time_col, columns=None, lags=(1, 2, 3), windows=(3, 7), group_by=None,
                 include_target=True, target=None, fit_only=None):
        super().__init__(fit_only=fit_only, time_col=time_col, columns=columns, lags=list(lags),
                         windows=list(windows), group_by=group_by, include_target=include_target,
                         target=target)

    def _fit(self, df):
        p = self.params
        if p["time_col"] not in df.columns:
            raise ValueError(f"time_col '{p['time_col']}' is not in the data")
        target = self._target(required=False)
        cols = list(p["columns"] or [])
        if p["include_target"] and target and target not in cols:
            cols.insert(0, target)
        return {"target": target, "columns": cols}

    def _transform(self, df):
        p = self.params
        groups = [p["group_by"]] if isinstance(p["group_by"], str) else (p["group_by"] or [])
        work = df.with_row_index(_ROW).sort([*groups, p["time_col"], _ROW])
        exprs = []
        for c in self.state_["columns"]:
            names = [f"{c}_lag{k}" for k in p["lags"]] + [f"{c}_roll{w}" for w in p["windows"]]
            if c not in df.columns:
                exprs += [pl.lit(None, dtype=pl.Float64).alias(n) for n in names]
                continue
            base = pl.col(c).cast(pl.Float64)
            built = [base.shift(k) for k in p["lags"]]
            built += [base.shift(1).rolling_mean(window_size=w, min_samples=1) for w in p["windows"]]
            for expr, alias in zip(built, names):
                exprs.append((expr.over(groups) if groups else expr).alias(alias))
        return work.with_columns(exprs).sort(_ROW).drop(_ROW)
