"""Time-series forecasting: multi-step horizons, many series at once, backtests, intervals.

``Forecaster(target, time_col, horizon=14, group="store")`` learns from history and predicts the
next ``horizon`` periods for every series:

* **models** — a seasonal-naive baseline and a gradient-boosting model on lags, rolling means and
  calendar features (recursive multi-step); the better one on the backtest is kept.
* **backtest** — rolling-origin evaluation (``folds`` cut-offs, each forecasting ``horizon`` steps
  on data the model never saw): MAE, RMSE, sMAPE and MASE (error relative to seasonal-naive).
* **intervals** — conformal: the empirical spread of backtest errors at each step ahead, so the
  stated coverage (``coverage=0.9``) holds on past forecasts.
* ``reconcile_bottom_up`` makes forecasts of a hierarchy add up (stores -> region -> total).
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import polars as pl

from ._detect import is_temporal

_FREQ_SECONDS = {"h": 3600, "d": 86400, "w": 7 * 86400, "mo": 30.44 * 86400, "q": 91.3 * 86400, "y": 365.25 * 86400}
_DEFAULT_SEASON = {"h": 24, "d": 7, "w": 52, "mo": 12, "q": 4, "y": 1}


def infer_frequency(times: pl.Series) -> str:
    """'h', 'd', 'w', 'mo', 'q' or 'y' from the median spacing of the timestamps."""
    t = times.drop_nulls().unique().sort()
    if t.len() < 3:
        raise ValueError("Need at least 3 distinct timestamps to infer the frequency")
    diffs = t.diff().drop_nulls()
    raw = diffs.dt.total_seconds().median() if diffs.dtype == pl.Duration else diffs.median()
    seconds = float(raw)  # type: ignore[arg-type]
    return min(_FREQ_SECONDS, key=lambda k: abs(math.log(max(seconds, 1) / _FREQ_SECONDS[k])))


def _step(freq: str) -> str:
    return {"h": "1h", "d": "1d", "w": "1w", "mo": "1mo", "q": "3mo", "y": "1y"}[freq]


def _metrics(y, f, naive_scale):
    y, f = np.asarray(y, float), np.asarray(f, float)
    err = f - y
    denom = np.abs(y) + np.abs(f)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "smape": float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(err) / np.where(denom == 0, 1, denom)))),
            "mase": float(np.mean(np.abs(err)) / naive_scale) if naive_scale else None}


@dataclasses.dataclass
class BacktestReport:
    model: str
    folds: int
    horizon: int
    metrics: dict               # model -> {"mae", "rmse", "smape", "mase"}
    by_step: pl.DataFrame       # step-ahead errors of the chosen model
    coverage: float | None = None

    def summary(self) -> str:
        lines = [f"Backtest: {self.folds} folds x {self.horizon} steps ahead -> chosen: {self.model}"]
        for name, m in self.metrics.items():
            mase = f", MASE {m['mase']:.3f}" if m.get("mase") is not None else ""
            lines.append(f"  {name:<16} MAE {m['mae']:.4g}, RMSE {m['rmse']:.4g}, sMAPE {m['smape']:.1%}{mase}")
        if self.coverage is not None:
            lines.append(f"  interval coverage on the backtest: {self.coverage:.0%}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class Forecaster:
    """>>> fc = Forecaster("sales", "date", horizon=14, group="store").fit(history)
    >>> fc.backtest_                  # how accurate it was on held-out history
    >>> fc.predict()                  # store, date, forecast, lower, upper for the next 14 days
    """

    def __init__(self, target: str, time_col: str, *, horizon: int = 14, group: str | None = None,
                 freq: str | None = None, season: int | None = None, lags: list | None = None,
                 folds: int = 3, coverage: float = 0.9, models=("seasonal_naive", "gradient_boosting"),
                 known_future: list | None = None, seed: int = 42):
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.target, self.time_col, self.horizon, self.group = target, time_col, horizon, group
        self.freq, self.season, self.lags, self.folds = freq, season, lags, folds
        self.coverage, self.models, self.known_future, self.seed = coverage, tuple(models), list(known_future or []), seed
        self.model_ = None

    # ── data ────────────────────────────────────────────────────────────────
    def _prepare(self, df) -> pl.DataFrame:
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        for c in [self.target, self.time_col] + ([self.group] if self.group else []):
            if c not in df.columns:
                raise ValueError(f"Column '{c}' is not in the data")
        if not is_temporal(df.schema[self.time_col]):
            df = df.with_columns(pl.col(self.time_col).str.to_datetime(strict=False))
        if df.schema[self.time_col] == pl.Date:
            df = df.with_columns(pl.col(self.time_col).cast(pl.Datetime("us")))
        g = self.group or "__series__"
        if not self.group:
            df = df.with_columns(pl.lit("all").alias(g))
        keep = [g, self.time_col, self.target, *self.known_future]
        return (df.select(keep).with_columns(pl.col(self.target).cast(pl.Float64), pl.col(g).cast(pl.String))
                .drop_nulls([self.time_col]).sort([g, self.time_col]))

    def _lags(self) -> list:
        if self.lags:
            return sorted(set(int(x) for x in self.lags))
        base = {1, 2, 3, self.season_, 2 * self.season_} if self.season_ > 1 else {1, 2, 3, 4}
        return sorted(x for x in base if x >= 1)

    def _features(self, frame: pl.DataFrame, g: str) -> pl.DataFrame:
        y = pl.col(self.target)
        t = pl.col(self.time_col)
        exprs = [y.shift(k).over(g).alias(f"lag_{k}") for k in self.lags_]
        for w in sorted({3, self.season_} - {0, 1}):
            exprs.append(y.shift(1).rolling_mean(w, min_samples=1).over(g).alias(f"roll_mean_{w}"))
        cal = [t.dt.month().alias("month"), t.dt.weekday().alias("weekday"), t.dt.day().alias("day"),
               t.dt.ordinal_day().alias("day_of_year")]
        if self.freq_ == "h":
            cal.append(t.dt.hour().alias("hour"))
        return frame.with_columns(exprs + cal + [pl.col(g).alias("series")])

    def _xy(self, feats: pl.DataFrame, g: str):
        cols = [c for c in feats.columns if c not in (self.target, self.time_col, g)]
        X = feats.select(cols).to_pandas()
        X["series"] = X["series"].astype("category")
        return X, feats[self.target].to_numpy()

    def _fit_gbm(self, frame, g):
        from sklearn.ensemble import HistGradientBoostingRegressor
        feats = self._features(frame, g).drop_nulls([f"lag_{self.lags_[0]}", self.target])
        if feats.height < 10:
            raise ValueError("Not enough history after building lag features")
        X, y = self._xy(feats, g)
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features="from_dtype",
                                              random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        return model

    def _forecast(self, name, model, history: pl.DataFrame, g: str, horizon: int, future: pl.DataFrame | None = None):
        """Recursive multi-step forecast for every series in ``history`` (all series per model call)."""
        cols = [g, self.time_col, self.target, *self.known_future]
        keep = max(self.lags_) + self.season_ + 3
        work = (history.select(cols).sort([g, self.time_col]).group_by(g, maintain_order=True).tail(keep)
                .with_columns(pl.col(self.target).cast(pl.Float64), pl.lit(False).alias("_new")))
        last = work.group_by(g, maintain_order=True).agg(pl.col(self.time_col).max().alias("_last"))
        outputs = []
        for step in range(1, horizon + 1):
            nxt = last.with_columns(pl.col("_last").dt.offset_by(self._offset(step)).alias(self.time_col)).drop("_last")
            if self.known_future:
                if future is None:
                    raise ValueError(f"Pass future= with {self.known_future} for the forecast period")
                nxt = nxt.join(future.select([g, self.time_col, *self.known_future]).with_columns(pl.col(g).cast(pl.String)),
                               on=[g, self.time_col], how="left")
            nxt = nxt.with_columns(pl.lit(None, dtype=pl.Float64).alias(self.target), pl.lit(True).alias("_new"))
            work = pl.concat([work, nxt.select(work.columns)], how="vertical_relaxed").sort([g, self.time_col])
            if name == "seasonal_naive":
                s = self.season_
                pred = work.with_columns(pl.col(self.target).shift(s).over(g).alias("_p"),
                                         pl.col(self.target).shift(1).over(g).alias("_p1"))
                pred = pred.filter(pl.col("_new") & pl.col(self.target).is_null()).with_columns(
                    pl.coalesce("_p", "_p1").alias("_pred"))
            else:
                feats = self._features(work.drop("_new"), g).with_columns(work["_new"])
                rows = feats.filter(pl.col("_new") & pl.col(self.target).is_null())
                X, _ = self._xy(rows.drop("_new"), g)
                X["series"] = X["series"].astype(pd_categorical(self.series_levels_))
                pred = rows.with_columns(pl.Series("_pred", model.predict(X) if len(X) else [], dtype=pl.Float64))
            pred = pred.select(g, self.time_col, "_pred")
            work = work.join(pred, on=[g, self.time_col], how="left").with_columns(
                pl.coalesce(self.target, "_pred").alias(self.target)).drop("_pred")
            outputs.append(pred.rename({"_pred": "forecast"}).with_columns(pl.lit(step).alias("step")))
        return pl.concat(outputs).select(g, self.time_col, "step", "forecast").sort([g, "step"])

    def _offset(self, steps: int) -> str:
        step = _step(self.freq_)
        n, unit = int("".join(ch for ch in step if ch.isdigit())), step.lstrip("0123456789")
        return f"{n * steps}{unit}"

    # ── fit / backtest ──────────────────────────────────────────────────────
    def fit(self, df) -> "Forecaster":
        frame = self._prepare(df)
        g = self.group or "__series__"
        self.freq_ = self.freq or infer_frequency(frame[self.time_col])
        self.season_ = self.season or _DEFAULT_SEASON[self.freq_]
        self.lags_ = self._lags()
        self.series_levels_ = sorted(frame[g].unique().to_list())
        self.backtest_ = self._backtest(frame, g)
        self.model_name_ = self.backtest_.model
        self.model_ = self._fit_gbm(frame, g) if self.model_name_ == "gradient_boosting" else None
        self.history_ = frame
        return self

    def _backtest(self, frame: pl.DataFrame, g: str) -> BacktestReport:
        times = frame[self.time_col].unique().sort()
        needed = self.horizon * self.folds + max(self._lags()) + 10
        if times.len() < needed:
            raise ValueError(f"Need at least {needed} periods of history for {self.folds} backtest folds of "
                             f"{self.horizon} steps (have {times.len()})")
        results: dict = {m: [] for m in self.models}
        for k in range(self.folds, 0, -1):
            cutoff = times[-(k * self.horizon) - 1]
            train = frame.filter(pl.col(self.time_col) <= cutoff)
            test = frame.filter((pl.col(self.time_col) > cutoff) & (pl.col(self.time_col) <= times[min(
                times.len() - 1, times.len() - (k - 1) * self.horizon - 1)]))
            gbm = self._fit_gbm(train, g) if "gradient_boosting" in self.models else None
            for m in self.models:
                fc = self._forecast(m, gbm, train, g, self.horizon, test if self.known_future else None)
                joined = fc.join(test.select(g, self.time_col, self.target), on=[g, self.time_col], how="inner")
                results[m].append(joined)
        naive_scale = self._naive_scale(frame, g)
        metrics, frames = {}, {}
        for m, parts in results.items():
            allp = pl.concat(parts)
            frames[m] = allp
            metrics[m] = _metrics(allp[self.target], allp["forecast"], naive_scale)
        best = min(metrics, key=lambda m: metrics[m]["mae"])
        errs = frames[best].with_columns((pl.col(self.target) - pl.col("forecast")).alias("error"))
        alpha = 1 - self.coverage
        by_step = errs.group_by("step").agg(
            pl.col("error").abs().mean().alias("mae"),
            pl.col("error").quantile(alpha / 2, interpolation="linear").alias("q_low"),
            pl.col("error").quantile(1 - alpha / 2, interpolation="linear").alias("q_high"),
            pl.len().alias("n")).sort("step")
        self._interval_ = {r["step"]: (r["q_low"], r["q_high"]) for r in by_step.iter_rows(named=True)}
        inside = errs.join(by_step.select("step", "q_low", "q_high"), on="step").select(
            ((pl.col("error") >= pl.col("q_low")) & (pl.col("error") <= pl.col("q_high"))).mean()).item()
        return BacktestReport(best, self.folds, self.horizon, metrics, by_step, coverage=float(inside))

    def _naive_scale(self, frame, g) -> float:
        s = self.season_
        diffs = frame.with_columns((pl.col(self.target) - pl.col(self.target).shift(s).over(g)).abs().alias("d"))["d"]
        v = diffs.drop_nulls().mean()
        return float(v) if v else 0.0

    # ── predict ─────────────────────────────────────────────────────────────
    def predict(self, horizon: int | None = None, *, future=None, history=None) -> pl.DataFrame:
        """Forecasts (with ``lower``/``upper`` bounds) for the ``horizon`` periods after the end of
        the training data, or after ``history`` (newer data, same columns) if given."""
        if not hasattr(self, "model_name_"):
            raise RuntimeError("Call fit() first")
        horizon = horizon or self.horizon
        g = self.group or "__series__"
        hist = self._prepare(history) if history is not None else self.history_
        fut = None
        if future is not None:
            fut = future if isinstance(future, pl.DataFrame) else pl.from_pandas(future)
            if not self.group:
                fut = fut.with_columns(pl.lit("all").alias(g))
            if fut.schema[self.time_col] == pl.Date:
                fut = fut.with_columns(pl.col(self.time_col).cast(pl.Datetime("us")))
        fc = self._forecast(self.model_name_, self.model_, hist, g, horizon, fut)
        last = max(self._interval_)
        lo = [self._interval_.get(s, self._interval_[last])[0] * (math.sqrt(s / last) if s > last else 1)
              for s in fc["step"].to_list()]
        hi = [self._interval_.get(s, self._interval_[last])[1] * (math.sqrt(s / last) if s > last else 1)
              for s in fc["step"].to_list()]
        fc = fc.with_columns((pl.col("forecast") + pl.Series(lo)).alias("lower"),
                             (pl.col("forecast") + pl.Series(hi)).alias("upper"))
        if not self.group:
            fc = fc.drop(g)
        return fc

    def save(self, path):
        """Save to a directory (checksummed model + manifest + the history tail it needs)."""
        from .automl._persist import save_artifact
        if not hasattr(self, "model_name_"):
            raise RuntimeError("Call fit() first")
        params = {k: getattr(self, k) for k in ("target", "time_col", "horizon", "group", "freq", "season", "lags",
                                               "folds", "coverage", "known_future", "seed")}
        params["models"] = list(self.models)
        manifest = {"params": params, "freq_": self.freq_, "season_": self.season_, "lags_": self.lags_,
                    "series_levels_": self.series_levels_, "model_name_": self.model_name_,
                    "interval_": {str(k): list(v) for k, v in self._interval_.items()},
                    "backtest": {"model": self.backtest_.model, "metrics": self.backtest_.metrics,
                                 "coverage": self.backtest_.coverage}}
        out = save_artifact(path, "forecast", self.model_, manifest, None)
        g = self.group or "__series__"
        keep = max(self.lags_) + self.season_ + 3
        self.history_.group_by(g, maintain_order=True).tail(keep).write_parquet(out / "history.parquet")
        return out

    @classmethod
    def load(cls, path) -> "Forecaster":
        from pathlib import Path

        from .automl._persist import load_artifact
        manifest, model, _ = load_artifact(path, "forecast")
        obj = cls(**manifest["params"])
        obj.model_, obj.freq_, obj.season_ = model, manifest["freq_"], manifest["season_"]
        obj.lags_, obj.series_levels_, obj.model_name_ = manifest["lags_"], manifest["series_levels_"], manifest["model_name_"]
        obj._interval_ = {int(k): tuple(v) for k, v in manifest["interval_"].items()}
        bt = manifest["backtest"]
        obj.backtest_ = BacktestReport(bt["model"], obj.folds, obj.horizon, bt["metrics"], pl.DataFrame(), bt["coverage"])
        obj.history_ = pl.read_parquet(Path(path) / "history.parquet")
        return obj

    def summary(self) -> str:
        return (f"Forecaster: '{self.target}' every {self.freq_} (season {self.season_}), "
                f"{len(self.series_levels_)} series, horizon {self.horizon}\n" + self.backtest_.summary())

    def __repr__(self) -> str:
        return self.summary() if hasattr(self, "backtest_") else f"Forecaster({self.target!r}, unfitted)"


def pd_categorical(levels):
    import pandas as pd
    return pd.CategoricalDtype(categories=levels)


def reconcile_bottom_up(forecasts: pl.DataFrame, hierarchy: dict, *, group: str, time_col: str,
                        value_cols=("forecast", "lower", "upper")) -> pl.DataFrame:
    """Add parent levels to bottom-level forecasts so the hierarchy adds up.

    ``hierarchy`` maps each bottom series to its ancestors, e.g.
    ``{"store_1": ["north", "total"], "store_2": ["north", "total"], "store_3": ["south", "total"]}``.
    Point forecasts are summed exactly; interval bounds are summed too (conservative)."""
    rows = [forecasts]
    links = pl.DataFrame([{group: k, "_parent": p} for k, parents in hierarchy.items() for p in parents])
    cols = [c for c in value_cols if c in forecasts.columns]
    parents = (forecasts.join(links, on=group, how="inner").group_by(["_parent", time_col])
               .agg([pl.col(c).sum() for c in cols]).rename({"_parent": group}))
    rows.append(parents.select(forecasts.select([group, time_col, *cols]).columns))
    return pl.concat([r.select([group, time_col, *cols]) for r in rows], how="vertical_relaxed").sort([group, time_col])
