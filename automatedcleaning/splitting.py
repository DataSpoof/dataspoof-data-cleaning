"""Data splitting: train / validation / test splits and cross-validation folds that don't leak.

>>> s = split_data(df, target="churned", test_size=0.2, val_size=0.1)        # stratified (classification)
>>> s = split_data(df, target="churned", group="customer_id")                # a customer is never on both sides
>>> s = split_data(df, target="sales", time="date", gap="7d")                # train on the past, test on the future
>>> s.train, s.val, s.test
>>> print(s.report())                       # sizes, class balance, time ranges, and the leakage checks
>>> check_split(my_train, my_test, group="patient_id")   # audit a split you made yourself
>>> for train, test in cv_folds(df, target="churned", n_splits=5, group="customer_id"): ...

Strategy is chosen from the arguments:

* ``time=``   chronological: the oldest rows train, the newest test (``gap`` drops the rows just
              before each boundary, so features computed over a window can't leak forward).
* ``group=``  every row of a group lands in the same split (patients, customers, sessions...).
              With a classification target, groups are also balanced by class.
* otherwise   random; stratified by the target when it is a classification target.

Every split is checked afterwards: groups appearing in two splits, time ranges overlapping, and
identical rows appearing in both train and test (duplicates are the most common silent leak).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import re
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from .core import AutomatedCleaningWarning


class SplitWarning(AutomatedCleaningWarning):
    """A split could not be made exactly as requested (e.g. a class too rare to stratify)."""


@dataclasses.dataclass
class SplitResult:
    train: pl.DataFrame
    test: pl.DataFrame
    val: pl.DataFrame | None = None
    strategy: str = "random"
    target: str | None = None
    group: str | None = None
    time: str | None = None
    dropped_for_gap: int = 0
    issues: list = dataclasses.field(default_factory=list)

    @property
    def parts(self) -> dict:
        out = {"train": self.train}
        if self.val is not None:
            out["val"] = self.val
        out["test"] = self.test
        return out

    def sizes(self) -> dict:
        return {k: v.height for k, v in self.parts.items()}

    def report(self) -> str:
        total = sum(self.sizes().values()) or 1
        lines = [f"Split strategy: {self.strategy}"]
        for k, v in self.parts.items():
            line = f"  {k:<5} {v.height:>8,} rows ({v.height / total:.0%})"
            if self.time and v.height:
                line += f"  {v[self.time].min()} .. {v[self.time].max()}"
            if self.group and v.height:
                line += f"  {v[self.group].n_unique():,} groups"
            lines.append(line)
        if self.target and _is_classification(self.train[self.target]) if self.train.height else False:
            for k, v in self.parts.items():
                dist = v[self.target].value_counts(normalize=True, sort=True).head(6)
                share = ", ".join(f"{r[0]}={r[1]:.2f}" for r in dist.iter_rows())
                lines.append(f"  {k} classes: {share}")
        if self.dropped_for_gap:
            lines.append(f"  {self.dropped_for_gap:,} rows dropped inside the gap before each boundary")
        lines.append("  Leakage checks: " + ("passed" if not self.issues else "; ".join(self.issues)))
        return "\n".join(lines)

    def save(self, directory) -> Path:
        """Write each part as Parquet plus ``split.json`` (strategy, sizes, row counts, checks)."""
        import json
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in self.parts.items():
            v.write_parquet(d / f"{k}.parquet")
        (d / "split.json").write_text(json.dumps({"strategy": self.strategy, "target": self.target, "group": self.group,
                                                  "time": self.time, "sizes": self.sizes(), "issues": self.issues,
                                                  "dropped_for_gap": self.dropped_for_gap}, indent=2), encoding="utf-8")
        return d

    def __iter__(self):
        return iter(self.parts.values())

    def __repr__(self) -> str:
        return f"SplitResult({self.strategy}, {self.sizes()}, issues={len(self.issues)})"


# ── helpers ─────────────────────────────────────────────────────────────────
def _frame(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _is_classification(s: pl.Series) -> bool:
    if s.dtype in (pl.String, pl.Boolean, pl.Categorical) or isinstance(s.dtype, pl.Enum):
        return True
    if s.dtype.is_integer():
        return s.n_unique() <= max(20, int(0.05 * s.len()))
    return False


def _gap(gap, time_dtype):
    """``gap``: rows (int), a timedelta, or a string like ``"7d"`` / ``"12h"`` / ``"30m"``."""
    if gap in (None, 0, ""):
        return None
    if isinstance(gap, (int, np.integer)):
        return int(gap)
    if isinstance(gap, _dt.timedelta):
        return gap
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([wdhms])\s*", str(gap).lower())
    if not m:
        raise ValueError(f"gap must be rows (int), a timedelta or like '7d', '12h', '30m'; got {gap!r}")
    n, unit = float(m.group(1)), m.group(2)
    return _dt.timedelta(**{{"w": "weeks", "d": "days", "h": "hours", "m": "minutes", "s": "seconds"}[unit]: n})


def _sizes(n: int, test_size, val_size) -> tuple:
    def count(x):
        if x is None or x == 0:
            return 0
        return int(round(x * n)) if isinstance(x, float) and x < 1 else int(x)
    n_test, n_val = count(test_size), count(val_size)
    if n_test <= 0 or n_test + n_val >= n:
        raise ValueError(f"test_size/val_size leave no training rows ({n} rows, test {n_test}, val {n_val})")
    return n_test, n_val


# ── splitting ───────────────────────────────────────────────────────────────
def split_data(df, target: str | None = None, *, test_size: float = 0.2, val_size: float = 0.0,
               stratify: bool | None = None, group: str | None = None, time: str | None = None, gap=None,
               seed: int = 0, shuffle: bool = True) -> SplitResult:
    """Split ``df`` into train / (val) / test. Sizes are fractions (< 1) or row counts. See the
    module docstring for how the strategy is chosen."""
    df = _frame(df)
    if group and time:
        raise ValueError("Use either group= or time= (for grouped time series, split by time: groups are then "
                         "checked for overlap and reported)")
    for col in (target, group, time):
        if col and col not in df.columns:
            raise ValueError(f"Column {col!r} not in the data")
    n_test, n_val = _sizes(df.height, test_size, val_size)
    y = df[target] if target else None
    if stratify is None:
        stratify = y is not None and _is_classification(y) and not time
    idx = np.arange(df.height)
    dropped = 0
    rng = np.random.default_rng(seed)

    if time:
        order = np.argsort(df[time].to_numpy(), kind="stable")
        ts = df[time].to_numpy()[order]
        n_train = df.height - n_test - n_val
        cut1, cut2 = n_train, n_train + n_val
        # keep equal timestamps on the same side of a boundary
        while 0 < cut1 < len(ts) and ts[cut1] == ts[cut1 - 1]:
            cut1 += 1
        cut2 = max(cut2, cut1)
        while n_val and 0 < cut2 < len(ts) and ts[cut2] == ts[cut2 - 1]:
            cut2 += 1
        parts = [order[:cut1], order[cut1:cut2], order[cut2:]]
        g = _gap(gap, df.schema[time])
        if g is not None:
            keep = []
            for i, part in enumerate(parts):
                nxt = next((p for p in parts[i + 1:] if len(p)), None)
                if nxt is None or not len(part):
                    keep.append(part)
                    continue
                if isinstance(g, int):
                    kept = part[:max(0, len(part) - g)]
                else:
                    boundary = df[time].to_numpy()[nxt].min()
                    kept = part[df[time].to_numpy()[part] < boundary - np.timedelta64(int(g.total_seconds() * 1e6), "us")]
                dropped += len(part) - len(kept)
                keep.append(kept)
            parts = keep
        strategy = f"time-ordered by '{time}'" + (f" with a gap of {gap}" if g is not None else "")
        train_i, val_i, test_i = parts
    elif group:
        groups = df[group].to_numpy()
        uniq = np.unique(groups.astype(str)) if groups.dtype == object else np.unique(groups)
        if len(uniq) < 3:
            raise ValueError(f"Need at least 3 distinct '{group}' values to split by group")
        if stratify and y is not None:
            train_i, val_i, test_i = _stratified_groups(df, y, group, n_test, n_val, rng)
            strategy = f"grouped by '{group}', stratified by '{target}'"
        else:
            gsize = df.group_by(group).len().sample(fraction=1.0, shuffle=True, seed=seed)
            test_g, val_g, acc_t, acc_v = [], [], 0, 0
            for gid, size in gsize.iter_rows():
                if acc_t < n_test:
                    test_g.append(gid)
                    acc_t += size
                elif acc_v < n_val:
                    val_g.append(gid)
                    acc_v += size
            col = df[group]
            test_i = idx[col.is_in(test_g).to_numpy()]
            val_i = idx[col.is_in(val_g).to_numpy()]
            train_i = np.setdiff1d(idx, np.concatenate([test_i, val_i]))
            strategy = f"grouped by '{group}'"
    else:
        if stratify and y is not None:
            train_i, val_i, test_i = _stratified(y, n_test, n_val, rng)
            strategy = f"stratified by '{target}'"
        else:
            perm = rng.permutation(df.height) if shuffle else idx
            test_i, val_i, train_i = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
            strategy = "random" if shuffle else "sequential"
    take = (lambda ii: df[np.sort(ii)]) if not time else (lambda ii: df[ii])
    res = SplitResult(train=take(train_i), test=take(test_i), val=take(val_i) if n_val else None, strategy=strategy,
                      target=target, group=group, time=time, dropped_for_gap=dropped)
    res.issues = check_split(res.train, res.test, res.val, group=group, time=time, target=target, _report=True)
    return res


def _stratified(y: pl.Series, n_test: int, n_val: int, rng) -> tuple:
    """Per-class proportional allocation; classes too small to split stay in train (with a warning)."""
    labels = y.cast(pl.String).fill_null("<null>").to_numpy()
    n = len(labels)
    test, val, train = [], [], []
    rare = []
    classes, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(classes, counts):
        members = rng.permutation(np.flatnonzero(labels == cls))
        if cnt < 2:
            rare.append(cls)
            train.extend(members)
            continue
        t = int(round(cnt * n_test / n))
        v = int(round(cnt * n_val / n))
        t = min(max(t, 1), cnt - 1)
        v = min(v, cnt - 1 - t)
        test.extend(members[:t])
        val.extend(members[t:t + v])
        train.extend(members[t + v:])
    if rare:
        warnings.warn(f"Classes {rare[:5]} have a single row; they can only go to the training set", SplitWarning,
                      stacklevel=3)
    return np.array(train, int), np.array(val, int), np.array(test, int)


def _stratified_groups(df, y, group, n_test, n_val, rng) -> tuple:
    """Whole groups per split, chosen greedily so each split's class mix stays close to overall."""
    labels = y.cast(pl.String).fill_null("<null>")
    classes = sorted(labels.unique().to_list())
    tab = (pl.DataFrame({"g": df[group], "y": labels}).group_by("g", "y").len()
           .pivot(on="y", index="g", values="len").fill_null(0))
    for c in classes:
        if c not in tab.columns:
            tab = tab.with_columns(pl.lit(0).alias(c))
    gids = tab["g"].to_list()
    counts = tab.select(classes).to_numpy().astype(float)
    overall = counts.sum(0) / counts.sum()
    order = rng.permutation(len(gids))
    targets = {"test": n_test, "val": n_val}
    acc = {k: np.zeros(len(classes)) for k in targets}
    chosen: dict = {"test": [], "val": []}
    for i in order:
        need = [k for k in ("test", "val") if acc[k].sum() < targets[k]]
        if not need:
            break
        # put the group where it moves that split's class mix closest to the overall mix
        best = min(need, key=lambda k: np.abs((acc[k] + counts[i]) / (acc[k].sum() + counts[i].sum()) - overall).sum())
        acc[best] += counts[i]
        chosen[best].append(gids[i])
    col = df[group]
    idx = np.arange(df.height)
    test_i = idx[col.is_in(chosen["test"]).to_numpy()]
    val_i = idx[col.is_in(chosen["val"]).to_numpy()]
    train_i = np.setdiff1d(idx, np.concatenate([test_i, val_i]))
    return train_i, val_i, test_i


# ── checking ────────────────────────────────────────────────────────────────
def check_split(train, test, val=None, *, group: str | None = None, time: str | None = None,
                target: str | None = None, ignore: list | None = None, _report: bool = False) -> list:
    """Leakage problems between splits (an empty list means none found):

    * the same ``group`` value in two splits;
    * ``time``: a later split containing rows older than the newest training row;
    * identical rows (ignoring ``target`` and ``ignore`` columns) in train and test/val;
    * a classification class present in test but never seen in train.
    """
    parts = {"train": _frame(train), "test": _frame(test)}
    if val is not None:
        parts["val"] = _frame(val)
    issues = []
    tr = parts["train"]
    for name in [k for k in parts if k != "train"]:
        other = parts[name]
        if not other.height or not tr.height:
            continue
        if group and group in tr.columns:
            shared = tr.select(group).unique().join(other.select(group).unique(), on=group, how="inner").height
            if shared:
                issues.append(f"{shared:,} '{group}' values appear in both train and {name}")
        if time and time in tr.columns:
            if other[time].min() < tr[time].max():
                n = other.filter(pl.col(time) < tr[time].max()).height
                issues.append(f"{n:,} {name} rows are older than the newest training row ('{time}')")
        cols = [c for c in tr.columns if c != target and c not in (ignore or []) and c in other.columns]
        if cols:
            dup = tr.select(cols).unique().join(other.select(cols), on=cols, how="inner", **_NULLS_EQUAL)
            if dup.height:
                issues.append(f"{dup.height:,} {name} rows are exact copies of training rows")
        if target and target in tr.columns and _is_classification(tr[target]):
            unseen = set(other[target].drop_nulls().unique().to_list()) - set(tr[target].drop_nulls().unique().to_list())
            if unseen:
                issues.append(f"{name} has classes never seen in training: {sorted(map(str, unseen))[:5]}")
    if issues and not _report:
        return issues
    if issues and _report:
        warnings.warn("Possible leakage between splits: " + "; ".join(issues), SplitWarning, stacklevel=3)
    return issues


def _nulls_equal_kw() -> dict:
    """Missing values count as equal when looking for duplicated rows (renamed in Polars 1.24)."""
    import inspect
    params = inspect.signature(pl.DataFrame.join).parameters
    return {"nulls_equal": True} if "nulls_equal" in params else {"join_nulls": True} if "join_nulls" in params else {}


_NULLS_EQUAL: dict = _nulls_equal_kw()


# ── cross-validation ────────────────────────────────────────────────────────
def cv_folds(df, target: str | None = None, *, n_splits: int = 5, stratify: bool | None = None,
             group: str | None = None, time: str | None = None, gap=0, seed: int = 0, indices: bool = False):
    """Cross-validation folds as ``(train, test)`` DataFrames (or index arrays with
    ``indices=True``), using the same strategy rules as :func:`split_data`. Time folds expand
    forward (train on everything before each test window); ``gap`` rows are skipped between."""
    from sklearn import model_selection as ms
    df = _frame(df)
    y = df[target] if target else None
    if stratify is None:
        stratify = y is not None and _is_classification(y) and not time
    X = np.zeros((df.height, 1))
    if time:
        order = np.argsort(df[time].to_numpy(), kind="stable")
        splitter = ms.TimeSeriesSplit(n_splits=n_splits, gap=int(gap or 0))
        gen = ((order[a], order[b]) for a, b in splitter.split(X))
    elif group:
        g = df[group].to_numpy()
        if stratify and y is not None:
            splitter = ms.StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            gen = splitter.split(X, y.cast(pl.String).to_numpy(), g)
        else:
            gen = ms.GroupKFold(n_splits=n_splits).split(X, groups=g)
    elif stratify and y is not None:
        gen = ms.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y.cast(pl.String).to_numpy())
    else:
        gen = ms.KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    for a, b in gen:
        yield (a, b) if indices else (df[np.sort(a)] if not time else df[a], df[np.sort(b)] if not time else df[b])
