"""Public data splitting: stratified, grouped, time-ordered splits, CV folds and leakage checks."""

import datetime as dt
import warnings

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.splitting import SplitWarning, check_split, cv_folds, split_data


@pytest.fixture()
def df():
    rng = np.random.default_rng(0)
    n = 2000
    return pl.DataFrame({
        "customer": rng.integers(0, 300, n),
        "date": [dt.datetime(2026, 1, 1) + dt.timedelta(hours=int(h)) for h in rng.permutation(n)],
        "x": rng.normal(size=n),
        "label": rng.choice(["a", "b", "c"], n, p=[0.7, 0.2, 0.1]),
    })


def test_stratified_three_way_split_keeps_class_mix(df):
    s = split_data(df, target="label", test_size=0.2, val_size=0.1)
    assert s.strategy.startswith("stratified") and s.sizes() == {"train": 1400, "val": 200, "test": 400}
    overall = df["label"].value_counts(normalize=True).sort("label")["proportion"].to_numpy()
    for part in s:
        share = part["label"].value_counts(normalize=True).sort("label")["proportion"].to_numpy()
        assert np.abs(share - overall).max() < 0.02
    assert sum(s.sizes().values()) == df.height and not s.issues
    assert "Leakage checks: passed" in s.report()


def test_group_split_never_shares_a_group(df):
    s = split_data(df, target="label", group="customer", test_size=0.25, val_size=0.1)
    tr, va, te = (set(p["customer"].to_list()) for p in s)
    assert not (tr & te) and not (tr & va) and not (va & te)
    assert 0.2 < s.test.height / df.height < 0.32 and not s.issues
    s2 = split_data(df, group="customer", test_size=0.25)          # no target: plain grouped split
    assert not set(s2.train["customer"]) & set(s2.test["customer"])


def test_time_split_with_gap(df):
    s = split_data(df, target="x", time="date", test_size=0.2, val_size=0.1, gap="2d")
    assert s.train["date"].max() < s.val["date"].min() and s.val["date"].max() < s.test["date"].min()
    assert s.val["date"].min() - s.train["date"].max() >= dt.timedelta(days=2)
    assert s.test["date"].min() - s.val["date"].max() >= dt.timedelta(days=2)
    assert s.dropped_for_gap > 0 and s.test.height == 400 and not s.issues
    rows = split_data(df, time="date", test_size=0.2, gap=10)                 # gap in rows
    assert rows.dropped_for_gap == 10 and rows.train.height == 1590


def test_check_split_finds_leaks(df):
    tr, te = df.head(1500), df.tail(600)                             # 100 identical rows overlap
    issues = check_split(tr, te, group="customer", time="date", target="label")
    text = " | ".join(issues)
    assert "exact copies" in text and "'customer' values appear in both" in text and "older than" in text
    with pytest.warns(SplitWarning, match="single row"):
        tiny = df.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("rare")).otherwise("label")
                               .alias("label"))
        s = split_data(tiny, target="label")
    assert "rare" in s.train["label"].to_list()


def test_cv_folds_follow_the_same_rules(df):
    folds = list(cv_folds(df, target="label", n_splits=4, group="customer"))
    assert len(folds) == 4
    for tr, te in folds:
        assert not set(tr["customer"]) & set(te["customer"])
    times = list(cv_folds(df, time="date", n_splits=3, gap=5, indices=True))
    dates = df["date"].to_numpy()
    for a, b in times:
        assert dates[a].max() < dates[b].min()
    covered = np.concatenate([b for _, b in cv_folds(df, target="label", n_splits=5, indices=True)])
    assert sorted(covered.tolist()) == list(range(df.height))


def test_save_and_top_level_export(df, tmp_path):
    s = ac.split_data(df, target="label")
    out = s.save(tmp_path / "split")
    assert pl.read_parquet(out / "test.parquet").equals(s.test)
    assert '"strategy"' in (out / "split.json").read_text()
    with pytest.raises(ValueError, match="no training rows"):
        split_data(df, test_size=0.9, val_size=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split_data(df, target="label")                              # clean split: no warnings
