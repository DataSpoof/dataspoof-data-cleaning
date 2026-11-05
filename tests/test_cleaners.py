"""Tests for the stateful cleaning steps and the default cleaning pipeline."""

from pathlib import Path

import polars as pl
import pytest

import automatedcleaning as ac

ROOT = Path(__file__).resolve().parent.parent


def _fit(step, df):
    return ac.Pipeline([step]).fit(df)


# ── type repair ──────────────────────────────────────────────────────────────

def test_convert_numeric_text_keeps_signs_and_formats():
    df = pl.DataFrame({"amount": ["1,200", "$50", "(30)", "-7.5", "-", "12%", "NA", "3"]})
    pipe = _fit(ac.ConvertNumericText(), df)
    assert pipe.steps[0].state_["columns"] == ["amount"]
    assert pipe.transform(df)["amount"].to_list() == [1200.0, 50.0, -30.0, -7.5, None, 12.0, None, 3.0]


def test_convert_numeric_text_skips_codes():
    df = pl.DataFrame({
        "zip": ["02134", "10001", "60601", "94105", "73301"],          # code-like name
        "store": ["0012", "0045", "0078", "0099", "0101"],             # leading zeros
        "label": ["a", "b", "c", "d", "e"],
    })
    assert _fit(ac.ConvertNumericText(), df).steps[0].state_["columns"] == []


def test_parse_dates_ignores_number_like_text():
    df = pl.DataFrame({
        "joined": ["2021-01-05", "2021-02-11", "2021-03-01", "2021-04-20", "2021-05-15"],
        "ticket": ["113803", "373450", "330877", "17463", "349909"],   # dateutil would misread these
    })
    pipe = _fit(ac.ParseDates(), df)
    assert pipe.steps[0].state_["columns"] == ["joined"]
    out = pipe.transform(df)
    assert out["joined"].dtype in (pl.Date, pl.Datetime) and out["ticket"].dtype == pl.String


# ── columns ──────────────────────────────────────────────────────────────────

def test_drop_columns_reasons_and_keep():
    n = 20
    df = pl.DataFrame({
        "id": list(range(n)),
        "mostly_null": [None] * 15 + [1] * 5,
        "constant": ["x"] * n,
        "value": [float(i % 7) for i in range(n)],
        "value_copy": [float(i % 7) for i in range(n)],
        "price": [i * 1.5 for i in range(n)],                           # unique floats are fine
        "name": [f"p{i}" for i in range(n)],
    })
    reasons = _fit(ac.DropColumns(keep=["name"]), df).steps[0].state_["reasons"]
    assert reasons == {"id": "ID-like (100% unique)", "mostly_null": "75% missing",
                       "constant": "constant", "value_copy": "duplicate of 'value'"}


def test_drop_collinear():
    df = pl.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 4, 6, 8, 10], "c": [5.0, 3, 6, 1, 9]})
    pipe = _fit(ac.DropCollinear(threshold=0.95), df)
    assert pipe.transform(df).columns == ["a", "c"]


# ── missing values ───────────────────────────────────────────────────────────

def test_impute_keeps_integer_dtype_and_stable_indicators():
    pipe = _fit(ac.Impute(add_indicators=True), pl.DataFrame({"n": [1, 2, None, 4]}))
    out = pipe.transform(pl.DataFrame({"n": [1, 2, None, 4]}))
    assert out["n"].dtype == pl.Int64 and out["n"].to_list() == [1, 2, 2, 4]
    assert out["n_was_missing"].to_list() == [False, False, True, False]
    # a later batch without nulls still gets the indicator column (stable schema)
    assert "n_was_missing" in pipe.transform(pl.DataFrame({"n": [5]})).columns


def test_impute_indicator_threshold():
    df = pl.DataFrame({"rare": [None] + [1.0] * 99, "common": [None] * 10 + [1.0] * 90})
    out = _fit(ac.Impute(add_indicators=True, indicator_min_missing=0.05), df).transform(df)
    assert "common_was_missing" in out.columns and "rare_was_missing" not in out.columns


def test_impute_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        ac.Impute(numeric="knn")


# ── outliers & distributions ─────────────────────────────────────────────────

def test_clip_outliers_skips_binary_and_zero_width_columns():
    df = pl.DataFrame({
        "v": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000],
        "flag": [0, 1] * 5,
        "mostly_zero": [0] * 9 + [5],                                  # IQR = 0
    })
    pipe = _fit(ac.ClipOutliers(), df)
    assert set(pipe.steps[0].state_["bounds"]) == {"v"}
    out = pipe.transform(df)
    assert out["v"].dtype == pl.Int64 and out["v"].max() < 1000
    assert out["mostly_zero"].to_list() == df["mostly_zero"].to_list()


def test_remove_outliers_is_fit_only_and_keeps_nulls():
    df = pl.DataFrame({"v": [10.0, 11, 12, 13, 14, 15, 16, 17, None, 1000]})
    pipe = ac.Pipeline([ac.RemoveOutliers()])
    trained = pipe.fit_transform(df)
    assert 1000.0 not in trained["v"].to_list() and trained["v"].null_count() == 1
    assert pipe.transform(df).height == df.height          # skipped on new data


def test_log_transform_only_positive_skew_nonnegative():
    df = pl.DataFrame({"skewed": [1.0] * 8 + [50.0, 500.0], "negative": [-1.0] * 8 + [50.0, 500.0]})
    assert _fit(ac.LogTransformSkewed(), df).steps[0].state_["columns"] == ["skewed"]


def test_scale_uses_training_parameters():
    pipe = _fit(ac.Scale(method="minmax"), pl.DataFrame({"x": [0.0, 5.0, 10.0]}))
    assert pipe.transform(pl.DataFrame({"x": [20.0]}))["x"].to_list() == [2.0]


# ── categories ───────────────────────────────────────────────────────────────

def test_group_rare_categories_handles_unseen_and_null():
    df = pl.DataFrame({"city": ["Delhi"] * 98 + ["Goa", "Agra"]})
    pipe = _fit(ac.GroupRareCategories(min_frequency=0.05), df)
    out = pipe.transform(pl.DataFrame({"city": ["Delhi", "Goa", "Mars", None]}))
    assert out["city"].to_list() == ["Delhi", "Other", "Other", None]


def test_onehot_has_stable_columns_and_zeros_for_unseen():
    pipe = _fit(ac.EncodeCategorical(method="onehot"), pl.DataFrame({"c": ["a", "b", "a"]}))
    out = pipe.transform(pl.DataFrame({"c": ["z", "a", None]}))
    assert out.columns == ["c_a", "c_b"]
    assert out.rows() == [(0, 0), (1, 0), (0, 0)]


def test_ordinal_and_frequency_encoding_handle_unseen():
    df = pl.DataFrame({"c": ["b", "a", "a", "c"]})
    ordinal = _fit(ac.EncodeCategorical(method="ordinal"), df).transform(
        pl.DataFrame({"c": ["a", "c", "zz", None]}))
    assert ordinal["c"].to_list() == [0, 2, -1, None]
    freq = _fit(ac.EncodeCategorical(method="frequency"), df).transform(
        pl.DataFrame({"c": ["a", "zz"]}))
    assert freq["c"].to_list() == [0.5, 0.0]


# ── default pipeline ─────────────────────────────────────────────────────────

def test_default_pipeline_overrides():
    pipe = ac.default_cleaning_pipeline(clip_outliers=False, impute={"numeric": "mean"})
    names = [s.label for s in pipe]
    assert "clip_outliers" not in names
    assert pipe.steps[names.index("impute")].params["numeric"] == "mean"
    with pytest.raises(ValueError, match="Unknown default step"):
        ac.default_cleaning_pipeline(not_a_step=False)


def test_default_pipeline_dedupes_before_dropping_ids():
    # Two different customers who share every non-ID value must both survive.
    df = pl.DataFrame({"customer_id": list(range(12)),
                       "plan": ["basic", "pro"] * 6, "spend": [10.0, 20.0] * 6})
    out = ac.default_cleaning_pipeline().fit_transform(df)
    assert "customer_id" not in out.columns     # dropped as ID-like ...
    assert out.height == 12                     # ... but only after de-duplication ran


@pytest.mark.parametrize("name", ["titanic.csv", "Company_Data.csv"])
def test_default_pipeline_on_real_data(tmp_path, name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not available")
    df = pl.read_csv(path, infer_schema_length=10000)
    pipe = ac.default_cleaning_pipeline()
    clean = pipe.fit_transform(df)
    assert clean.null_count().sum_horizontal()[0] == 0
    assert clean.height <= df.height
    reloaded = ac.Pipeline.load(pipe.save(tmp_path / "pipe.json"))
    assert reloaded.transform(df).equals(pipe.transform(df))
    if name == "Company_Data.csv":
        assert df["Advertising"].dtype == pl.String and clean["Advertising"].dtype == pl.Float64
