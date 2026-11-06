"""Tests for target-aware pipelines and the feature-engineering module."""

import datetime as dt
import math
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import automatedcleaning as ac

ROOT = Path(__file__).resolve().parent.parent


# ── core: target protection ──────────────────────────────────────────────────

def test_pipeline_hides_target_from_ordinary_steps():
    df = pl.DataFrame({"x": [1.0, None, 3.0, 4.0], "y": [None, 1.0, 0.0, 1000.0]})
    out = ac.Pipeline([ac.Impute(), ac.ClipOutliers()], target="y").fit_transform(df)
    assert out["y"].to_list() == [None, 1.0, 0.0, 1000.0]        # not imputed, not clipped
    assert out["x"].null_count() == 0


def test_row_filter_keeps_target_aligned():
    df = pl.DataFrame({"x": [10.0, 11, 12, 13, 14, 15, 16, 17, 18, 1000], "y": list(range(10))})
    out = ac.Pipeline([ac.RemoveOutliers()], target="y").fit_transform(df)
    assert out["y"].to_list() == list(range(9)) and 1000.0 not in out["x"].to_list()


def test_row_changing_step_without_row_mask_is_rejected():
    df = pl.DataFrame({"x": [1, 1, 2], "y": [0, 1, 0]})
    with pytest.raises(ac.PipelineStepError, match="row_mask"):
        ac.Pipeline(["drop_duplicates_subset"], target="y").fit(df)


def test_duplicates_include_the_target():
    df = pl.DataFrame({"x": [1, 1, 2], "y": [0, 1, 0]})
    assert ac.Pipeline([ac.DropDuplicates()], target="y").fit_transform(df).height == 3


def test_target_may_be_absent_when_predicting(tmp_path):
    pipe = ac.Pipeline([ac.Impute()], target="y").fit(pl.DataFrame({"x": [1.0, None, 3.0], "y": [0, 1, 0]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ac.PipelineWarning)       # no "missing column" warning
        out = pipe.transform(pl.DataFrame({"x": [None]}))
    assert out["x"].to_list() == [2.0]
    assert ac.Pipeline.load(pipe.save(tmp_path / "p.json")).target == "y"


def test_fit_requires_the_target():
    with pytest.raises(ValueError, match="not in the training data"):
        ac.Pipeline([ac.Impute()], target="y").fit(pl.DataFrame({"x": [1]}))


# ── dates ────────────────────────────────────────────────────────────────────

def test_datetime_features():
    df = pl.DataFrame({"d": [dt.datetime(2024, 1, 6, 13, 30), dt.datetime(2024, 1, 31),
                             dt.datetime(2024, 3, 1, 8)]})
    out = ac.Pipeline([ac.DateTimeFeatures()]).fit_transform(df)
    assert "d" not in out.columns
    assert out["d_dayofweek"].to_list() == [6, 3, 5]             # Sat, Wed, Fri
    assert out["d_is_weekend"].to_list() == [1, 0, 0]
    assert out["d_is_month_end"].to_list() == [0, 1, 0]
    assert out["d_is_month_start"].to_list() == [0, 0, 1]
    assert out["d_hour"].to_list() == [13, 0, 8]
    assert out["d_elapsed_days"][0] == 0.0
    assert math.isclose(out["d_month_sin"][0], math.sin(2 * math.pi / 12))


def test_dates_without_time_get_no_hour_features():
    df = pl.DataFrame({"d": [dt.date(2024, 1, 1), dt.date(2024, 6, 1)]})
    out = ac.Pipeline([ac.DateTimeFeatures()]).fit_transform(df)
    assert "d_hour" not in out.columns and "d_hour_sin" not in out.columns
    assert out["d_elapsed_days"].to_list() == [0.0, 152.0]


# ── target encoding ──────────────────────────────────────────────────────────

def test_target_encode_is_out_of_fold_with_prior_for_unseen():
    df = pl.DataFrame({"c": ["a"] * 50 + ["b"] * 50, "y": [1] * 40 + [0] * 10 + [0] * 45 + [1] * 5})
    pipe = ac.Pipeline([ac.TargetEncode(smoothing=0)], target="y")
    trained = pipe.fit_transform(df)
    new = pipe.transform(pl.DataFrame({"c": ["a", "b", "zzz", None]}))
    assert new["c"].to_list() == pytest.approx([0.8, 0.1, 0.45, 0.45])
    # training rows were encoded from the other folds, not with their own labels
    assert not trained["c"].equals(pipe.transform(df.drop("y"))["c"])


def test_target_encode_multiclass_makes_one_column_per_class():
    df = pl.DataFrame({"c": ["a", "a", "b", "b"] * 10, "y": ["x", "y", "z", "x"] * 10})
    out = ac.Pipeline([ac.TargetEncode()], target="y").fit_transform(df)
    assert {"c_te_x", "c_te_y", "c_te_z"} <= set(out.columns) and "c" not in out.columns


def test_target_encode_skips_identifiers():
    df = pl.DataFrame({"name": [f"n{i}" for i in range(20)], "y": [0, 1] * 10})
    step = ac.TargetEncode()
    ac.Pipeline([step], target="y").fit(df)
    assert step.state_["maps"] == {}


# ── generators ───────────────────────────────────────────────────────────────

def test_interactions():
    df = pl.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 0, 1, 4]})
    out = ac.Pipeline([ac.Interactions()]).fit_transform(df)
    assert out["a_x_b"].to_list() == [2.0, 0.0, 3.0, 16.0]
    assert out["a_div_b"].to_list() == [0.5, None, 3.0, 1.0]


def test_group_aggregations_exclude_target_and_handle_unseen():
    train = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 10.0, 20.0], "y": [5.0, 6.0, 7.0, 8.0]})
    pipe = ac.Pipeline([ac.GroupAggregations(group_by=["g"], aggregations=["mean"])], target="y").fit(train)
    out = pipe.transform(pl.DataFrame({"g": ["b", "a", "zz"], "v": [0.0, 0.0, 0.0]}))
    assert out["v_mean_by_g"].to_list() == [15.0, 2.0, None]
    assert out["g_count"].to_list() == [2, 2, None]
    assert "y_mean_by_g" not in out.columns


def test_lag_features_per_group_preserve_row_order():
    df = pl.DataFrame({"t": [3, 1, 2, 1, 2], "g": ["x", "x", "x", "y", "y"],
                       "y": [30.0, 10.0, 20.0, 100.0, 200.0]})
    pipe = ac.Pipeline([ac.LagFeatures(time_col="t", group_by="g", lags=[1], windows=[2])], target="y")
    out = pipe.fit_transform(df)
    assert out["y_lag1"].to_list() == [20.0, None, 10.0, None, 100.0]
    assert out["y_roll2"].to_list() == [15.0, None, 10.0, None, 100.0]
    assert pipe.transform(df.drop("y"))["y_lag1"].null_count() == 5     # no history of the target


# ── text ─────────────────────────────────────────────────────────────────────

def test_text_embeddings_are_deterministic_and_meaningful(tmp_path):
    texts = ["the cat sat on the mat", "dogs chase the cat", "stock prices rose sharply today",
             "markets fell as prices dropped", "the dog sat on the mat"] * 4
    df = pl.DataFrame({"review": texts})
    pipe = ac.Pipeline([ac.TextEmbeddings(columns=["review"], n_components=3)])
    out = pipe.fit_transform(df)
    assert out.columns == ["review_emb0", "review_emb1", "review_emb2"]
    assert ac.Pipeline.load(pipe.save(tmp_path / "p.json")).transform(df).equals(out)
    v = out.to_numpy()
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(v[0], v[4]) > cos(v[0], v[2])         # "cat on mat" is closer to "dog on mat" than to markets


def test_text_embeddings_auto_detect_text_columns():
    reviews = [f"customer {i} wrote a long review about product quality and delivery speed {i}" for i in range(30)]
    out = ac.Pipeline([ac.TextEmbeddings(n_components=2)]).fit_transform(
        pl.DataFrame({"review": reviews, "plan": ["a", "b"] * 15}))
    assert "review" not in out.columns and "plan" in out.columns and "review_emb0" in out.columns


def test_sentence_transformers_method(monkeypatch):
    class FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, **_):
            return np.array([[len(t), t.count(" ")] for t in texts], dtype=float)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    out = ac.Pipeline([ac.TextEmbeddings(columns=["t"], method="sentence_transformers")]).fit_transform(
        pl.DataFrame({"t": ["a b c", "hello world"]}))
    assert out["t_emb0"].to_list() == [5.0, 11.0] and out["t_emb1"].to_list() == [2.0, 1.0]


# ── selection & leakage ──────────────────────────────────────────────────────

def test_select_features_keeps_signal_and_drops_noise():
    rng = np.random.default_rng(0)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pl.DataFrame({"x1": x1, "x2": x2, **{f"noise{i}": rng.normal(size=n) for i in range(6)},
                       "const": [1.0] * n, "label": ["k"] * n, "y": (x1 + 0.5 * x2 > 0).astype(int)})
    step = ac.SelectFeatures(max_features=2)
    out = ac.Pipeline([step], target="y").fit_transform(df)
    assert set(step.state_["selected"]) == {"x1", "x2"} and step.state_["constant"] == ["const"]
    assert set(out.columns) == {"x1", "x2", "label", "y"}


@pytest.fixture
def leaky():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, n)
    return pl.DataFrame({
        "feature": rng.normal(size=n) + y * 0.5,
        "refund_issued": y * 10 + rng.normal(0, 0.01, n),      # recorded after the outcome
        "status_text": np.where(y == 1, "closed", "open"),       # a text copy of the label
        "churn_risk": y * 0.3 + rng.normal(0, 1, n),             # name mentions the target
        "plan": rng.choice(["a", "b", "c"], n),
        "churn": y,
    })


def test_detect_leakage(leaky):
    report = ac.detect_leakage(leaky, "churn")
    assert set(report.leaky_columns) == {"refund_issued", "status_text"}
    medium = {f.column for f in report.findings if f.severity == "medium"}
    assert medium == {"churn_risk"}
    assert report.metric == "roc_auc" and report.to_frame().height == len(report.findings)


def test_drop_leakage_step(leaky):
    pipe = ac.Pipeline([ac.DropLeakage()], target="churn")
    out = pipe.fit_transform(leaky)
    assert not {"refund_issued", "status_text"} & set(out.columns) and "churn" in out.columns
    assert "refund_issued" not in pipe.transform(leaky.drop("churn")).columns
    with pytest.raises(ac.PipelineStepError) as excinfo:
        ac.Pipeline([ac.DropLeakage(on_detect="raise")], target="churn").fit(leaky)
    assert isinstance(excinfo.value.__cause__, ac.LeakageError)


# ── default pipeline ─────────────────────────────────────────────────────────

def test_default_feature_pipeline_shape_and_overrides():
    names = lambda p: [s.label for s in p]
    assert names(ac.default_feature_pipeline("y")) == [
        "datetime_features", "drop_leakage", "text_embeddings", "target_encode",
        "encode_categorical", "select_features"]
    custom = ac.default_feature_pipeline("y", interactions=True, select_features=False)
    assert "interactions" in names(custom) and "select_features" not in names(custom)
    ts = ac.default_feature_pipeline("y", time_col="t")
    assert names(ts)[0] == "lag_features" and "target_encode" not in names(ts)
    with pytest.raises(ValueError, match="Unknown feature step"):
        ac.default_feature_pipeline("y", bogus=True)


def test_cleaning_plus_features_on_titanic():
    path = ROOT / "titanic.csv"
    if not path.exists():
        pytest.skip("titanic.csv not available")
    df = pl.read_csv(path, infer_schema_length=10000)
    pipe = ac.Pipeline([*ac.default_cleaning_pipeline().steps,
                        *ac.default_feature_pipeline("Survived").steps], target="Survived")
    train = pipe.fit_transform(df.head(700))
    assert train["Survived"].null_count() == 0 and set(train["Survived"].unique()) <= {0, 1}
    assert all(t.is_numeric() for c, t in train.schema.items())
    new = pipe.transform(df.tail(191).drop("Survived"))
    assert new.columns == [c for c in train.columns if c != "Survived"]
