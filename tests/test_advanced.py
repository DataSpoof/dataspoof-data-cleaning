"""Forecasting, conformal AutoML, Optuna search, imbalance, anomaly detection, causal effects,
uplift and the orchestration integrations."""

import datetime as dt
import importlib
import json
import sys
import types

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression

import automatedcleaning as ac
from automatedcleaning.causal import UpliftModel, estimate_effect, qini
from automatedcleaning.integrations import tasks

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=3)


# ── forecasting ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sales():
    rng = np.random.default_rng(0)
    days = pl.date_range(dt.date(2023, 1, 1), dt.date(2024, 6, 30), eager=True)
    n = days.len()
    t = np.arange(n)
    frames = [pl.DataFrame({"store": f"s{k}", "date": days,
                            "sales": lvl + tr * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, n)})
              for k, (lvl, tr) in enumerate([(100, 0.05), (50, 0.02), (200, -0.03)])]
    return pl.concat(frames)


def test_forecaster_beats_naive_and_intervals_cover(tmp_path, sales):
    cut = dt.date(2024, 6, 17)
    fc = ac.Forecaster("sales", "date", horizon=14, group="store").fit(sales.filter(pl.col("date") < cut))
    assert fc.freq_ == "d" and fc.season_ == 7
    bt = fc.backtest_
    assert set(bt.metrics) == {"seasonal_naive", "gradient_boosting"}
    assert bt.metrics[bt.model]["mae"] <= bt.metrics["seasonal_naive"]["mae"]
    pred = fc.predict()
    assert pred.height == 42 and pred.columns == ["store", "date", "step", "forecast", "lower", "upper"]
    assert (pred["lower"] <= pred["forecast"]).all() and (pred["forecast"] <= pred["upper"]).all()
    actual = sales.filter(pl.col("date") >= cut).with_columns(pl.col("date").cast(pl.Datetime("us")))
    j = pred.join(actual, on=["store", "date"])
    assert (j["forecast"] - j["sales"]).abs().mean() < 6                     # noise sd is 3
    assert ((j["sales"] >= j["lower"]) & (j["sales"] <= j["upper"])).mean() >= 0.7
    loaded = ac.load_model(fc.save(tmp_path / "fc"))
    assert np.allclose(loaded.predict()["forecast"].to_numpy(), pred["forecast"].to_numpy())
    total = ac.reconcile_bottom_up(pred, {"s0": ["north", "total"], "s1": ["north", "total"], "s2": ["total"]},
                                   group="store", time_col="date")
    day1 = total.filter(pl.col("date") == pred["date"].min())
    v = dict(zip(day1["store"], day1["forecast"]))
    assert v["total"] == pytest.approx(v["s0"] + v["s1"] + v["s2"]) and v["north"] == pytest.approx(v["s0"] + v["s1"])


def test_forecaster_single_series_and_guards():
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2024, 4, 30), eager=True)
    df = pl.DataFrame({"day": days, "y": np.arange(days.len(), dtype=float) * 2 + 5})
    fc = ac.Forecaster("y", "day", horizon=7, models=("seasonal_naive",)).fit(df)
    out = fc.predict()
    assert "store" not in out.columns and out.height == 7 and fc.model_name_ == "seasonal_naive"
    with pytest.raises(ValueError, match="history"):
        ac.Forecaster("y", "day", horizon=60, folds=3).fit(df)


# ── AutoML upgrades ─────────────────────────────────────────────────────────

def test_conformal_regression_interval_and_optuna(tmp_path):
    pytest.importorskip("optuna")
    X, y = make_regression(1200, 5, noise=20, random_state=0)
    df = pl.DataFrame(X, schema=list("abcde")).with_columns(pl.Series("t", y))
    m = ac.AutoML("t", search="optuna", include=["ridge", "hist_gradient_boosting"], **FAST).fit(df.head(800))
    iv = m.predict_interval(df.tail(400), coverage=0.9)
    yt = df.tail(400)["t"].to_numpy()
    cover = ((yt >= iv["lower"].to_numpy()) & (yt <= iv["upper"].to_numpy())).mean()
    assert 0.85 <= cover <= 0.97
    wide = m.predict_interval(df.tail(5), coverage=0.99)
    assert (wide["upper"] - wide["lower"]).min() > (iv["upper"] - iv["lower"]).max()
    assert ac.AutoML.load(m.save(tmp_path / "m")).predict_interval(df.tail(3)).height == 3
    with pytest.raises(ValueError, match="predict_set"):
        m.predict_set(df.tail(3))


def test_conformal_sets_and_imbalance():
    X, y = make_classification(2000, 6, n_informative=4, weights=[0.93], random_state=1)
    d = pl.DataFrame(X, schema=list("abcdef")).with_columns(pl.Series("y", y))
    m = ac.AutoML("y", include=["logistic_regression", "hist_gradient_boosting"], **FAST).fit(d.head(1400))
    assert m.class_weight_ == "balanced"
    sets = m.predict_set(d.tail(600), 0.9).to_list()
    truth = d.tail(600)["y"].to_list()
    assert np.mean([t in s for t, s in zip(truth, sets)]) >= 0.85
    plain = ac.AutoML("y", imbalance=None, include=["logistic_regression"], **FAST).fit(d.head(1400))
    assert plain.class_weight_ is None
    with pytest.raises(ValueError):
        ac.AutoML("y", search="grid")


# ── anomaly detection ───────────────────────────────────────────────────────

def test_auto_anomaly_flags_and_explains():
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000), "c": rng.choice(["x", "y"], 2000)})
    det = ac.AutoAnomaly().fit(X)
    assert det.score(X)["is_anomaly"].mean() <= 0.05
    odd = pl.DataFrame({"a": [9.0, 0.1, 0.0], "b": [0.0, -8.5, 0.1], "c": ["x", "y", "x"]})
    s = det.score(odd)
    assert s["is_anomaly"].to_list() == [True, True, False]
    reasons = det.explain(odd)["reasons"].to_list()
    assert "a = 9 is unusually high" in reasons[0] and "b = -8.5 is unusually low" in reasons[1]
    with pytest.raises(ValueError):
        ac.AutoAnomaly(methods=("magic",))


# ── causal & uplift ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def campaign():
    rng = np.random.default_rng(0)
    n = 4000
    income = rng.normal(50, 15, n)
    loyal = rng.random(n) < 0.4
    p = 1 / (1 + np.exp(-(income - 50) / 25 - 0.5 * loyal))        # confounded, decent overlap
    t = (rng.random(n) < p).astype(int)
    effect = np.where(loyal, 2.0, 8.0)
    y = 20 + 0.5 * income + 5 * loyal + effect * t + rng.normal(0, 3, n)
    return pl.DataFrame({"income": income, "loyal": loyal, "discount": t, "revenue": y}), effect


def test_estimate_effect_removes_confounding(campaign):
    df, effect = campaign
    est = estimate_effect(df, "discount", "revenue")
    truth = float(effect.mean())
    assert est.ci[0] <= truth <= est.ci[1] and est.significant
    assert abs(est.naive_difference - truth) > abs(est.ate - truth)              # naive is confounded
    assert est.overlap["extreme_share"] < 0.1
    assert est.balance["smd_weighted"].max() < est.balance["smd_raw"].max()
    assert "Plain English" in est.summary() and "Assumes" in est.summary()
    with pytest.raises(ValueError, match="binary"):
        estimate_effect(df.with_columns(pl.Series("discount", np.arange(df.height) % 3)), "discount", "revenue")


def test_binary_outcome_and_uplift(campaign):
    df, effect = campaign
    df2 = df.with_columns((pl.col("revenue") > pl.col("revenue").median()).alias("high"))
    est = estimate_effect(df2.drop("revenue"), "discount", "high")
    assert 0 < est.ate < 1
    up = UpliftModel("discount", "revenue", features=["income", "loyal"]).fit(df.head(3000))
    u = up.predict_uplift(df.tail(1000)).to_numpy()
    assert np.corrcoef(u, effect[3000:])[0, 1] > 0.6
    q = qini(u, df.tail(1000)["discount"].to_numpy(), df.tail(1000)["revenue"].to_numpy())
    assert q["coefficient"] > 0 and len(q["qini"]) == 11


# ── integrations ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def churn_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("churn")
    X, y = make_classification(600, 4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("churned", y))
    df.write_parquet(d / "train.parquet")
    df.tail(150).drop("churned").write_parquet(d / "new.parquet")
    df.tail(150).write_parquet(d / "holdout.parquet")
    return d


def test_tasks_end_to_end(churn_files, tmp_path):
    d = churn_files
    out = tasks.train_and_register(d / "train.parquet", "churned", registry=str(tmp_path / "reg"), name="churn",
                                   tracker=str(tmp_path / "runs"), promote_to="production",
                                   require={"test_roc_auc": 0.55},
                                   automl_kwargs={"time_budget": 20, "max_trials": 2, "include": ["logistic_regression"]})
    assert out["version"] == 1 and out["promoted"] == "production" and out["run_id"]
    blocked = tasks.train_and_register(d / "train.parquet", "churned", registry=str(tmp_path / "reg"), name="churn",
                                       promote_to="production", require={"test_roc_auc": 1.01},
                                       automl_kwargs={"time_budget": 20, "max_trials": 1,
                                                      "include": ["logistic_regression"]})
    assert blocked["promoted"].startswith("blocked") and "test_roc_auc" in blocked["metrics"]
    scored = tasks.batch_score_from_registry(str(tmp_path / "reg"), "churn", d / "new.parquet", tmp_path / "s.parquet")
    assert scored["rows"] == 150 and scored["version"] == 1
    mon = tasks.save_monitor(str(tmp_path / "reg"), "churn", d / "holdout.parquet", tmp_path / "mon.json")
    res = tasks.monitor_and_retrain(str(tmp_path / "reg"), "churn", d / "holdout.parquet", mon)
    assert res["retrain_recommended"] is False and res["retrained"] is None


def _fake_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_airflow_operators_with_stub(monkeypatch, churn_files, tmp_path):
    class BaseOperator:
        def __init__(self, task_id=None, **kw):
            self.task_id = task_id
    _fake_module(monkeypatch, "airflow.models", BaseOperator=BaseOperator)
    monkeypatch.delitem(sys.modules, "automatedcleaning.integrations.airflow", raising=False)
    af = importlib.import_module("automatedcleaning.integrations.airflow")
    op = af.AutoMLTrainOperator(task_id="train", source=str(churn_files / "train.parquet"), target="churned",
                                registry=str(tmp_path / "reg"), name="c", promote_to="production",
                                automl_kwargs={"time_budget": 20, "max_trials": 1, "include": ["logistic_regression"]})
    assert op.task_id == "train" and "source" in op.template_fields
    assert op.execute({})["version"] == 1
    score = af.BatchScoreOperator(task_id="s", registry=str(tmp_path / "reg"), name="c",
                                  source=str(churn_files / "new.parquet"), output=str(tmp_path / "o.parquet"))
    assert score.execute({})["rows"] == 150


def test_prefect_and_dagster_with_stubs(monkeypatch):
    calls = []

    def task(fn=None, **kw):
        calls.append(("task", kw.get("name")))
        return fn

    def flow(**kw):
        calls.append(("flow", kw.get("name")))
        return lambda fn: fn
    _fake_module(monkeypatch, "prefect", task=task, flow=flow)
    monkeypatch.delitem(sys.modules, "automatedcleaning.integrations.prefect", raising=False)
    pf = importlib.import_module("automatedcleaning.integrations.prefect")
    assert pf.train_task is tasks.train_and_register and ("flow", "automatedcleaning-ml-lifecycle") in calls

    def asset(**kw):
        def wrap(fn):
            fn.asset_kw = kw
            return fn
        return wrap
    _fake_module(monkeypatch, "dagster", asset=asset, Output=lambda value, metadata=None: (value, metadata))
    monkeypatch.delitem(sys.modules, "automatedcleaning.integrations.dagster", raising=False)
    dg = importlib.import_module("automatedcleaning.integrations.dagster")
    assets = dg.ml_assets(source="x", target="y", registry="r", name="churn", score_source="a", score_output="b")
    assert [a.asset_kw["name"] for a in assets] == ["churn_trained_model", "churn_scored_batch"]


@pytest.mark.parametrize("module, extra", [("mlflow", "mlflow"), ("feast", "feast")])
def test_tool_bridges_explain_how_to_install(monkeypatch, module, extra):
    """Real behaviour is tested against the real tools in test_real_integrations.py; without the
    tool installed, importing the bridge must say which extra to install."""
    monkeypatch.setitem(sys.modules, module, None)            # simulate "not installed"
    monkeypatch.delitem(sys.modules, f"automatedcleaning.integrations.{module}", raising=False)
    with pytest.raises(ImportError, match=rf"AutomatedCleaning\[{extra}\]"):
        importlib.import_module(f"automatedcleaning.integrations.{module}")
    monkeypatch.delitem(sys.modules, f"automatedcleaning.integrations.{module}", raising=False)


def test_dvc_bridge_explains_how_to_install(monkeypatch, tmp_path):
    from automatedcleaning.integrations import dvc as acd
    monkeypatch.setattr(acd.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(acd.shutil, "which", lambda name: None)
    with pytest.raises(ImportError, match=r"AutomatedCleaning\[dvc\]"):
        acd.DVCRepo.init(tmp_path / "repo", git=False)
    with pytest.raises(acd.DVCError, match="not a DVC repository"):
        acd.DVCRepo(tmp_path)


def test_kafka_stream_scoring_at_least_once(churn_files):
    from automatedcleaning.integrations.kafka import KafkaSource, score_stream
    X, y = make_classification(300, 4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("churned", y))
    model = ac.AutoML("churned", include=["logistic_regression"], **{**FAST, "max_trials": 1}).fit(df)

    class Msg:
        def __init__(self, v):
            self.v = v

        def error(self):
            return None

        def value(self):
            return self.v

    class Consumer:
        def __init__(self):
            rows = [json.dumps({**r, "id": i}).encode() for i, r in enumerate(df.drop("churned").head(25).to_dicts())]
            self.queue = [[Msg(v) for v in rows[:10]] + [Msg(b"not json")], [Msg(v) for v in rows[10:]], []]
            self.committed = []

        def subscribe(self, topics):
            self.topics = topics

        def consume(self, num_messages, timeout):
            return self.queue.pop(0) if self.queue else []

        def commit(self, message, asynchronous):
            self.committed.append(message)
    consumer = Consumer()
    out = []
    stats = score_stream(model, KafkaSource("orders", bootstrap_servers="x", group_id="g", consumer=consumer),
                         lambda rows: out.extend(rows), key="id", max_batches=3)
    assert stats["records"] == 25 and stats["batches"] == 2 and stats["skipped"] == 1
    assert len(consumer.committed) == 2 and out[0]["id"] == 0 and "prediction" in out[0]
