"""Integrations tested against the REAL tools, not stand-ins:

* MLflow  - a real ``mlflow server`` process (SQLite backend, proxied artifact store), plus
            ``mlflow models serve`` answering HTTP requests.
* DVC     - real ``git`` + ``dvc`` repositories with a local remote (push, pull, old revisions).
* Feast   - a generated Feast repo, ``apply``-ed and materialized; results compared with ours.

Each group is skipped when its tool isn't installed (``pip install mlflow dvc feast``).
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import polars as pl
import pytest

import mlpilot as ac

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
has = lambda m: importlib.util.find_spec(m) is not None  # noqa: E731
needs_mlflow = pytest.mark.skipif(not has("mlflow"), reason="needs mlflow")
needs_dvc = pytest.mark.skipif(not (has("dvc") and shutil.which("git")), reason="needs dvc and git")
needs_feast = pytest.mark.skipif(not has("feast"), reason="needs feast")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, proc, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited ({proc.returncode}): {_tail(proc)}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"{url}: {_tail(proc)}")


def _spawn(cmd: list, log, env: dict, cwd=None):
    """Start a server with its output going to a file (an unread pipe fills up and blocks it)."""
    fh = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    proc.log_path, proc.log_fh = log, fh
    return proc


def _tail(proc) -> str:
    try:
        proc.log_fh.flush()
        return open(proc.log_path, encoding="utf-8", errors="replace").read()[-3000:]
    except Exception:
        return ""


def _env() -> dict:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {**os.environ, "PYTHONPATH": os.pathsep.join([repo, os.environ.get("PYTHONPATH", "")]),
            "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONIOENCODING": "utf-8"}


def _stop(proc) -> None:
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
    if getattr(proc, "log_fh", None):
        proc.log_fh.close()


# ═════════════════════════════════════ MLflow ═══════════════════════════════════
@pytest.fixture(scope="module")
def mlflow_uri(tmp_path_factory):
    if not has("mlflow"):
        pytest.skip("needs mlflow")
    d = tmp_path_factory.mktemp("mlflow")
    port = _free_port()
    proc = _spawn([sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(port),
                   "--backend-store-uri", f"sqlite:///{(d / 'mlflow.db').as_posix()}",
                   "--artifacts-destination", (d / "artifacts").as_uri(), "--workers", "1"],
                  d / "server.log", _env(), cwd=d)
    try:
        _wait(f"http://127.0.0.1:{port}/health", proc)
        uri = f"http://127.0.0.1:{port}"
        import mlflow
        mlflow.set_tracking_uri(uri)
        yield uri
    finally:
        _stop(proc)


@pytest.fixture(scope="module")
def churn():
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    plan = rng.choice(["basic", "pro", "team"], n)
    churned = ((x1 + 0.8 * x2 + (plan == "basic") * 0.7 + rng.normal(0, 0.5, n)) > 0.4).astype(int)
    df = pl.DataFrame({"tenure": x1, "usage": x2, "plan": plan, "churned": churned})
    model = ac.AutoML("churned", time_budget=30, cv=3, verbose=False, max_trials=1,
                      include=["logistic_regression"]).fit(df.head(220))
    return model, df.tail(80)


@needs_mlflow
def test_mlflow_export_run_and_tracker(mlflow_uri, tmp_path):
    from mlflow.tracking import MlflowClient

    from mlpilot.integrations import mlflow as acm
    tracker = ac.Tracker(tmp_path / "runs", experiment="churn-exp")
    with tracker.start_run("baseline", tags={"team": "growth"}, params={"lr": 0.1, "depth": None}) as run:
        for step, loss in enumerate([0.9, 0.6, 0.45]):
            run.log_metric("loss", loss, step=step)
        run.log_metric("auc", 0.81)
        run.log_dataset(pl.DataFrame({"a": [1, 2, 3]}), "train")
        run.log_text("hello from mlpilot", "notes.txt")
    with tracker.start_run("second") as run2:
        run2.log_metric("auc", 0.84)

    rid = acm.export_run(run, tracking_uri=mlflow_uri)
    client = MlflowClient(mlflow_uri)
    got = client.get_run(rid)
    assert got.info.status == "FINISHED" and got.info.run_name == "baseline"
    assert got.data.params == {"lr": "0.1", "depth": ""}
    assert got.data.tags["team"] == "growth" and got.data.tags[acm.SOURCE_TAG] == run.id
    assert got.data.tags["dataset.train"].startswith("3x1 ")
    hist = client.get_metric_history(rid, "loss")
    assert [(m.step, m.value) for m in sorted(hist, key=lambda m: m.step)] == [(0, 0.9), (1, 0.6), (2, 0.45)]
    assert got.data.metrics["auc"] == pytest.approx(0.81)
    assert any(a.path == "notes.txt" for a in client.list_artifacts(rid))
    local = client.download_artifacts(rid, "notes.txt", str(tmp_path))
    assert open(local, encoding="utf-8").read() == "hello from mlpilot"
    # idempotent: exporting again (alone or via the whole tracker) doesn't duplicate
    assert acm.export_run(run, tracking_uri=mlflow_uri) == rid
    mapping = acm.export_tracker(tracker, tracking_uri=mlflow_uri)
    assert mapping[run.id] == rid and set(mapping) == {run.id, run2.id}
    exp = client.get_experiment_by_name("churn-exp")
    assert len(client.search_runs([exp.experiment_id])) == 2


@needs_mlflow
def test_mlflow_log_automl_registers_a_working_pyfunc(mlflow_uri, churn):
    import mlflow

    from mlpilot.integrations import mlflow as acm
    model, test = churn
    info = acm.log_automl(model, test, tracking_uri=mlflow_uri, experiment="churn-models", register_as="churn")
    assert info.registered_model_version is not None and int(info.registered_model_version) >= 1
    run = mlflow.get_run(info.run_id)
    assert run.data.params["algorithm"] == model.best_["model"]
    assert "test_accuracy" in run.data.metrics and any(k.startswith("cv_") for k in run.data.metrics)
    X = test.drop("churned")
    expected = model.predict(X).to_list()
    pyf = mlflow.pyfunc.load_model(f"models:/churn/{info.registered_model_version}")
    assert pyf.metadata.get_input_schema() is not None                    # signature recorded
    out = pyf.predict(X.to_pandas())
    assert out["churned_pred"].tolist() == expected
    assert {"proba_0", "proba_1"} <= set(out.columns) and np.allclose(out[["proba_0", "proba_1"]].sum(axis=1), 1)
    back = acm.load_automl(info.model_uri, tracking_uri=mlflow_uri)          # back to an AutoML object
    assert isinstance(back, ac.AutoML) and back.predict(X).to_list() == expected


@needs_mlflow
def test_mlflow_export_registry_maps_stages_to_aliases(mlflow_uri, churn, tmp_path):
    from mlflow.tracking import MlflowClient

    from mlpilot.integrations import mlflow as acm
    model, test = churn
    reg = ac.ModelRegistry(tmp_path / "reg")
    reg.register(model, "retention", metrics={"test_auc": 0.80}, description="first")
    reg.register(model, "retention", metrics={"test_auc": 0.83}, description="second")
    reg.promote("retention", 1, "production")
    reg.promote("retention", 2, "staging")
    ex = test.drop("churned").head(3).to_pandas()
    m = acm.export_registry(reg, "retention", register_as="retention-mlflow", tracking_uri=mlflow_uri, example=ex)
    client = MlflowClient(mlflow_uri)
    assert client.get_model_version_by_alias("retention-mlflow", "production").version == str(m[1])
    assert client.get_model_version_by_alias("retention-mlflow", "staging").version == str(m[2])
    mv1 = client.get_model_version("retention-mlflow", str(m[1]))
    assert mv1.description == "first" and mv1.tags["stage"] == "production"
    assert client.get_run(mv1.run_id).data.metrics["test_auc"] == pytest.approx(0.80)
    # promote in our registry, sync again: no duplicates, aliases follow
    reg.promote("retention", 2, "production")
    assert acm.export_registry(reg, "retention", register_as="retention-mlflow", tracking_uri=mlflow_uri) == m
    assert len(client.search_model_versions("name='retention-mlflow'")) == 2
    assert client.get_model_version_by_alias("retention-mlflow", "production").version == str(m[2])
    assert client.get_model_version("retention-mlflow", str(m[1])).tags["stage"] == "archived"
    with pytest.raises(Exception):
        client.get_model_version_by_alias("retention-mlflow", "staging")
    prod = acm.load_automl("models:/retention-mlflow@production", tracking_uri=mlflow_uri)
    assert prod.predict(test).to_list() == model.predict(test).to_list()


@needs_mlflow
def test_mlflow_models_serve_answers_http(mlflow_uri, churn, tmp_path):
    """The registered model runs in MLflow's own scoring server."""
    from mlpilot.integrations import mlflow as acm
    model, test = churn
    info = acm.log_automl(model, test, tracking_uri=mlflow_uri, experiment="churn-serve", register_as="churn-serve")
    port = _free_port()
    scripts = os.path.dirname(sys.executable)          # `mlflow models serve` runs `uvicorn` from PATH
    env = {**_env(), "MLFLOW_TRACKING_URI": mlflow_uri, "PATH": os.pathsep.join([scripts, os.environ.get("PATH", "")])}
    for attempt in range(3):
        # On Windows, `mlflow models serve` deletes its temp download right away and can hit a
        # transient lock (WinError 32, an antivirus / indexer scan) - an MLflow-side race; retry it.
        proc = _spawn([sys.executable, "-m", "mlflow", "models", "serve", "-m",
                       f"models:/churn-serve/{info.registered_model_version}", "--env-manager", "local",
                       "--host", "127.0.0.1", "-p", str(port), "-w", "1"], tmp_path / f"serve{attempt}.log", env)
        try:
            _wait(f"http://127.0.0.1:{port}/ping", proc, timeout=180)
            break
        except RuntimeError as exc:
            _stop(proc)
            if "WinError 32" not in str(exc) or attempt == 2:
                raise
    try:
        X = test.drop("churned").head(10)
        body = json.dumps({"dataframe_split": {"columns": X.columns, "data": X.rows()}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/invocations", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            preds = json.loads(r.read())["predictions"]
        assert [p["churned_pred"] for p in preds] == model.predict(X).to_list()
    finally:
        _stop(proc)


# ══════════════════════════════════════ DVC ═════════════════════════════════════
def _force(func, path, _exc):
    os.chmod(path, 0o700)
    func(path)


@needs_dvc
def test_dvc_track_versions_push_pull_and_old_revisions(tmp_path):
    from mlpilot.integrations.dvc import DVCRepo
    repo = DVCRepo.init(tmp_path / "ml-data")
    repo.add_remote("storage", str(tmp_path / "remote"))
    df1 = pl.DataFrame({"id": [1, 2, 3], "spend": [10.0, 20.0, 30.0]})
    df2 = df1.with_columns(pl.col("spend") * 2).vstack(pl.DataFrame({"id": [4], "spend": [5.0]}))
    v1 = repo.track(df1, "data/customers.parquet", message="initial", tag="customers-v1", push=True)
    v2 = repo.track(df2, "data/customers.parquet", message="refresh", tag="customers-v2", push=True)
    same = repo.track(df2, "data/customers.parquet", message="no-op")           # unchanged: no new commit
    assert v1.md5 != v2.md5 and same.rev == v2.rev
    log = repo.versions("data/customers.parquet")
    assert log.height == 2 and log["message"].to_list() == ["refresh", "initial"]
    assert log["tags"].to_list() == ["customers-v2", "customers-v1"]
    assert (tmp_path / "ml-data" / "data" / "customers.parquet.dvc").exists()
    assert "customers.parquet" in (tmp_path / "ml-data" / "data" / ".gitignore").read_text()
    assert repo.read("data/customers.parquet").equals(df2)
    assert repo.read("data/customers.parquet", rev="customers-v1").equals(df1)
    assert repo.read("data/customers.parquet", rev=v1.rev).equals(df1)
    assert any((tmp_path / "remote").rglob("*")), "data pushed to the remote"

    # lose the cache and the workspace file: everything comes back from the remote
    shutil.rmtree(tmp_path / "ml-data" / ".dvc" / "cache", onerror=_force)    # DVC makes cache files read-only
    (tmp_path / "ml-data" / "data" / "customers.parquet").unlink()
    assert repo.read("data/customers.parquet", rev="customers-v1").equals(df1)     # streamed from remote
    repo.pull()
    assert pl.read_parquet(tmp_path / "ml-data" / "data" / "customers.parquet").equals(df2)
    assert pl.read_parquet(repo.checkout("data/customers.parquet", "customers-v1")).equals(df1)

    # lineage into a tracking run
    tracker = ac.Tracker(tmp_path / "runs")
    with tracker.start_run("train") as run:
        v = repo.log_to_run(run, "data/customers.parquet", rev="customers-v2")
    assert run.tags["dvc.data/customers.parquet"] == f"{v2.rev}:{v2.md5}" and v.md5 == v2.md5
    assert run.datasets[0]["rows"] == 4


@needs_dvc
def test_dvc_export_data_repository_history(tmp_path):
    from mlpilot.integrations.dvc import DVCRepo, export_repository
    data_repo = ac.DataRepository(tmp_path / "data_repo")
    frames = [pl.DataFrame({"k": list(range(n)), "v": [float(i * n) for i in range(n)]}) for n in (5, 8, 13)]
    for i, f in enumerate(frames, 1):
        data_repo.commit(f, "orders", message=f"load {i}")
    dvc_repo = DVCRepo.init(tmp_path / "dvc")
    out = export_repository(data_repo, "orders", dvc_repo)
    assert [v.tags for v in out] == [["orders-v1"], ["orders-v2"], ["orders-v3"]]
    assert "orders@v2#" in out[1].message and "load 2" in out[1].message
    for i, f in enumerate(frames, 1):
        assert dvc_repo.read("data/orders.parquet", rev=f"orders-v{i}").equals(data_repo.checkout("orders", i))
    data_repo.commit(frames[0].head(2), "orders", message="load 4")
    again = export_repository(data_repo, "orders", dvc_repo)                    # only the new version is added
    assert len(again) == 4 and [v.rev for v in again[:3]] == [v.rev for v in out]
    assert dvc_repo.versions("data/orders.parquet").height == 4
    assert "up to date" in dvc_repo.status().lower() or dvc_repo.status() == ""


# ══════════════════════════════════════ Feast ═══════════════════════════════════
@pytest.fixture()
def our_store(tmp_path):
    rng = np.random.default_rng(3)
    now = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None, microsecond=0)
    store = ac.FeatureStore(tmp_path / "fs")
    store.apply(ac.Entity("customer", ["customer_id"]),
                ac.Entity("shelf", ["store_id", "product_id"]),
                ac.FeatureView("customer_stats", entities=["customer"], timestamp="as_of",
                               features=["orders_30d", "spend_30d", "segment"], ttl_days=10,
                               description="rolling activity", owner="growth@example.com", tags={"tier": "gold"}),
                ac.FeatureView("shelf_stock", entities=["shelf"], timestamp="ts", features=["units"]))
    rows = []
    for c in range(1, 21):
        for days_ago in sorted(rng.choice([40, 30, 21, 14, 9, 5, 2, 1], size=4, replace=False), reverse=True):
            rows.append({"customer_id": c, "as_of": now - _dt.timedelta(days=int(days_ago), minutes=c),
                         "orders_30d": int(rng.integers(0, 20)), "spend_30d": float(rng.normal(100, 30)),
                         "segment": str(rng.choice(["a", "b", "c"]))})
    store.ingest("customer_stats", pl.DataFrame(rows))
    # one customer whose latest value is old: stale online for both systems
    store.ingest("shelf_stock", pl.DataFrame({"store_id": [1, 1, 2, 2], "product_id": ["x", "y", "x", "x"],
                                              "ts": [now - _dt.timedelta(days=d) for d in (6, 5, 4, 1)],
                                              "units": [10, 3, 7, 9]}))
    times = [now - _dt.timedelta(days=int(d), hours=int(h)) for d, h in zip(rng.integers(0, 45, 60),
                                                                             rng.integers(0, 24, 60))]
    entity_df = pl.DataFrame({"customer_id": rng.integers(1, 23, 60).tolist(),       # 21, 22 never seen
                              "event_timestamp": times, "label": rng.integers(0, 2, 60).tolist()})
    return store, entity_df, now


@needs_feast
def test_feast_export_matches_our_point_in_time_joins(our_store, tmp_path):
    from mlpilot.integrations.feast import export_to_feast, feast_features, to_feast_entity_df
    store, entity_df, now = our_store
    fs = export_to_feast(store, tmp_path / "feast_repo", project="acparity")
    assert {v.name for v in fs.list_feature_views()} == {"customer_stats", "shelf_stock"}
    cs = fs.get_feature_view("customer_stats")
    assert cs.ttl == _dt.timedelta(days=10) and cs.owner == "growth@example.com" and cs.tags == {"tier": "gold"}
    assert {e.name for e in fs.list_entities()} >= {"customer", "shelf__store_id", "shelf__product_id"}
    assert (tmp_path / "feast_repo" / "features.py").exists() and (tmp_path / "feast_repo" / "feature_store.yaml").exists()

    refs = ["customer_stats:*"]
    ours = store.get_historical_features(entity_df, refs)
    theirs = pl.from_pandas(fs.get_historical_features(to_feast_entity_df(entity_df),
                                                       feast_features(store, refs)).to_df())
    key = ["customer_id", "event_timestamp", "label"]
    theirs = theirs.with_columns(pl.col("event_timestamp").dt.replace_time_zone(None).cast(pl.Datetime("us")))
    feats = ["orders_30d", "spend_30d", "segment"]
    a = ours.sort(key).select(key + feats)
    b = theirs.sort(key).select(key + feats).cast(a.schema)
    assert a.height == 60 and b.height >= 20
    # Feast's file offline store leaves out entity rows with no value inside the TTL window;
    # we keep them (with nulls) so the training set keeps every labelled row.
    dropped = a.join(b.select(key), on=key, how="anti")
    assert dropped.height == 60 - b.height and dropped.select(feats).null_count().row(0) == (dropped.height,) * 3
    both = a.join(b.select(key), on=key, how="semi").sort(key)
    assert both.equals(b), f"point-in-time results differ: {both} vs {b}"
    assert 0 < a["spend_30d"].null_count() < 60                     # TTL and unseen customers produce nulls

    # multi-key entity view
    shelf_q = pl.DataFrame({"store_id": [1, 2, 2], "product_id": ["x", "x", "y"],
                            "event_timestamp": [now - _dt.timedelta(days=d) for d in (5.5, 2, 1)]})
    ours_s = store.get_historical_features(shelf_q, ["shelf_stock:units"]).sort("store_id", "product_id")
    theirs_s = pl.from_pandas(fs.get_historical_features(to_feast_entity_df(shelf_q),
                                                         ["shelf_stock:units"]).to_df()).sort("store_id", "product_id")
    assert ours_s["units"].to_list() == [10, 7, None]
    got = {(r["store_id"], r["product_id"]): r["units"] for r in theirs_s.iter_rows(named=True)}
    assert got.get((1, "x")) == 10 and got.get((2, "x")) == 7 and got.get((2, "y")) is None


@needs_feast
def test_feast_online_store_matches_ours(our_store, tmp_path):
    from mlpilot.integrations.feast import export_to_feast
    store, _, now = our_store
    fs = export_to_feast(store, tmp_path / "feast_repo", project="aconline")
    latest = store._offline("customer_stats").collect().sort("as_of").group_by("customer_id").last()
    fresh = latest.filter(pl.col("as_of") > now - _dt.timedelta(days=9))["customer_id"].sort().to_list()
    assert len(fresh) >= 5
    rows = [{"customer_id": c} for c in fresh]
    ours = store.get_online_features(["customer_stats:orders_30d", "customer_stats:spend_30d"], rows)
    theirs = fs.get_online_features(["customer_stats:orders_30d", "customer_stats:spend_30d"], rows).to_dict()
    assert ours["orders_30d"].to_list() == theirs["orders_30d"]
    assert np.allclose(ours["spend_30d"].to_numpy(), np.array(theirs["spend_30d"], dtype=float))
    shelf = fs.get_online_features(["shelf_stock:units"], [{"store_id": 2, "product_id": "x"}]).to_dict()
    assert shelf["units"] == [9]
