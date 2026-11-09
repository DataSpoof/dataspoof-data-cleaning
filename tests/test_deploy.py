"""Tests for deployment: scoring, batch jobs, the REST API, exports and Docker contexts."""

import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification

import mlpilot as ac
from mlpilot.deploy import load_pickle

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=2, include=["logistic_regression", "random_forest"])


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    rng = np.random.default_rng(0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(
        pl.Series("city", rng.choice(["x", "y", "z"], 300)), pl.Series("id", np.arange(300)),
        pl.Series("y", np.where(y == 1, "yes", "no")))
    return df.head(220), df.tail(80)


@pytest.fixture(scope="module")
def model(data):
    return ac.AutoML("y", **FAST).fit(data[0].drop("id"))


def call(app, method, path, body=None, key=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_LENGTH": str(len(data)),
           "CONTENT_TYPE": "application/json", "wsgi.input": io.BytesIO(data)}
    if key is not None:
        env["HTTP_X_API_KEY"] = key
    out = {}
    body = b"".join(app(env, lambda s, h: out.update(status=int(s.split()[0]), headers=dict(h))))
    return out["status"], json.loads(body), out["headers"]


def test_score_and_batch_score(tmp_path, data, model):
    _, test = data
    scored = ac.score(model, test, keep=["id"])
    assert scored.columns == ["id", "prediction", "proba_no", "proba_yes"]
    assert scored["prediction"].equals(model.predict(test).alias("prediction"))

    test.write_csv(tmp_path / "in.csv")
    test.write_parquet(tmp_path / "in.parquet")
    s1 = ac.batch_score(model, tmp_path / "in.csv", tmp_path / "out.parquet", chunk_size=25, keep=["id"])
    s2 = ac.batch_score(model, tmp_path / "in.parquet", tmp_path / "out.csv", chunk_size=30)
    assert s1["rows"] == s2["rows"] == 80 and s1["chunks"] >= 3 and s1["missing_columns"] == []
    out = pl.read_parquet(tmp_path / "out.parquet")
    assert out.height == 80 and out["prediction"].to_list() == scored["prediction"].to_list()
    assert pl.read_csv(tmp_path / "out.csv").width == test.width + 3
    assert sum(s1["prediction_counts"].values()) == 80
    assert not (tmp_path / "out.parquet.partial").exists()
    with pytest.raises(ValueError, match="parquet or .csv"):
        ac.batch_score(model, tmp_path / "in.csv", tmp_path / "out.xlsx")


def test_rest_api(tmp_path, data, model):
    _, test = data
    app = ac.make_app(model, api_key="s3cret", name="m", version=3, max_rows=5,
                      prediction_log=tmp_path / "pred.jsonl")
    rec = [{k: v for k, v in r.items() if k != "y"} for r in test.head(3).to_dicts()]
    assert call(app, "GET", "/health")[:2] == (200, {"status": "ok"})          # no key needed
    assert call(app, "GET", "/metadata")[0] == 401
    assert call(app, "GET", "/metadata", key="wrong")[0] == 401
    status, meta, _ = call(app, "GET", "/metadata", key="s3cret")
    assert status == 200 and meta["classes"] == ["no", "yes"] and "city" in meta["input_columns"]
    status, res, headers = call(app, "POST", "/predict", {"records": rec}, key="s3cret")
    assert status == 200 and res["n"] == 3 and res["version"] == 3 and "X-Request-ID" in headers
    assert res["predictions"] == model.predict(pl.DataFrame(rec)).to_list()
    assert abs(sum(res["probabilities"][0].values()) - 1) < 1e-9
    assert call(app, "POST", "/predict", rec[0], key="s3cret")[1]["n"] == 1    # a single record
    assert call(app, "POST", "/predict", {"records": rec * 2}, key="s3cret")[0] == 413
    assert call(app, "POST", "/predict", raw=b"{nope", key="s3cret")[0] == 400
    assert call(app, "POST", "/predict", {"records": []}, key="s3cret")[0] == 422
    assert call(app, "GET", "/predict", key="s3cret")[0] == 405
    assert call(app, "GET", "/admin", key="s3cret")[0] == 404
    logged = [json.loads(line) for line in (tmp_path / "pred.jsonl").read_text().splitlines()]
    assert len(logged) == 4 and logged[0]["input"]["city"] == rec[0]["city"]
    frame = ac.monitoring.read_prediction_log(tmp_path / "pred.jsonl")
    assert frame.height == 4 and "a" in frame.columns


def test_dev_server_over_http(data, model):
    from wsgiref.simple_server import WSGIServer, make_server
    _, test = data
    app = ac.make_app(model)
    httpd = make_server("127.0.0.1", 0, app, server_class=WSGIServer)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}"
        rec = [{k: v for k, v in r.items() if k != "y"} for r in test.head(2).to_dicts()]
        req = urllib.request.Request(url + "/predict", data=json.dumps({"records": rec}).encode(),
                                     headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=30).read())
        assert len(res["predictions"]) == 2
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(url + "/nope", timeout=10)
        assert err.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_exports(tmp_path, data, model):
    _, test = data
    bundle = ac.export_model(model, tmp_path / "bundle")
    assert ac.load_model(bundle).predict(test).equals(model.predict(test))
    pkl = ac.export_model(model, tmp_path / "model", format="pickle")
    assert pkl.suffix == ".joblib" and load_pickle(pkl).predict(test).equals(model.predict(test))
    pkl.write_bytes(pkl.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_pickle(pkl)
    with pytest.raises(ValueError, match="format"):
        ac.export_model(model, tmp_path / "x", format="tflite")


def test_onnx_export_matches_original(tmp_path, data):
    pytest.importorskip("skl2onnx")
    ort = pytest.importorskip("onnxruntime")
    train, test = data
    m = ac.AutoML("y", **{**FAST, "include": ["logistic_regression"], "ensemble": False}).fit(train.drop("id"))
    path = ac.export_model(m, tmp_path / "onnx", format="onnx", sample=test)      # verifies internally
    io_spec = json.loads((path.parent / "io.json").read_text())
    assert path.name == "model.onnx" and (path.parent / "preprocess.json").exists()
    assert {i["name"] for i in io_spec["inputs"]} == set(m.feature_kinds_)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    labels = sess.run(None, ac.deploy.onnx_inputs(m, test))[0].ravel()
    assert (np.array(m.classes_)[labels.astype(int)] == m.predict(test).to_numpy()).mean() >= 0.99


def test_build_docker_context(tmp_path, data, model):
    out = ac.build_docker(model, tmp_path / "ctx", name="Churn Model!", version="7", port=9000)
    docker = (out / "Dockerfile").read_text()
    assert "USER app" in docker and "HEALTHCHECK" in docker and "AC_PORT=9000" in docker
    assert '"mlpilot.deploy", "container"' in docker and "AC_MODEL_VERSION=\"7\"" in docker
    assert "docker build -t churn-model" in docker
    reqs = (out / "requirements.txt").read_text()
    assert "mlpilot" in reqs and "gunicorn" in reqs and "scikit-learn==" in reqs
    assert ac.load_model(out / "model").predict(data[1]).len() == 80
    local = tmp_path / "mlpilot-9.9.9.tar.gz"
    local.write_bytes(b"fake")
    out2 = ac.build_docker(model, tmp_path / "ctx2", library_spec=str(local))
    assert (out2 / "dist" / local.name).exists()
    assert "/tmp/dist/mlpilot-9.9.9.tar.gz" in (out2 / "requirements.txt").read_text()


def test_wsgi_entry_point(tmp_path, monkeypatch, model, data):
    import importlib
    path = model.save(tmp_path / "m")
    monkeypatch.setenv("AC_MODEL_PATH", str(path))
    monkeypatch.setenv("AC_API_KEY", "k")
    import mlpilot.deploy.wsgi as wsgi
    wsgi = importlib.reload(wsgi)
    assert call(wsgi.app, "GET", "/metadata", key="k")[0] == 200
    assert call(wsgi.app, "GET", "/metadata")[0] == 401


def test_cli(tmp_path, data, model, capsys):
    from mlpilot.deploy.cli import main
    _, test = data
    path = model.save(tmp_path / "m")
    test.write_csv(tmp_path / "in.csv")
    assert main(["score", "--model", str(path), "--input", str(tmp_path / "in.csv"),
                 "--output", str(tmp_path / "o.parquet"), "--keep", "id"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 80
    reg = ac.ModelRegistry(tmp_path / "reg")
    reg.register(model, "m", stage="staging")
    assert main(["--registry", str(tmp_path / "reg"), "docker", "--model", "m:staging",
                 "--out", str(tmp_path / "ctx")]) == 0
    assert (tmp_path / "ctx" / "Dockerfile").exists()
