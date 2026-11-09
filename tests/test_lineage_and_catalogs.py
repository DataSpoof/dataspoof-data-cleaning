"""OpenLineage events and the lineage graph; AWS Glue catalog + jobs (moto server, real S3
protocol); Azure Data Factory runs (against a fake ARM endpoint). Marquez itself is exercised in
test_docker_stacks.py."""

import importlib.util
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import polars as pl
import pytest

import mlpilot as ac
from mlpilot.lineage import FileTransport, Lineage, LineageGraph, dataset


# ── lineage ─────────────────────────────────────────────────────────────────
def test_run_events_and_graph(tmp_path):
    path = tmp_path / "lineage.jsonl"
    lin = Lineage(FileTransport(path), namespace="acme")
    raw = pl.DataFrame({"id": [1, 2, 3], "spend": [10.0, None, 30.0]})
    with lin.run("clean_customers", inputs=[dataset("s3://raw/customers.csv", raw)]) as run:
        run.outputs.append(dataset("warehouse.customers_clean", raw.drop_nulls()))
    with lin.run("build_features", inputs=[dataset("warehouse.customers_clean")],
                 outputs=[dataset("features.customer_stats")]):
        pass
    with lin.run("train_churn", inputs=[dataset("features.customer_stats"), dataset("warehouse.labels")],
                 outputs=[dataset("models/churn/v4")]):
        pass
    with pytest.raises(ZeroDivisionError):
        with lin.run("broken_job", inputs=[dataset("warehouse.customers_clean")], outputs=[dataset("nowhere")]):
            1 / 0
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["eventType"] for e in events] == ["START", "COMPLETE"] * 3 + ["START", "FAIL"]
    start = events[0]
    assert start["job"] == {"namespace": "acme", "name": "clean_customers", "facets": start["job"]["facets"]}
    src = start["inputs"][0]
    assert (src["namespace"], src["name"]) == ("s3://raw", "customers.csv")
    assert src["facets"]["schema"]["fields"] == [{"name": "id", "type": "Int64"}, {"name": "spend", "type": "Float64"}]
    assert src["facets"]["dataQualityMetrics"]["rowCount"] == 3
    assert src["facets"]["dataQualityMetrics"]["columnMetrics"]["spend"]["nullCount"] == 1
    assert events[1]["outputs"][0]["name"] == "warehouse.customers_clean"            # added during the run
    assert "ZeroDivisionError" in events[-1]["run"]["facets"]["errorMessage"]["stackTrace"]
    g = LineageGraph.from_file(path)
    assert g.upstream("models/churn/v4") == ["mlpilot:features.customer_stats",
                                             "mlpilot:warehouse.customers_clean",
                                             "mlpilot:warehouse.labels", "s3://raw:customers.csv"]
    assert g.downstream("mlpilot:warehouse.customers_clean") == [
        "mlpilot:features.customer_stats", "mlpilot:models/churn/v4"]
    assert "mlpilot:nowhere" not in g.downstream("mlpilot:warehouse.customers_clean")  # FAIL: no edge


def test_tracked_runs_and_dataset_versions(tmp_path):
    path = tmp_path / "l.jsonl"
    lin = Lineage(FileTransport(path))
    tracker = ac.Tracker(tmp_path / "runs", experiment="churn")
    with tracker.start_run("weekly") as run:
        run.log_dataset(pl.DataFrame({"a": [1, 2]}), "train")
        run.log_metric("auc", 0.8)
        run.log_text("card", "model_card.md")
    rid = lin.emit_tracked_run(run)
    repo = ac.DataRepository(tmp_path / "repo")
    v = repo.commit(pl.DataFrame({"a": [1, 2]}), "customers", source="s3://raw/customers.csv")
    lin.emit_dataset_version(v)
    events = [json.loads(x) for x in path.read_text().splitlines()]
    assert events[1]["run"]["runId"] == rid and events[1]["eventType"] == "COMPLETE"
    assert events[1]["inputs"][0]["facets"]["version"]["datasetVersion"] == run.datasets[0]["content_hash"]
    assert events[1]["outputs"][0]["name"] == "churn/weekly/model_card.md"
    assert events[1]["run"]["facets"]["mlpilot_run"]["metrics"]["auc"] == 0.8
    assert events[3]["outputs"][0]["facets"]["version"]["datasetVersion"].startswith("v1:")
    assert LineageGraph(events).upstream("customers") == ["s3://raw:customers.csv"]


def test_transport_failure_never_breaks_the_job(tmp_path):
    from mlpilot.lineage import HttpTransport
    lin = Lineage(HttpTransport("http://127.0.0.1:9"), FileTransport(tmp_path / "ok.jsonl"))
    with pytest.warns(Warning, match="not delivered"):
        with lin.run("job"):
            pass
    assert len((tmp_path / "ok.jsonl").read_text().splitlines()) == 2 and lin.errors
    strict = Lineage(HttpTransport("http://127.0.0.1:9"), strict=True)
    with pytest.raises(Exception):
        with strict.run("job"):
            pass


# ── AWS Glue (moto server: real S3 protocol for Polars) ─────────────────────
@pytest.fixture(scope="module")
def moto_server():
    if importlib.util.find_spec("moto") is None:
        pytest.skip("needs moto")
    import os

    from moto.server import ThreadedMotoServer
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(AWS_ACCESS_KEY_ID="testing", AWS_SECRET_ACCESS_KEY="testing", AWS_DEFAULT_REGION="us-east-1")
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


def test_glue_catalog_write_read_and_jobs(moto_server):
    import boto3

    from mlpilot.integrations import aws_glue as glue
    kw = dict(endpoint_url=moto_server, region_name="us-east-1")
    boto3.client("s3", **kw).create_bucket(Bucket="lake")
    g = boto3.client("glue", **kw)
    so = {"aws_access_key_id": "testing", "aws_secret_access_key": "testing", "aws_region": "us-east-1",
          "aws_endpoint_url": moto_server, "aws_allow_http": "true"}
    df = pl.DataFrame({"id": [1, 2, 3, 4], "spend": [1.5, 2.0, None, 4.0], "country": ["IN", "IN", "US", "US"],
                       "tags": [["a"], [], ["b", "c"], None]})
    out = glue.write_table(df, "analytics", "customers", "s3://lake/analytics/customers/", partition_by=["country"],
                           storage_options=so, glue=g, description="Cleaned customers",
                           parameters={"quality_score": 97.5}, column_comments={"spend": "30-day spend, EUR"})
    assert out["PartitionKeys"] == [{"Name": "country", "Type": "string"}]
    t = glue.describe_table("analytics", "customers", glue=g)
    cols = {c["Name"]: c for c in t["StorageDescriptor"]["Columns"]}
    assert cols["tags"]["Type"] == "array<string>" and cols["spend"]["Comment"] == "30-day spend, EUR"
    assert t["Parameters"]["quality_score"] == "97.5" and t["Parameters"]["classification"] == "parquet"
    parts = g.get_partitions(DatabaseName="analytics", TableName="customers")["Partitions"]
    assert sorted(p["Values"][0] for p in parts) == ["IN", "US"]
    assert glue.table_schema("analytics", "customers", glue=g) == {"id": pl.Int64, "spend": pl.Float64,
                                                                   "tags": pl.List(pl.String), "country": pl.String}
    back = glue.read_table("analytics", "customers", storage_options=so, glue=g).sort("id")
    assert back.select(df.columns).equals(df)
    glue.register_table({"id": pl.Int64}, "analytics", "customers", "s3://lake/x/", glue=g)       # update in place
    assert glue.describe_table("analytics", "customers", glue=g)["StorageDescriptor"]["Location"] == "s3://lake/x/"
    g.create_job(Name="nightly", Role="arn:aws:iam::123456789012:role/glue", Command={"Name": "glueetl"})
    res = glue.run_job("nightly", {"--date": "2026-11-08"}, poll=0.1, glue=g)
    assert res["state"] == "SUCCEEDED" and res["run_id"]


# ── Azure Data Factory (fake ARM endpoint) ─────────────────────────────────
@pytest.fixture()
def fake_arm():
    state = {"runs": {}, "auth": [], "bodies": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            state["auth"].append(self.headers.get("Authorization"))
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            assert "api-version=2018-06-01" in self.path
            if "/createRun" in self.path:
                name = self.path.split("/pipelines/")[1].split("/")[0]
                if name == "Missing":
                    return self._send(404, {"error": {"code": "PipelineNotFound", "message": "no such pipeline"}})
                rid = f"run-{len(state['runs']) + 1}"
                state["runs"][rid] = {"name": name, "polls": 0, "params": body}
                state["bodies"].append(body)
                return self._send(200, {"runId": rid})
            if "/queryActivityruns" in self.path:
                return self._send(200, {"value": [{"activityName": "Copy", "activityType": "Copy", "status": "Succeeded",
                                                   "durationInMs": 1200, "output": {"rowsCopied": 42}}]})
            return self._send(404, {})

        def do_GET(self):
            state["auth"].append(self.headers.get("Authorization"))
            rid = self.path.split("/pipelineruns/")[1].split("?")[0]
            run = state["runs"][rid]
            run["polls"] += 1
            status = ["Queued", "InProgress"][min(run["polls"] - 1, 1)] if run["polls"] < 3 else \
                ("Failed" if run["name"] == "Flaky" else "Succeeded")
            return self._send(200, {"runId": rid, "pipelineName": run["name"], "status": status, "durationInMs": 3000,
                                    "message": "Copy activity failed" if status == "Failed" else "",
                                    "parameters": run["params"]})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()


def test_adf_pipeline_runs(fake_arm, monkeypatch):
    from mlpilot.integrations.azure_data_factory import DataFactory, DataFactoryError
    url, state = fake_arm
    monkeypatch.setenv("ADF_TOKEN", "tok-1")
    adf = DataFactory("sub-1", "data-rg", "corp-adf", token="env://ADF_TOKEN", base_url=url)
    run = adf.run_pipeline("CopyOrders", {"date": "2026-11-08"}, poll=0.01)
    assert run["status"] == "Succeeded" and run["parameters"] == {"date": "2026-11-08"}
    assert set(state["auth"]) == {"Bearer tok-1"}
    assert adf.activity_runs(run["run_id"])[0]["output"] == {"rowsCopied": 42}
    with pytest.raises(DataFactoryError, match="Failed: Copy activity failed"):
        adf.run_pipeline("Flaky", poll=0.01)
    with pytest.raises(DataFactoryError, match="PipelineNotFound"):
        adf.start_pipeline("Missing")
    assert adf.run_pipeline("CopyOrders", wait=False)["status"] == "Queued"
