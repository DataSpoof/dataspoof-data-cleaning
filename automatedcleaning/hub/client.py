"""Python client for the Hub (standard library only).

>>> hub = HubClient("https://hub.example.com", token=os.environ["AC_HUB_TOKEN"])
>>> run = hub.start_run("churn", experiment="weekly", name="automl")
>>> hub.log(run, params={"time_budget": 60}, metrics={"test_roc_auc": 0.91})
>>> v = hub.register("churn", "churn-model", model, run_id=run["id"], metrics={"test_roc_auc": 0.91})
>>> hub.promote("churn", "churn-model", v["version"], "production", reason="beats v3")  # -> pending approval
>>> model = hub.load_model("churn", "churn-model", stage="production")
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class HubError(Exception):
    def __init__(self, status: int, detail):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


class HubClient:
    def __init__(self, url: str | None = None, token: str | None = None, *, timeout: float = 60.0):
        self.url = (url or os.environ.get("AC_HUB_URL", "")).rstrip("/")
        self.token = token or os.environ.get("AC_HUB_TOKEN")
        if not self.url or not self.token:
            raise ValueError("Pass url= and token= (or set AC_HUB_URL and AC_HUB_TOKEN)")
        if self.url.startswith("http://") and not urllib.parse.urlsplit(self.url).hostname in ("localhost", "127.0.0.1"):
            raise ValueError("Use https:// for a remote Hub (tokens must not travel in clear text)")
        self.timeout = timeout

    def _request(self, method: str, path: str, *, body=None, data: bytes | None = None, content_type=None,
                 raw: bool = False):
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if body is not None:
            data, content_type = json.dumps(body, default=str).encode(), "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:   # noqa: S310 - http(s) only
                payload = resp.read()
                return payload if raw else (json.loads(payload) if payload else None)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail")
            except Exception:
                detail = exc.reason
            raise HubError(exc.code, detail) from None

    # identity
    def me(self) -> dict:
        return self._request("GET", "/api/me")

    # runs
    def start_run(self, workspace: str, *, experiment: str, name: str, params: dict | None = None,
                  tags: dict | None = None) -> dict:
        return self._request("POST", f"/api/workspaces/{workspace}/runs",
                             body={"experiment": experiment, "name": name, "params": params or {}, "tags": tags or {}})

    def log(self, run, *, params=None, metrics=None, tags=None, datasets=None, status=None) -> dict:
        run_id = run["id"] if isinstance(run, dict) else run
        body = {k: v for k, v in {"params": params, "metrics": metrics, "tags": tags, "datasets": datasets,
                                  "status": status}.items() if v is not None}
        return self._request("PATCH", f"/api/runs/{run_id}", body=body)

    def end_run(self, run, status: str = "finished") -> dict:
        return self.log(run, status=status)

    # models
    def register(self, workspace: str, name: str, model, *, run_id: str | None = None, metrics: dict | None = None,
                 description: str | None = None) -> dict:
        """Upload a fitted model (or a saved model directory) as a new version (stage ``dev``)."""
        from .storage import tar_directory
        if isinstance(model, (str, Path)):
            blob = tar_directory(model)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                blob = tar_directory(model.save(Path(tmp) / "model"))
        fields = {"run_id": run_id, "metrics": json.dumps(metrics) if metrics else None, "description": description}
        boundary = uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in fields.items():
            if v is not None:
                buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"artifact\"; filename=\"model.tar.gz\"\r\n"
                  "Content-Type: application/gzip\r\n\r\n".encode())
        buf.write(blob)
        buf.write(f"\r\n--{boundary}--\r\n".encode())
        return self._request("POST", f"/api/workspaces/{workspace}/models/{name}/versions", data=buf.getvalue(),
                             content_type=f"multipart/form-data; boundary={boundary}")

    def versions(self, workspace: str, name: str) -> list:
        return self._request("GET", f"/api/workspaces/{workspace}/models/{name}/versions")

    def promote(self, workspace: str, name: str, version: int, stage: str, *, reason: str | None = None,
                evidence: dict | None = None) -> dict:
        return self._request("POST", f"/api/workspaces/{workspace}/models/{name}/versions/{version}/transitions",
                             body={"stage": stage, "reason": reason, "evidence": evidence or {}})

    def pending_approvals(self, workspace: str) -> list:
        return self._request("GET", f"/api/workspaces/{workspace}/promotions?status=pending")

    def approve(self, request_id: str, note: str | None = None) -> dict:
        return self._request("POST", f"/api/promotions/{request_id}/approve", body={"note": note})

    def reject(self, request_id: str, note: str | None = None) -> dict:
        return self._request("POST", f"/api/promotions/{request_id}/reject", body={"note": note})

    def download(self, workspace: str, name: str, *, version: int | None = None, stage: str = "production",
                 dest=None) -> Path:
        """Download and safely extract a model version; returns its directory."""
        from .storage import sha256, untar_to
        if version is None:
            version = self._request("GET", f"/api/workspaces/{workspace}/models/{name}/resolve?stage={stage}")["version"]
        info = next(v for v in self.versions(workspace, name) if v["version"] == version)
        blob = self._request("GET", f"/api/workspaces/{workspace}/models/{name}/versions/{version}/artifact", raw=True)
        if sha256(blob) != info["artifact_sha256"]:
            raise HubError(0, "Downloaded artifact failed its checksum")
        dest = Path(dest or tempfile.mkdtemp(prefix=f"{name}-v{version}-"))
        return untar_to(blob, dest)

    def load_model(self, workspace: str, name: str, *, version: int | None = None, stage: str = "production"):
        from ..automl import load_model
        return load_model(self.download(workspace, name, version=version, stage=stage))

    def report_monitoring(self, workspace: str, name: str, report, version: int | None = None) -> dict:
        """Send a ``Monitor.check`` result to the Hub (``Monitor.on_check(lambda r: hub.report_monitoring(...))``)."""
        return self._request("POST", f"/api/workspaces/{workspace}/models/{name}/monitoring", body={
            "version": version, "rows": report.rows, "drift_share": report.data_drift.drift_share,
            "metric": (report.performance or {}).get("metric"),
            "metric_value": (report.performance or {}).get("current"),
            "alerts": [{"severity": a.severity, "kind": a.kind, "message": a.message} for a in report.alerts],
            "retrain_recommended": report.retrain_recommended})
