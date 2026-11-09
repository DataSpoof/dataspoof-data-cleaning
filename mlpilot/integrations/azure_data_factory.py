"""Azure Data Factory: trigger pipelines and wait for them (Azure Resource Manager REST API).

>>> from mlpilot.integrations.azure_data_factory import DataFactory
>>> adf = DataFactory(subscription="…", resource_group="data-rg", factory="corp-adf")
>>> run = adf.run_pipeline("CopyOrdersToLake", {"date": "2026-11-08"})     # waits by default
>>> run["status"], run["duration_ms"]
>>> adf.activity_runs(run["run_id"])                                          # per-activity results

Authentication: ``token=`` (a bearer token or a secret reference such as
``vault://secret/azure#token``), else ``AZURE_ACCESS_TOKEN``, else ``azure-identity``'s
``DefaultAzureCredential`` (managed identity, workload identity, ``az login``...).
No Azure SDK is needed for the API calls themselves.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_VERSION = "2018-06-01"
_DONE = {"Succeeded", "Failed", "Cancelled"}


class DataFactoryError(RuntimeError):
    pass


class DataFactory:
    def __init__(self, subscription: str, resource_group: str, factory: str, *, token: str | None = None,
                 base_url: str = "https://management.azure.com", timeout: float = 30.0):
        from ..secretstore import resolve
        self.sub, self.rg, self.factory = subscription, resource_group, factory
        self.base = base_url.rstrip("/")
        self._token = resolve(token) or resolve(os.environ.get("AZURE_ACCESS_TOKEN"))
        self._credential: Any = None
        self.timeout = timeout

    def _bearer(self) -> str:
        if self._token:
            return self._token
        if self._credential is None:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise DataFactoryError("No token: pass token=, set AZURE_ACCESS_TOKEN, or pip install azure-identity") \
                    from None
            self._credential = DefaultAzureCredential()
        return self._credential.get_token("https://management.azure.com/.default").token

    def _url(self, path: str, **query) -> str:
        q = urllib.parse.urlencode({"api-version": API_VERSION, **query})
        return (f"{self.base}/subscriptions/{self.sub}/resourceGroups/{self.rg}/providers/Microsoft.DataFactory/"
                f"factories/{self.factory}/{path}?{q}")

    def _call(self, method: str, path: str, body=None, **query) -> dict:
        req = urllib.request.Request(self._url(path, **query), method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self._bearer()}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read()).get("error", {})
            except Exception:
                err = {}
            raise DataFactoryError(f"ADF {method} {path}: HTTP {exc.code} {err.get('code', '')} "
                                   f"{err.get('message', '')}".strip()) from None

    def start_pipeline(self, pipeline: str, parameters: dict | None = None, *, reference_run_id: str | None = None) -> str:
        query = {"referencePipelineRunId": reference_run_id, "isRecovery": "true"} if reference_run_id else {}
        res = self._call("POST", f"pipelines/{urllib.parse.quote(pipeline)}/createRun", parameters or {}, **query)
        return res["runId"]

    def pipeline_run(self, run_id: str) -> dict:
        r = self._call("GET", f"pipelineruns/{run_id}")
        return {"run_id": run_id, "pipeline": r.get("pipelineName"), "status": r.get("status"),
                "duration_ms": r.get("durationInMs"), "message": r.get("message"), "start": r.get("runStart"),
                "end": r.get("runEnd"), "parameters": r.get("parameters")}

    def run_pipeline(self, pipeline: str, parameters: dict | None = None, *, wait: bool = True, poll: float = 15.0,
                     timeout: float = 6 * 3600) -> dict:
        """Trigger ``pipeline``; with ``wait``, poll until it finishes and raise if it didn't succeed."""
        run_id = self.start_pipeline(pipeline, parameters)
        if not wait:
            return {"run_id": run_id, "pipeline": pipeline, "status": "Queued"}
        t0 = time.time()
        while True:
            info = self.pipeline_run(run_id)
            if info["status"] in _DONE:
                if info["status"] != "Succeeded":
                    raise DataFactoryError(f"ADF pipeline {pipeline} run {run_id} {info['status']}: "
                                           f"{info.get('message') or ''}".strip())
                return info
            if time.time() - t0 > timeout:
                raise DataFactoryError(f"ADF pipeline {pipeline} run {run_id} still {info['status']} after {timeout:.0f}s")
            time.sleep(poll)

    def cancel(self, run_id: str) -> None:
        self._call("POST", f"pipelineruns/{run_id}/cancel")

    def activity_runs(self, run_id: str, *, days: int = 7) -> list:
        now = _dt.datetime.now(_dt.timezone.utc)
        body = {"lastUpdatedAfter": (now - _dt.timedelta(days=days)).isoformat(),
                "lastUpdatedBefore": (now + _dt.timedelta(days=1)).isoformat()}
        res = self._call("POST", f"pipelineruns/{run_id}/queryActivityruns", body)
        return [{"activity": a.get("activityName"), "type": a.get("activityType"), "status": a.get("status"),
                 "duration_ms": a.get("durationInMs"), "error": (a.get("error") or {}).get("message") or None,
                 "output": a.get("output")} for a in res.get("value", [])]
