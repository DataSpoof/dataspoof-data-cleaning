"""Data lineage with OpenLineage: which data went into which job, and what it produced.

OpenLineage is the open standard read by Marquez, DataHub, Atlan, Astronomer, Microsoft Purview,
Google Dataplex and others. Every pipeline step here can report what it read and wrote:

>>> lin = Lineage.from_env()                       # OPENLINEAGE_URL / _API_KEY / _NAMESPACE, or a file
>>> with lin.run("train_churn", inputs=[dataset("warehouse.analytics.customers", train_df)],
...              outputs=[dataset("models/churn/v4")]) as run:
...     model = ac.AutoML("churned").fit(train_df)  # START, then COMPLETE (or FAIL with the error)
>>> lin.emit_tracked_run(tracker_run)              # an existing Tracker run -> lineage events
>>> lin.emit_dataset_version(repo.get("customers")) # a DataRepository version (with its version id)
>>> graph = LineageGraph.from_file("lineage.jsonl")
>>> graph.upstream("models/churn/v4")              # everything the model depends on, transitively

Datasets carry the OpenLineage ``schema`` facet (columns and types), ``dataQualityMetrics``
(row count, nulls and distinct values per column) and, for versioned data, the ``version`` facet.
Transports: HTTP (``/api/v1/lineage``, Bearer API key), a JSONL file, or several at once.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import json
import os
import threading
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import polars as pl

PRODUCER = "https://github.com/DataSpoof/dataspoof-mlpilot"
SCHEMA_URL = "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
_FACET = "https://openlineage.io/spec/facets/1-0-1/{}.json#/$defs/{}"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _facet(kind: str, schema: str, /, **fields) -> dict:
    return {"_producer": PRODUCER, "_schemaURL": _FACET.format(kind, schema), **fields}


def _split_name(name: str, namespace: str | None) -> tuple:
    """``s3://bucket/key`` -> (``s3://bucket``, ``key``); otherwise the given namespace."""
    if namespace:
        return namespace, name
    if "://" in name:
        scheme, rest = name.split("://", 1)
        bucket, _, key = rest.partition("/")
        return f"{scheme}://{bucket}", key or "/"
    return os.environ.get("OPENLINEAGE_DATASET_NAMESPACE", "mlpilot"), name


def dataset(name: str, df=None, *, namespace: str | None = None, version: str | None = None,
            stats: bool = True, description: str | None = None) -> dict:
    """An OpenLineage dataset. With ``df``: schema facet and (``stats``) data-quality metrics."""
    ns, nm = _split_name(name, namespace)
    facets: dict = {}
    if df is not None:
        frame = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        facets["schema"] = _facet("SchemaDatasetFacet", "SchemaDatasetFacet",
                                  fields=[{"name": c, "type": str(t)} for c, t in frame.schema.items()])
        if stats:
            cols = {}
            for c in frame.columns[:500]:
                s = frame[c]
                m: dict = {"nullCount": int(s.null_count()), "distinctCount": int(s.n_unique())}
                if s.dtype.is_numeric() and s.drop_nulls().len():
                    m.update(min=float(s.min()), max=float(s.max()))  # type: ignore[arg-type]
                cols[c] = m
            facets["dataQualityMetrics"] = _facet("DataQualityMetricsInputDatasetFacet",
                                                  "DataQualityMetricsInputDatasetFacet",
                                                  rowCount=frame.height, columnMetrics=cols)
    if version:
        facets["version"] = _facet("DatasetVersionDatasetFacet", "DatasetVersionDatasetFacet", datasetVersion=str(version))
    if description:
        facets["documentation"] = _facet("DocumentationDatasetFacet", "DocumentationDatasetFacet",
                                         description=description)
    if ns.startswith(("s3://", "gs://", "abfss://", "az://", "file://", "postgres://", "snowflake://")):
        facets["dataSource"] = _facet("DatasourceDatasetFacet", "DatasourceDatasetFacet", name=ns, uri=ns)
    return {"namespace": ns, "name": nm, "facets": facets}


# ── transports ──────────────────────────────────────────────────────────────
class HttpTransport:
    """POSTs events to an OpenLineage endpoint (Marquez: ``http://marquez:5000``)."""

    def __init__(self, url: str, *, api_key: str | None = None, endpoint: str = "api/v1/lineage", timeout: float = 5.0):
        from .secretstore import resolve
        self.url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
        self.api_key, self.timeout = resolve(api_key), timeout

    def emit(self, event: dict) -> None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(event, default=str).encode(), headers=headers,
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status >= 300:                    # pragma: no cover - urllib raises for >= 400
                raise RuntimeError(f"lineage endpoint answered {r.status}")


class FileTransport:
    """Appends events to a JSONL file (read it back with :class:`LineageGraph`)."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def emit(self, event: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")


class Lineage:
    """Emits OpenLineage run events. A transport failing never breaks the job (it is logged as a
    warning) unless ``strict=True``."""

    def __init__(self, *transports, namespace: str | None = None, strict: bool = False):
        self.transports = list(transports)
        self.namespace = namespace or os.environ.get("OPENLINEAGE_NAMESPACE", "mlpilot")
        self.strict = strict
        self.errors: list = []

    @classmethod
    def from_env(cls, *, file=None, strict: bool = False) -> "Lineage":
        """HTTP from ``OPENLINEAGE_URL`` (+ ``OPENLINEAGE_API_KEY``, ``OPENLINEAGE_ENDPOINT``) and/or
        a JSONL file (``file=`` or ``OPENLINEAGE_FILE``)."""
        ts: list = []
        if os.environ.get("OPENLINEAGE_URL"):
            ts.append(HttpTransport(os.environ["OPENLINEAGE_URL"], api_key=os.environ.get("OPENLINEAGE_API_KEY"),
                                    endpoint=os.environ.get("OPENLINEAGE_ENDPOINT", "api/v1/lineage")))
        path = file or os.environ.get("OPENLINEAGE_FILE")
        if path:
            ts.append(FileTransport(path))
        return cls(*ts, strict=strict)

    def emit(self, event: dict) -> dict:
        for t in self.transports:
            try:
                t.emit(event)
            except Exception as exc:
                if self.strict:
                    raise
                import warnings

                from .core import ServiceWarning
                self.errors.append(f"{type(t).__name__}: {exc}")
                warnings.warn(f"Lineage event not delivered ({type(t).__name__}): {exc}", ServiceWarning, stacklevel=3)
        return event

    def event(self, event_type: str, job: str, run_id: str, *, inputs=(), outputs=(), run_facets=None,
              job_facets=None, event_time: str | None = None) -> dict:
        return {"eventType": event_type, "eventTime": event_time or _now(), "producer": PRODUCER,
                "schemaURL": SCHEMA_URL,
                "run": {"runId": run_id, "facets": run_facets or {}},
                "job": {"namespace": self.namespace, "name": job,
                        "facets": {"jobType": _facet("JobTypeJobFacet", "JobTypeJobFacet", processingType="BATCH",
                                                     integration="MLPILOT", jobType="JOB"),
                                   **(job_facets or {})}},
                "inputs": list(inputs), "outputs": list(outputs)}

    @contextlib.contextmanager
    def run(self, job: str, *, inputs=(), outputs=(), description: str | None = None, parent: dict | None = None):
        """START on entry, COMPLETE on success, FAIL (with the error) when an exception escapes.
        Outputs can be added during the run: ``run.outputs.append(dataset(...))``."""
        rid = str(uuid.uuid4())
        state = _RunState(rid, list(inputs), list(outputs))
        job_facets = {"documentation": _facet("DocumentationJobFacet", "DocumentationJobFacet", description=description)} \
            if description else None
        run_facets = {"parent": _facet("ParentRunFacet", "ParentRunFacet", run={"runId": parent["run_id"]},
                                       job={"namespace": parent.get("namespace", self.namespace),
                                            "name": parent["job"]})} if parent else None
        self.emit(self.event("START", job, rid, inputs=state.inputs, outputs=state.outputs, job_facets=job_facets,
                             run_facets=run_facets))
        try:
            yield state
        except BaseException as exc:
            err = _facet("ErrorMessageRunFacet", "ErrorMessageRunFacet", message=str(exc)[:2000],
                         programmingLanguage="python", stackTrace="".join(traceback.format_exc())[-8000:])
            self.emit(self.event("FAIL", job, rid, inputs=state.inputs, outputs=state.outputs, job_facets=job_facets,
                                 run_facets={**(run_facets or {}), "errorMessage": err}))
            raise
        # job facets are repeated: consumers keep the latest event's version of them
        self.emit(self.event("COMPLETE", job, rid, inputs=state.inputs, outputs=state.outputs, job_facets=job_facets,
                             run_facets=run_facets))

    # ── adapters for the library's own records ──────────────────────────────
    def emit_tracked_run(self, run, *, job: str | None = None, model_namespace: str = "mlpilot-models") -> str:
        """Turn a finished :class:`~mlpilot.registry.Run` into START + COMPLETE/FAIL events:
        its logged datasets are inputs (by fingerprint), its artifacts are outputs."""
        meta = run.to_dict()
        rid = str(uuid.UUID(run.id)) if len(run.id) == 32 else str(uuid.uuid4())
        inputs = [{"namespace": os.environ.get("OPENLINEAGE_DATASET_NAMESPACE", "mlpilot"), "name": d["name"],
                   "facets": {"schema": _facet("SchemaDatasetFacet", "SchemaDatasetFacet",
                                               fields=[{"name": c, "type": t} for c, t in (d.get("schema") or {}).items()]),
                              "version": _facet("DatasetVersionDatasetFacet", "DatasetVersionDatasetFacet",
                                                datasetVersion=d.get("content_hash", "")),
                              "dataQualityMetrics": _facet("DataQualityMetricsInputDatasetFacet",
                                                           "DataQualityMetricsInputDatasetFacet",
                                                           rowCount=d.get("rows"), columnMetrics={})}}
                  for d in run.datasets if d.get("role") != "output"]
        outputs = [{"namespace": model_namespace, "name": f"{run.experiment}/{run.name}/{p.name}", "facets": {}}
                   for p in sorted(run.artifacts_dir.iterdir())] if run.artifacts_dir.exists() else []
        name = job or f"{run.experiment}.{run.name}"
        facets = {"mlpilot_run": {"_producer": PRODUCER, "_schemaURL": f"{PRODUCER}#run",
                                            "trackerRunId": run.id, "params": meta.get("params"),
                                            "metrics": meta.get("metrics")}}
        self.emit(self.event("START", name, rid, inputs=inputs, run_facets=facets, event_time=_iso(meta.get("started_at"))))
        final = "FAIL" if run.status == "failed" else "COMPLETE"
        self.emit(self.event(final, name, rid, inputs=inputs, outputs=outputs, run_facets=facets,
                             event_time=_iso(meta.get("ended_at"))))
        return rid

    def emit_dataset_version(self, version, *, job: str | None = None, namespace: str | None = None) -> str:
        """A :class:`~mlpilot.versioning.DatasetVersion` commit as a job that wrote it
        (its ``source`` becomes the input when given)."""
        rid = str(uuid.uuid4())
        out = {"namespace": namespace or os.environ.get("OPENLINEAGE_DATASET_NAMESPACE", "mlpilot"),
               "name": version.name,
               "facets": {"schema": _facet("SchemaDatasetFacet", "SchemaDatasetFacet",
                                           fields=[{"name": c, "type": t} for c, t in version.schema.items()]),
                          "version": _facet("DatasetVersionDatasetFacet", "DatasetVersionDatasetFacet",
                                            datasetVersion=f"v{version.version}:{version.content_hash[:16]}")}}
        inputs = [dataset(version.source, stats=False)] if version.source else []
        name = job or f"commit.{version.name}"
        self.emit(self.event("START", name, rid, inputs=inputs, outputs=[out]))
        self.emit(self.event("COMPLETE", name, rid, inputs=inputs, outputs=[out]))
        return rid


@dataclasses.dataclass
class _RunState:
    run_id: str
    inputs: list
    outputs: list


def _iso(value) -> str | None:
    if not value:
        return None
    s = str(value)
    return s if s.endswith("Z") or "+" in s[10:] else s + "Z"


# ── reading lineage back ────────────────────────────────────────────────────
class LineageGraph:
    """Dataset -> job -> dataset edges from OpenLineage events (COMPLETE events count)."""

    def __init__(self, events):
        self.edges: set = set()          # (input dataset, job, output dataset)
        self.jobs: dict = {}             # job -> {"inputs", "outputs", "last_run", "state"}
        for e in events:
            job = f"{e['job']['namespace']}:{e['job']['name']}"
            ins = [f"{d['namespace']}:{d['name']}" for d in e.get("inputs", [])]
            outs = [f"{d['namespace']}:{d['name']}" for d in e.get("outputs", [])]
            j = self.jobs.setdefault(job, {"inputs": set(), "outputs": set(), "last_run": None, "state": None})
            j["inputs"].update(ins)
            j["outputs"].update(outs)
            j["last_run"], j["state"] = e["run"]["runId"], e["eventType"]
            if e["eventType"] == "COMPLETE":
                for i in ins:
                    for o in outs:
                        self.edges.add((i, job, o))

    @classmethod
    def from_file(cls, path) -> "LineageGraph":
        events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(events)

    def _key(self, name: str) -> str:
        if ":" in name and not name.startswith(("s3://", "gs://")):
            return name
        candidates = {n for e in self.edges for n in (e[0], e[2]) if n.endswith(":" + name) or n == name
                      or n.replace("://", "~").split(":", 1)[-1] == name}
        return next(iter(candidates), name)

    def upstream(self, name: str) -> list:
        """Every dataset ``name`` was (transitively) derived from."""
        target, seen, stack = self._key(name), set(), [self._key(name)]
        while stack:
            cur = stack.pop()
            for i, _, o in self.edges:
                if o == cur and i not in seen:
                    seen.add(i)
                    stack.append(i)
        seen.discard(target)
        return sorted(seen)

    def downstream(self, name: str) -> list:
        """Every dataset derived (transitively) from ``name`` - what breaks if it is wrong."""
        source, seen, stack = self._key(name), set(), [self._key(name)]
        while stack:
            cur = stack.pop()
            for i, _, o in self.edges:
                if i == cur and o not in seen:
                    seen.add(o)
                    stack.append(o)
        seen.discard(source)
        return sorted(seen)

    def to_frame(self) -> pl.DataFrame:
        rows = [{"input": i, "job": j, "output": o} for i, j, o in sorted(self.edges)]
        return pl.DataFrame(rows, schema={"input": pl.String, "job": pl.String, "output": pl.String})
