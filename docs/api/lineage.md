# Lineage: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Lineage guide](../guide/lineage.md).

**On this page:**

- [`mlpilot.lineage`](#mlpilotlineage): `dataset`, `HttpTransport`, `FileTransport`, `Lineage`, `LineageGraph`

---

## `mlpilot.lineage`

<sub>Source: [`mlpilot/lineage.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/lineage.py)</sub>

Data lineage with OpenLineage: which data went into which job, and what it produced.

OpenLineage is the open standard read by Marquez, DataHub, Atlan, Astronomer, Microsoft Purview,
Google Dataplex and others. Every pipeline step here can report what it read and wrote:

```python
lin = Lineage.from_env()                       # OPENLINEAGE_URL / _API_KEY / _NAMESPACE, or a file
with lin.run("train_churn", inputs=[dataset("warehouse.analytics.customers", train_df)],
             outputs=[dataset("models/churn/v4")]) as run:
    model = ac.AutoML("churned").fit(train_df)  # START, then COMPLETE (or FAIL with the error)
lin.emit_tracked_run(tracker_run)              # an existing Tracker run -> lineage events
lin.emit_dataset_version(repo.get("customers")) # a DataRepository version (with its version id)
graph = LineageGraph.from_file("lineage.jsonl")
graph.upstream("models/churn/v4")              # everything the model depends on, transitively
```

Datasets carry the OpenLineage `schema` facet (columns and types), `dataQualityMetrics`
(row count, nulls and distinct values per column) and, for versioned data, the `version` facet.
Transports: HTTP (`/api/v1/lineage`, Bearer API key), a JSONL file, or several at once.

### `dataset()`

```python
def dataset(name: str, df=None, *, namespace: str | None=None, version: str | None=None, stats: bool=True, description: str | None=None) -> dict
```

An OpenLineage dataset. With `df`: schema facet and (`stats`) data-quality metrics.

### class `HttpTransport`

```python
class HttpTransport:
```

POSTs events to an OpenLineage endpoint (Marquez: `http://marquez:5000`).

**Constructor**

```python
HttpTransport(url: str, *, api_key: str | None=None, endpoint: str='api/v1/lineage', timeout: float=5.0)
```

#### `HttpTransport.emit()`

```python
emit(event: dict) -> None
```

### class `FileTransport`

```python
class FileTransport:
```

Appends events to a JSONL file (read it back with `LineageGraph`).

**Constructor**

```python
FileTransport(path)
```

#### `FileTransport.emit()`

```python
emit(event: dict) -> None
```

### class `Lineage`

```python
class Lineage:
```

Emits OpenLineage run events. A transport failing never breaks the job (it is logged as a
warning) unless `strict=True`.

**Constructor**

```python
Lineage(*transports, namespace: str | None=None, strict: bool=False)
```

#### `Lineage.from_env()` *(classmethod)*

```python
from_env(*, file=None, strict: bool=False) -> 'Lineage'
```

HTTP from `OPENLINEAGE_URL` (+ `OPENLINEAGE_API_KEY`, `OPENLINEAGE_ENDPOINT`) and/or
a JSONL file (`file=` or `OPENLINEAGE_FILE`).

#### `Lineage.emit()`

```python
emit(event: dict) -> dict
```

#### `Lineage.event()`

```python
event(event_type: str, job: str, run_id: str, *, inputs=(), outputs=(), run_facets=None, job_facets=None, event_time: str | None=None) -> dict
```

#### `Lineage.run()`

```python
run(job: str, *, inputs=(), outputs=(), description: str | None=None, parent: dict | None=None)
```

START on entry, COMPLETE on success, FAIL (with the error) when an exception escapes.
Outputs can be added during the run: `run.outputs.append(dataset(...))`.

#### `Lineage.emit_tracked_run()`

```python
emit_tracked_run(run, *, job: str | None=None, model_namespace: str='mlpilot-models') -> str
```

Turn a finished `Run` into START + COMPLETE/FAIL events:
its logged datasets are inputs (by fingerprint), its artifacts are outputs.

#### `Lineage.emit_dataset_version()`

```python
emit_dataset_version(version, *, job: str | None=None, namespace: str | None=None) -> str
```

A `DatasetVersion` commit as a job that wrote it
(its `source` becomes the input when given).

### class `LineageGraph`

```python
class LineageGraph:
```

Dataset -> job -> dataset edges from OpenLineage events (COMPLETE events count).

**Constructor**

```python
LineageGraph(events)
```

#### `LineageGraph.from_file()` *(classmethod)*

```python
from_file(path) -> 'LineageGraph'
```

#### `LineageGraph.upstream()`

```python
upstream(name: str) -> list
```

Every dataset `name` was (transitively) derived from.

#### `LineageGraph.downstream()`

```python
downstream(name: str) -> list
```

Every dataset derived (transitively) from `name` - what breaks if it is wrong.

#### `LineageGraph.to_frame()`

```python
to_frame() -> pl.DataFrame
```
