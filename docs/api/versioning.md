# Data versioning: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Data versioning guide](../guide/versioning.md).

**On this page:**

- [`mlpilot.versioning`](#mlpilotversioning): `DatasetVersion`, `DatasetDiff`, `DataRepository`

---

## `mlpilot.versioning`

<sub>Source: [`mlpilot/versioning.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/versioning.py)</sub>

Data version control: commit, tag, diff and restore datasets — like git, for DataFrames.

```python
repo = DataRepository("data_repo")                      # a folder (or a shared drive)
v1 = repo.commit(df, "customers", message="initial extract")
v2 = repo.commit(df2, "customers", message="Q2 refresh", tags=["gold"])
repo.log("customers")                                   # every version: who, when, rows, hash
repo.diff("customers", 1, 2, key="customer_id")         # schema changes + added/removed/changed rows
old = repo.checkout("customers", version=1)             # exact bytes back (checksum-verified)
repo.checkout("customers", tag="gold")
```

Each version is stored once as compressed Parquet under its content hash, so committing the
same data again doesn't duplicate it, and any tampering with a stored file is detected on
checkout. Commits record the author, message, parent, tags, optional source/lineage and the
tracking run that produced or used it (`run=`).

### class `DatasetVersion`

```python
class DatasetVersion:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `version` | `int` | required |
| `content_hash` | `str` | required |
| `schema_hash` | `str` | required |
| `rows` | `int` | required |
| `columns` | `int` | required |
| `schema` | `dict` | required |
| `created_at` | `str` | required |
| `author` | `str` | required |
| `message` | `str` | required |
| `parent` | `int \| None` | required |
| `tags` | `list` | required |
| `source` | `str \| None` | required |
| `run_id` | `str \| None` | required |
| `file` | `str` | required |
| `bytes` | `int` | required |
| `file_sha256` | `str` | `''` |

#### `DatasetVersion.ref` *(property)*

```python
ref: str
```

`name@vN#hash` - a stable reference to log with experiments.

### class `DatasetDiff`

```python
class DatasetDiff:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `a` | `int` | required |
| `b` | `int` | required |
| `rows_a` | `int` | required |
| `rows_b` | `int` | required |
| `added_columns` | `list` | required |
| `removed_columns` | `list` | required |
| `changed_types` | `dict` | required |
| `added_rows` | `int \| None` | `None` |
| `removed_rows` | `int \| None` | `None` |
| `changed_rows` | `int \| None` | `None` |
| `unchanged_rows` | `int \| None` | `None` |
| `changed_columns` | `dict` | `dict()` |

#### `DatasetDiff.identical` *(property)*

```python
identical: bool
```

#### `DatasetDiff.summary()`

```python
summary() -> str
```

### class `DataRepository`

```python
class DataRepository:
```

**Constructor**

```python
DataRepository(root='data_repo')
```

#### `DataRepository.commit()`

```python
commit(df, name: str, *, message: str='', tags=None, source: str | None=None, run=None, allow_unchanged: bool=False) -> DatasetVersion
```

Store `df` as the next version of `name`. If it is identical to the latest version,
that version is returned instead (unless `allow_unchanged`).

#### `DataRepository.tag()`

```python
tag(name: str, version: int, tag: str) -> DatasetVersion
```

Point `tag` (e.g. `"gold"`, `"training-2024Q2"`) at `version` (moved if it existed).

#### `DataRepository.get()`

```python
get(name: str, version: int | None=None, tag: str | None=None) -> DatasetVersion
```

#### `DataRepository.checkout()`

```python
checkout(name: str, version: int | None=None, *, tag: str | None=None, run=None, verify: bool=True, lazy: bool=False)
```

The exact data of a version (latest by default). The content hash is re-checked, so a
modified or corrupted stored file raises instead of silently returning different data.

#### `DataRepository.log()`

```python
log(name: str) -> pl.DataFrame
```

#### `DataRepository.datasets()`

```python
datasets() -> pl.DataFrame
```

#### `DataRepository.diff()`

```python
diff(name: str, a: int, b: int, *, key=None) -> DatasetDiff
```

Compare two versions: schema changes and, with `key` (column(s) identifying a row),
rows added / removed / changed and which columns changed most. Without a key, rows are
compared as whole records.

#### `DataRepository.gc()`

```python
gc() -> int
```

Delete stored objects no version refers to any more; returns how many were removed.

#### `DataRepository.export()`

```python
export(name: str, version: int, dest) -> Path
```

Copy a version's Parquet file out (e.g. to hand to another team).
