# Feature store: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Feature store guide](../guide/feature-store.md).

**On this page:**

- [`mlpilot.feature_store`](#mlpilotfeature_store): `Entity`, `FeatureView`, `FeatureStore`

---

## `mlpilot.feature_store`

<sub>Source: [`mlpilot/feature_store.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/feature_store.py)</sub>

Feature store: define features once, reuse them for training and serving without leakage.

```python
fs = FeatureStore("feature_repo")
fs.apply(Entity("customer", join_keys=["customer_id"]),
         FeatureView("customer_stats", entities=["customer"], timestamp="as_of",
                     features=["orders_30d", "spend_30d", "days_since_order"], ttl_days=30))
fs.ingest("customer_stats", daily_snapshot)                     # offline history + online latest
train = fs.get_historical_features(labels, ["customer_stats:*"]) # point-in-time correct
fs.get_online_features(["customer_stats:spend_30d"], [{"customer_id": 42}])   # serving lookup
```

* **Offline store** — append-only Parquet per feature view (full history, for training).
* **Point-in-time joins** — each training row gets the feature values as they were *at that
  row's timestamp* (an as-of join), never later values, so features can't leak the label.
  `ttl_days` ignores values older than that.
* **Online store** — the latest value per entity in SQLite (one file, no server) for low-latency
  lookups at prediction time; `materialize` rebuilds it from the offline history.
* A registry (JSON) records entities, views, owners, descriptions and schemas; re-applying a view
  with an incompatible schema is refused.

### class `Entity`

```python
class Entity:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `join_keys` | `list` | required |
| `description` | `str` | `''` |

### class `FeatureView`

```python
class FeatureView:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `entities` | `list` | required |
| `features` | `list` | required |
| `timestamp` | `str` | required |
| `ttl_days` | `float \| None` | `None` |
| `description` | `str` | `''` |
| `owner` | `str \| None` | `None` |
| `tags` | `dict` | `dict()` |

### class `FeatureStore`

```python
class FeatureStore:
```

**Constructor**

```python
FeatureStore(root='feature_repo')
```

#### `FeatureStore.apply()`

```python
apply(*objects) -> None
```

Register or update entities and feature views (idempotent).

#### `FeatureStore.view()`

```python
view(name: str) -> FeatureView
```

#### `FeatureStore.keys_of()`

```python
keys_of(view: FeatureView) -> list
```

#### `FeatureStore.list_views()`

```python
list_views() -> pl.DataFrame
```

#### `FeatureStore.ingest()`

```python
ingest(view_name: str, df, *, online: bool=True) -> int
```

Append feature rows (entity keys + timestamp + features) to the offline history, and
update the online store with each entity's newest values.

#### `FeatureStore.get_historical_features()`

```python
get_historical_features(entity_df, features: list, *, timestamp: str='event_timestamp', full_names: bool=False) -> pl.DataFrame
```

Training data: `entity_df` (keys + `timestamp` + e.g. labels) with feature values as
of each row's timestamp. `features`: `["view:feature", "view:*", ...]`.

#### `FeatureStore.get_online_features()`

```python
get_online_features(features: list, entity_rows, *, full_names: bool=False, now_ts: _dt.datetime | None=None) -> pl.DataFrame
```

Latest feature values for the given entities (one output row per input row). Values
older than the view's `ttl_days` are returned as null.

#### `FeatureStore.materialize()`

```python
materialize(view_name: str, *, end: _dt.datetime | None=None) -> int
```

Rebuild the online store for a view from the offline history (as of `end`).

#### `FeatureStore.feature_stats()`

```python
feature_stats(view_name: str) -> dict
```
