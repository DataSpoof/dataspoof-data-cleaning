"""Feature store: define features once, reuse them for training and serving without leakage.

>>> fs = FeatureStore("feature_repo")
>>> fs.apply(Entity("customer", join_keys=["customer_id"]),
...          FeatureView("customer_stats", entities=["customer"], timestamp="as_of",
...                      features=["orders_30d", "spend_30d", "days_since_order"], ttl_days=30))
>>> fs.ingest("customer_stats", daily_snapshot)                     # offline history + online latest
>>> train = fs.get_historical_features(labels, ["customer_stats:*"]) # point-in-time correct
>>> fs.get_online_features(["customer_stats:spend_30d"], [{"customer_id": 42}])   # serving lookup

* **Offline store** — append-only Parquet per feature view (full history, for training).
* **Point-in-time joins** — each training row gets the feature values as they were *at that
  row's timestamp* (an as-of join), never later values, so features can't leak the label.
  ``ttl_days`` ignores values older than that.
* **Online store** — the latest value per entity in SQLite (one file, no server) for low-latency
  lookups at prediction time; ``materialize`` rebuilds it from the offline history.
* A registry (JSON) records entities, views, owners, descriptions and schemas; re-applying a view
  with an incompatible schema is refused.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import inspect
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import polars as pl

from .registry._store import FileLock, check_name, now, read_json, write_json


@dataclasses.dataclass
class Entity:
    name: str
    join_keys: list
    description: str = ""

    def __post_init__(self):
        check_name(self.name, "entity name")
        if isinstance(self.join_keys, str):
            self.join_keys = [self.join_keys]
        if not self.join_keys:
            raise ValueError("An entity needs at least one join key")


@dataclasses.dataclass
class FeatureView:
    name: str
    entities: list
    features: list
    timestamp: str
    ttl_days: float | None = None
    description: str = ""
    owner: str | None = None
    tags: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        check_name(self.name, "feature view name")
        if isinstance(self.entities, str):
            self.entities = [self.entities]
        if not self.features:
            raise ValueError("A feature view needs at least one feature")


# both sides are sorted on the time column before the join; newer Polars warns that it can't
# verify that when grouping with ``by=``
_ASOF_KW: dict[str, Any] = {"check_sortedness": False} if "check_sortedness" in inspect.signature(pl.DataFrame.join_asof).parameters \
    else {}


def _ts(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Datetime("us"))


class FeatureStore:
    def __init__(self, root="feature_repo"):
        self.root = Path(root)
        (self.root / "offline").mkdir(parents=True, exist_ok=True)
        self._registry = self.root / "registry.json"
        self._local = threading.local()

    # ── registry ────────────────────────────────────────────────────────────
    def _reg(self) -> dict:
        return read_json(self._registry) if self._registry.exists() else {"entities": {}, "views": {}}

    def apply(self, *objects) -> None:
        """Register or update entities and feature views (idempotent)."""
        with FileLock(self.root / ".lock"):
            reg = self._reg()
            for obj in objects:
                if isinstance(obj, Entity):
                    reg["entities"][obj.name] = dataclasses.asdict(obj)
            for obj in objects:
                if isinstance(obj, FeatureView):
                    missing = [e for e in obj.entities if e not in reg["entities"]]
                    if missing:
                        raise ValueError(f"Unknown entities {missing}; apply() them first")
                    old = reg["views"].get(obj.name)
                    if old and old.get("schema"):
                        removed = [f for f in old["features"] if f not in obj.features]
                        if removed and self._has_data(obj.name):
                            raise ValueError(f"View '{obj.name}' already has data; removing features {removed} "
                                             "would break consumers - create a new view (e.g. a _v2) instead")
                    reg["views"][obj.name] = {**dataclasses.asdict(obj), "schema": (old or {}).get("schema"),
                                              "created_at": (old or {}).get("created_at", now()), "updated_at": now()}
                elif not isinstance(obj, Entity):
                    raise TypeError(f"Can't apply {type(obj).__name__}")
            write_json(self._registry, reg)

    def view(self, name: str) -> FeatureView:
        v = self._reg()["views"].get(name)
        if v is None:
            raise KeyError(f"No feature view {name!r}")
        return FeatureView(**{k: v[k] for k in ("name", "entities", "features", "timestamp", "ttl_days", "description",
                                                 "owner", "tags")})

    def keys_of(self, view: FeatureView) -> list:
        ents = self._reg()["entities"]
        return [k for e in view.entities for k in ents[e]["join_keys"]]

    def list_views(self) -> pl.DataFrame:
        rows = []
        for name, v in self._reg()["views"].items():
            st = self.feature_stats(name) if self._has_data(name) else {}
            rows.append({"view": name, "entities": ",".join(v["entities"]), "features": len(v["features"]),
                         "owner": v.get("owner"), "rows": st.get("rows", 0), "latest": st.get("latest"),
                         "description": v.get("description")})
        return pl.DataFrame(rows, infer_schema_length=None)

    # ── offline store ───────────────────────────────────────────────────────
    def _has_data(self, name: str) -> bool:
        d = self.root / "offline" / name
        return d.exists() and any(d.glob("*.parquet"))

    def _offline(self, name: str) -> pl.LazyFrame:
        if not self._has_data(name):
            raise ValueError(f"Feature view '{name}' has no data yet; ingest() some")
        return pl.scan_parquet(str(self.root / "offline" / name / "*.parquet"))

    def ingest(self, view_name: str, df, *, online: bool = True) -> int:
        """Append feature rows (entity keys + timestamp + features) to the offline history, and
        update the online store with each entity's newest values."""
        view = self.view(view_name)
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        keys = self.keys_of(view)
        need = keys + [view.timestamp] + list(view.features)
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for '{view_name}': {missing}")
        out = df.select(need).with_columns(_ts(view.timestamp)).drop_nulls(keys + [view.timestamp])
        if out.height == 0:
            return 0
        schema = {c: str(t) for c, t in out.schema.items()}
        with FileLock(self.root / ".lock"):
            reg = self._reg()
            known = reg["views"][view_name].get("schema")
            if known:
                bad = {c: f"{known[c]} -> {schema[c]}" for c in schema if c in known and known[c] != schema[c]
                       and "Null" not in (known[c], schema[c])}
                if bad:
                    raise ValueError(f"Type changes for '{view_name}': {bad}; cast the data or create a new view")
            else:
                reg["views"][view_name]["schema"] = schema
                write_json(self._registry, reg)
            if known:
                out = out.with_columns([pl.col(c).cast(_dtype(known[c]), strict=False) for c in out.columns
                                        if c in known and schema[c] != known[c]])
            d = self.root / "offline" / view_name
            d.mkdir(parents=True, exist_ok=True)
            part = d / f"{_dt.datetime.now(_dt.timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
            out.write_parquet(part.with_suffix(".partial"), compression="zstd")
            part.with_suffix(".partial").replace(part)
        if online:
            self._online_upsert(view, out)
        return out.height

    def get_historical_features(self, entity_df, features: list, *, timestamp: str = "event_timestamp",
                                full_names: bool = False) -> pl.DataFrame:
        """Training data: ``entity_df`` (keys + ``timestamp`` + e.g. labels) with feature values as
        of each row's timestamp. ``features``: ``["view:feature", "view:*", ...]``."""
        df = entity_df if isinstance(entity_df, pl.DataFrame) else pl.from_pandas(entity_df)
        if timestamp not in df.columns:
            raise ValueError(f"entity_df needs the '{timestamp}' column (when each row's prediction was made)")
        out = df.with_row_index("_row").with_columns(_ts(timestamp).alias("_ts"))
        for view_name, feats in _parse_refs(features).items():
            view = self.view(view_name)
            feats = list(view.features) if feats == ["*"] else feats
            bad = [f for f in feats if f not in view.features]
            if bad:
                raise ValueError(f"'{view_name}' has no features {bad}")
            keys = self.keys_of(view)
            missing = [k for k in keys if k not in out.columns]
            if missing:
                raise ValueError(f"entity_df lacks join keys {missing} for '{view_name}'")
            names = {f: (f"{view_name}__{f}" if full_names else f) for f in feats}
            clash = [n for n in names.values() if n in out.columns]
            if clash:
                raise ValueError(f"Feature names {clash} clash with existing columns; use full_names=True")
            hist = (self._offline(view_name).select(keys + [view.timestamp] + feats).collect()
                    .rename({view.timestamp: "_fts", **names}))
            hist = hist.with_columns([pl.col(k).cast(out.schema[k], strict=False) for k in keys]).sort("_fts")
            left = out.sort("_ts")
            joined = left.join_asof(hist, left_on="_ts", right_on="_fts", by=keys, strategy="backward",
                                    tolerance=_dt.timedelta(days=view.ttl_days) if view.ttl_days else None,
                                    **_ASOF_KW)
            out = joined.drop("_fts")
        return out.sort("_row").drop("_row", "_ts")

    # ── online store ────────────────────────────────────────────────────────
    def _db(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.root / "online.sqlite", timeout=30)
            con.execute("PRAGMA journal_mode=WAL")
            self._local.con = con
        return con

    def _online_upsert(self, view: FeatureView, df: pl.DataFrame):
        keys = self.keys_of(view)
        latest = df.sort(view.timestamp).unique(subset=keys, keep="last")
        table = f"fv_{view.name.replace('-', '_').replace('.', '_')}"
        con = self._db()
        con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" (entity_key TEXT PRIMARY KEY, event_ts REAL, "values" TEXT)')
        rows = []
        for r in latest.iter_rows(named=True):
            ts = r[view.timestamp]
            rows.append((json.dumps([str(r[k]) for k in keys]), ts.replace(tzinfo=_dt.timezone.utc).timestamp(),
                         json.dumps({f: _plain(r[f]) for f in view.features})))
        con.executemany(f'INSERT INTO "{table}" (entity_key, event_ts, "values") VALUES (?, ?, ?) '
                        f'ON CONFLICT(entity_key) DO UPDATE SET event_ts = excluded.event_ts, "values" = excluded."values" '
                        f'WHERE excluded.event_ts >= "{table}".event_ts', rows)
        con.commit()

    def get_online_features(self, features: list, entity_rows, *, full_names: bool = False,
                            now_ts: _dt.datetime | None = None) -> pl.DataFrame:
        """Latest feature values for the given entities (one output row per input row). Values
        older than the view's ``ttl_days`` are returned as null."""
        rows = entity_rows.to_dicts() if isinstance(entity_rows, pl.DataFrame) else list(entity_rows)
        out = [dict(r) for r in rows]
        current = (now_ts or _dt.datetime.now(_dt.timezone.utc)).timestamp()
        con = self._db()
        for view_name, feats in _parse_refs(features).items():
            view = self.view(view_name)
            feats = list(view.features) if feats == ["*"] else feats
            keys = self.keys_of(view)
            table = f"fv_{view.name.replace('-', '_').replace('.', '_')}"
            ek = [json.dumps([str(r.get(k)) for k in keys]) for r in rows]
            found: dict = {}
            try:
                for i in range(0, len(ek), 500):
                    chunk = ek[i:i + 500]
                    q = f'SELECT entity_key, event_ts, "values" FROM "{table}" WHERE entity_key IN ({",".join("?" * len(chunk))})'
                    for key, ts, vals in con.execute(q, chunk):
                        found[key] = (ts, json.loads(vals))
            except sqlite3.OperationalError:              # nothing materialised yet
                pass
            for r, key in zip(out, ek):
                ts, vals = found.get(key, (None, {}))
                fresh = ts is not None and (not view.ttl_days or current - ts <= view.ttl_days * 86400)
                for f in feats:
                    r[f"{view_name}__{f}" if full_names else f] = vals.get(f) if fresh else None
        return pl.DataFrame(out, infer_schema_length=None)

    def materialize(self, view_name: str, *, end: _dt.datetime | None = None) -> int:
        """Rebuild the online store for a view from the offline history (as of ``end``)."""
        view = self.view(view_name)
        hist = self._offline(view_name)
        if end is not None:
            hist = hist.filter(pl.col(view.timestamp) <= pl.lit(end).cast(pl.Datetime("us")))
        df = hist.collect()
        table = f"fv_{view.name.replace('-', '_').replace('.', '_')}"
        self._db().execute(f'DROP TABLE IF EXISTS "{table}"')
        self._online_upsert(view, df)
        return df.select(self.keys_of(view)).unique().height

    def feature_stats(self, view_name: str) -> dict:
        view = self.view(view_name)
        df = self._offline(view_name).collect()
        keys = self.keys_of(view)
        return {"rows": df.height, "entities": df.select(keys).unique().height,
                "earliest": str(df[view.timestamp].min()), "latest": str(df[view.timestamp].max()),
                "null_rate": {f: float(df[f].null_count() / max(df.height, 1)) for f in view.features}}


def _parse_refs(features) -> dict:
    out: dict = {}
    for ref in features:
        if ":" not in ref:
            raise ValueError(f"Feature reference {ref!r} must look like 'view:feature' or 'view:*'")
        view, feat = ref.split(":", 1)
        out.setdefault(view, [])
        if feat == "*":
            out[view] = ["*"]
        elif out[view] != ["*"]:
            out[view].append(feat)
    return out


def _plain(v):
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return v


def _dtype(name: str):
    from .core.pipeline import _parse_dtype
    return _parse_dtype(name) or pl.String
